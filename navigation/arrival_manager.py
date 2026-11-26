"""Destination Arrival Manager and Finite State Machine.

Manages destination approach, speed tapering, controlled stopping, and arrival
confirmation for autonomous vehicle navigation.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Optional, Tuple


class NavigationState(str, enum.Enum):
    """Navigation states for route following and destination arrival."""
    NAVIGATING = "NAVIGATING"
    APPROACHING = "APPROACHING"
    STOPPING = "STOPPING"
    ARRIVED = "ARRIVED"


class ArrivalManager:
    """Finite State Machine for autonomous destination approach and stopping.

    State transitions:
        NAVIGATING -> APPROACHING (when distance <= approach_distance)
        APPROACHING -> STOPPING   (when distance <= arrival_radius)
        STOPPING -> ARRIVED       (when speed <= stop_speed_threshold_kmh for confirm_frames)
        ARRIVED -> locked in ARRIVED (full stop, no further acceleration)
    """

    def __init__(
        self,
        destination: Any,
        cruise_speed_kmh: float = 25.0,
        approach_distance: float = 20.0,
        arrival_radius: float = 2.5,
        deceleration: float = 2.0,
        stop_speed_threshold_kmh: float = 0.5,
        confirm_frames: int = 10,
        min_approach_speed_kmh: float = 4.0,
    ):
        """Initializes the ArrivalManager.

        Args:
            destination: Tuple (x, y[, z]) or Location object with .x, .y
            cruise_speed_kmh: Normal target cruising speed in km/h (> 0).
            approach_distance: Distance in meters to start decelerating (> 0).
            arrival_radius: Radius in meters around destination for stopping zone (> 0).
            deceleration: Deceleration in m/s^2 (> 0).
            stop_speed_threshold_kmh: Max speed in km/h to count as stopped (>= 0).
            confirm_frames: Consecutive stopped frames required to confirm arrival (>= 1).
            min_approach_speed_kmh: Minimum crawling speed while approaching (> 0).
        """
        if cruise_speed_kmh <= 0:
            raise ValueError(f"cruise_speed_kmh must be positive, got {cruise_speed_kmh}")
        if approach_distance <= 0:
            raise ValueError(f"approach_distance must be positive, got {approach_distance}")
        if arrival_radius <= 0:
            raise ValueError(f"arrival_radius must be positive, got {arrival_radius}")
        if deceleration <= 0:
            raise ValueError(f"deceleration must be positive, got {deceleration}")
        if stop_speed_threshold_kmh < 0:
            raise ValueError(f"stop_speed_threshold_kmh cannot be negative, got {stop_speed_threshold_kmh}")
        if confirm_frames < 1:
            raise ValueError(f"confirm_frames must be >= 1, got {confirm_frames}")

        self.destination_x, self.destination_y = self._extract_coords(destination)
        self.cruise_speed_kmh = float(cruise_speed_kmh)
        self.approach_distance = float(approach_distance)
        self.arrival_radius = float(arrival_radius)
        self.deceleration = float(deceleration)
        self.stop_speed_threshold_kmh = float(stop_speed_threshold_kmh)
        self.confirm_frames = int(confirm_frames)
        self.min_approach_speed_kmh = float(min_approach_speed_kmh)

        # FSM state
        self.state = NavigationState.NAVIGATING
        self.target_speed_kmh: float = self.cruise_speed_kmh
        self.current_distance: float = float("inf")
        self.current_speed_kmh: float = 0.0
        self._stopped_frame_count: int = 0

    @staticmethod
    def _extract_coords(location: Any) -> Tuple[float, float]:
        """Extracts (x, y) coordinates from tuple, list, or CARLA Location object."""
        if hasattr(location, "x") and hasattr(location, "y"):
            return float(location.x), float(location.y)
        if isinstance(location, (tuple, list)) and len(location) >= 2:
            return float(location[0]), float(location[1])
        raise ValueError(f"Invalid destination location: {location}")

    def compute_target_speed(self, remaining_distance: float) -> float:
        """Calculates smooth target speed: v = min(v_cruise, sqrt(2 * a * d)) in km/h."""
        if remaining_distance <= 0.0:
            return 0.0
        v_cruise_ms = self.cruise_speed_kmh / 3.6
        v_profile_ms = math.sqrt(2.0 * self.deceleration * remaining_distance)
        v_target_ms = min(v_cruise_ms, v_profile_ms)
        v_target_kmh = v_target_ms * 3.6
        # Keep minimum crawl speed until inside arrival radius
        if remaining_distance > self.arrival_radius:
            v_target_kmh = max(self.min_approach_speed_kmh, v_target_kmh)
        return float(min(self.cruise_speed_kmh, max(0.0, v_target_kmh)))

    def update(
        self,
        current_position: Any,
        current_speed_kmh: float,
        sim_time_s: float = 0.0,
    ) -> Tuple[NavigationState, float]:
        """Updates the Arrival FSM state.

        Args:
            current_position: Current vehicle position (tuple (x, y[, z]) or Location object)
            current_speed_kmh: Current vehicle speed in km/h
            sim_time_s: Simulation timestamp (unused, kept for API compatibility)

        Returns:
            (state, target_speed_kmh)
        """
        self.current_distance = math.hypot(
            self._extract_coords(current_position)[0] - self.destination_x,
            self._extract_coords(current_position)[1] - self.destination_y,
        )
        self.current_speed_kmh = max(0.0, float(current_speed_kmh))

        if self.state == NavigationState.ARRIVED:
            # Locked — stays ARRIVED permanently
            self.target_speed_kmh = 0.0
            return self.state, self.target_speed_kmh

        if self.state == NavigationState.NAVIGATING:
            self.target_speed_kmh = self.cruise_speed_kmh
            if self.current_distance <= self.approach_distance:
                self.state = NavigationState.APPROACHING

        if self.state == NavigationState.APPROACHING:
            self.target_speed_kmh = self.compute_target_speed(self.current_distance)
            if self.current_distance <= self.arrival_radius:
                self.state = NavigationState.STOPPING
                self.target_speed_kmh = 0.0

        if self.state == NavigationState.STOPPING:
            self.target_speed_kmh = 0.0
            if self.current_speed_kmh <= self.stop_speed_threshold_kmh:
                self._stopped_frame_count += 1
                if self._stopped_frame_count >= self.confirm_frames:
                    self.state = NavigationState.ARRIVED
            else:
                # Speed blip: reset counter
                self._stopped_frame_count = 0

        return self.state, self.target_speed_kmh

    def get_control_override(self) -> Optional[dict]:
        """Returns control overrides when stopping or arrived."""
        if self.state == NavigationState.ARRIVED:
            return {
                "throttle": 0.0,
                "steer": 0.0,
                "brake": 1.0,
                "hand_brake": True,
            }
        elif self.state == NavigationState.STOPPING:
            brake_intensity = min(1.0, max(0.8, 0.5 + 0.5 * (self.current_speed_kmh / 10.0)))
            return {
                "throttle": 0.0,
                "steer": 0.0,
                "brake": float(brake_intensity),
                "hand_brake": self.current_speed_kmh <= self.stop_speed_threshold_kmh,
            }
        return None

    def reset(self) -> None:
        """Resets FSM to NAVIGATING state."""
        self.state = NavigationState.NAVIGATING
        self.target_speed_kmh = self.cruise_speed_kmh
        self._stopped_frame_count = 0

    def is_arrived(self) -> bool:
        return self.state == NavigationState.ARRIVED

    def is_stopped(self) -> bool:
        return self.state in (NavigationState.STOPPING, NavigationState.ARRIVED)

    def is_approaching(self) -> bool:
        return self.state == NavigationState.APPROACHING

    def is_navigating(self) -> bool:
        return self.state == NavigationState.NAVIGATING
