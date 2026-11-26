"""GPS and Vehicle Location Tracker.

Tracks vehicle world position, GPS coordinates, distance to destination,
odometry (distance driven), heading, speed, navigation state, and route progress.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from navigation.arrival_manager import NavigationState


class GPSTracker:
    """Provides unified location, GPS, odometry, and navigation progress tracking."""

    def __init__(
        self,
        carla_map: Optional[Any] = None,
        destination: Optional[Any] = None,
        planned_distance_m: float = 0.0,
        # Backward-compatible aliases
        destination_location: Optional[Any] = None,
        planned_route_distance: Optional[float] = None,
    ):
        """Initializes the GPSTracker.

        Args:
            carla_map: Optional CARLA map instance (for GPS coordinate conversion).
            destination: Target destination as (x, y[, z]) tuple or Location object.
            planned_distance_m: Total planned route distance in meters.
            destination_location: Alias for destination.
            planned_route_distance: Alias for planned_distance_m.
        """
        self.carla_map = carla_map

        # Support both argument name styles
        dest = destination or destination_location
        dist = planned_distance_m if planned_route_distance is None else planned_route_distance
        self.planned_distance_m: float = max(0.0, float(dist))

        if dest is not None:
            self._dest_x, self._dest_y = self._extract_xy(dest)
        else:
            self._dest_x, self._dest_y = None, None

        # Position state
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.altitude: float = 0.0

        # Motion state
        self.speed_kmh: float = 0.0
        self.speed_ms: float = 0.0
        self.heading_yaw: float = 0.0
        self.heading_yaw_deg: float = 0.0

        # Navigation state
        self.navigation_state: NavigationState = NavigationState.NAVIGATING

        # Odometry
        self.driven_distance_m: float = 0.0
        self.distance_to_destination_m: float = 0.0
        self.route_progress_pct: float = 0.0

        # Legacy alias
        self.distance_to_destination: float = 0.0
        self.distance_travelled: float = 0.0

        self._previous_location: Optional[Tuple[float, float, float]] = None

    @staticmethod
    def _extract_xy(location: Any) -> Tuple[float, float]:
        if hasattr(location, "x") and hasattr(location, "y"):
            return float(location.x), float(location.y)
        if isinstance(location, (tuple, list)) and len(location) >= 2:
            return float(location[0]), float(location[1])
        raise ValueError(f"Invalid location: {location}")

    def set_destination(self, destination: Any) -> None:
        """Updates the destination coordinate."""
        self._dest_x, self._dest_y = self._extract_xy(destination)

    def set_planned_distance(self, distance_m: float) -> None:
        """Sets the total planned route distance."""
        self.planned_distance_m = max(0.0, float(distance_m))

    def update(
        self,
        current_location: Any,
        current_speed_kmh: float,
        heading_yaw_deg: float = 0.0,
        sim_time_s: float = 0.0,
        navigation_state: NavigationState = NavigationState.NAVIGATING,
    ) -> Dict[str, Any]:
        """Updates tracker state from current vehicle telemetry.

        Args:
            current_location: (x, y[, z]) tuple or Location object.
            current_speed_kmh: Current vehicle speed in km/h.
            heading_yaw_deg: Vehicle heading (yaw) in degrees.
            sim_time_s: Simulation timestamp in seconds.
            navigation_state: Current NavigationState value.

        Returns:
            Dictionary snapshot of current telemetry.
        """
        # --- Extract position ---
        if isinstance(current_location, (tuple, list)):
            self.x = float(current_location[0])
            self.y = float(current_location[1])
            self.z = float(current_location[2]) if len(current_location) > 2 else 0.0
        elif hasattr(current_location, "x"):
            self.x = float(current_location.x)
            self.y = float(current_location.y)
            self.z = float(getattr(current_location, "z", 0.0))
        else:
            raise ValueError(f"Invalid current_location: {current_location}")

        # --- Motion ---
        self.speed_kmh = max(0.0, float(current_speed_kmh))
        self.speed_ms = self.speed_kmh / 3.6
        self.heading_yaw_deg = float(heading_yaw_deg)
        self.heading_yaw = self.heading_yaw_deg
        self.navigation_state = navigation_state

        # --- Odometry ---
        if self._previous_location is not None:
            px, py, pz = self._previous_location
            step = math.sqrt((self.x - px) ** 2 + (self.y - py) ** 2 + (self.z - pz) ** 2)
            if step > 0.001:
                self.driven_distance_m += step
                self.distance_travelled = self.driven_distance_m
        self._previous_location = (self.x, self.y, self.z)

        # --- GPS coordinates ---
        if self.carla_map is not None:
            try:
                from types import SimpleNamespace
                loc = SimpleNamespace(x=self.x, y=self.y, z=self.z)
                geo = self.carla_map.transform_to_geolocation(loc)
                self.latitude = float(geo.latitude)
                self.longitude = float(geo.longitude)
                self.altitude = float(geo.altitude)
            except Exception:
                self._approximate_gps()
        else:
            self._approximate_gps()

        self.current_gps_lat: float = self.latitude
        self.current_gps_lon: float = self.longitude

        # --- Distance to destination ---
        if self._dest_x is not None:
            self.distance_to_destination_m = math.hypot(
                self.x - self._dest_x, self.y - self._dest_y
            )
            self.distance_to_destination = self.distance_to_destination_m
        else:
            self.distance_to_destination_m = 0.0
            self.distance_to_destination = 0.0

        # --- Route progress ---
        if self.planned_distance_m > 0.0:
            self.route_progress_pct = min(
                100.0,
                (self.driven_distance_m / self.planned_distance_m) * 100.0
            )
        else:
            self.route_progress_pct = 0.0

        return self.get_telemetry_dict()

    def _approximate_gps(self) -> None:
        """Approximates GPS via WGS84 projection with CARLA reference origin."""
        base_lat = 49.0
        base_lon = 8.0
        lat = base_lat + (self.y / 111319.49)
        cos_lat = max(0.001, math.cos(math.radians(lat)))
        self.latitude = lat
        self.longitude = base_lon + (self.x / (111319.49 * cos_lat))
        self.altitude = self.z

    def set_navigation_state(self, state: NavigationState) -> None:
        """Explicitly sets the navigation state (used by tests and FSM)."""
        self.navigation_state = state

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns structured telemetry snapshot (new API)."""
        nav = self.navigation_state
        nav_str = str(getattr(nav, "value", nav))
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "altitude": round(self.altitude, 2),
            "speed_kmh": round(self.speed_kmh, 2),
            "speed_ms": round(self.speed_ms, 4),
            "heading_yaw_deg": round(self.heading_yaw_deg, 2),
            "distance_to_destination_m": round(self.distance_to_destination_m, 2),
            "driven_distance_m": round(self.driven_distance_m, 2),
            "route_progress_pct": round(self.route_progress_pct, 1),
            "nav_state": nav_str,
            "is_arrived": nav_str == "ARRIVED",
        }

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Alias of get_telemetry for backward compatibility."""
        return self.get_telemetry()
