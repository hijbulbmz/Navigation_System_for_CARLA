"""Autonomous Route Following Controller and Replanning Engine.

Tracks a planned route with speed-dependent lookahead Pure Pursuit / Stanley steering,
longitudinal speed regulation, destination arrival stopping, and automatic replanning.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np


class NavigationState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    FOLLOWING = "FOLLOWING"
    BLOCKED = "BLOCKED"
    REPLANNING = "REPLANNING"
    ARRIVED = "ARRIVED"
    FAILED = "FAILED"


class RouteFollower:
    """Follows an ordered sequence of CARLA waypoints to a destination."""

    def __init__(
        self,
        target_speed_kmh: float = 25.0,
        arrival_radius_m: float = 2.5,
        replan_off_route_m: float = 4.5,
        replan_cooldown_s: float = 2.0,
    ):
        self.cruise_speed_kmh = max(5.0, float(target_speed_kmh))
        self.arrival_radius_m = max(1.0, float(arrival_radius_m))
        self.replan_off_route_m = max(2.0, float(replan_off_route_m))
        self.replan_cooldown_s = max(0.5, float(replan_cooldown_s))

        self.state = NavigationState.IDLE
        self.route_plan = None
        self.waypoints: List[Any] = []
        self.polyline_2d: List[Tuple[float, float]] = []
        self.destination: Optional[Tuple[float, float]] = None

        self.current_index: int = 0
        self.remaining_distance_m: float = 0.0
        self.off_route_ticks: int = 0
        self.time_since_last_replan: float = 999.0
        self.previous_steer: float = 0.0

    def set_route(self, route_plan: Any) -> None:
        """Sets a newly calculated RoutePlan."""
        if route_plan is None or not route_plan.success or not route_plan.waypoints:
            self.state = NavigationState.FAILED
            self.route_plan = None
            self.waypoints = []
            self.polyline_2d = []
            self.destination = None
            return

        self.route_plan = route_plan
        self.waypoints = route_plan.waypoints
        self.polyline_2d = route_plan.polyline_2d
        self.destination = route_plan.destination_location
        self.current_index = 0
        self.remaining_distance_m = route_plan.total_distance_m
        self.off_route_ticks = 0
        self.state = NavigationState.FOLLOWING

    def clear(self) -> None:
        """Resets the route follower to IDLE."""
        self.state = NavigationState.IDLE
        self.route_plan = None
        self.waypoints = []
        self.polyline_2d = []
        self.destination = None
        self.current_index = 0
        self.remaining_distance_m = 0.0
        self.off_route_ticks = 0

    def step(
        self,
        current_x: float,
        current_y: float,
        current_yaw_deg: float,
        speed_kmh: float,
        delta_s: float = 0.05,
        failsafe_blocked: bool = False,
    ) -> Tuple[float, float, float, bool]:
        """Calculates (steer, throttle, brake, needs_replan) for the current simulation step."""
        self.time_since_last_replan += delta_s

        if self.state in (NavigationState.IDLE, NavigationState.FAILED):
            return 0.0, 0.0, 1.0, False

        if self.state == NavigationState.ARRIVED:
            return 0.0, 0.0, 1.0, False

        # If obstacle emergency stop is active, yield to safety
        if failsafe_blocked:
            self.state = NavigationState.BLOCKED
            return 0.0, 0.0, 1.0, False
        elif self.state == NavigationState.BLOCKED:
            self.state = NavigationState.FOLLOWING

        if not self.polyline_2d or self.destination is None:
            self.state = NavigationState.FAILED
            return 0.0, 0.0, 1.0, False

        # 1. Check distance to final destination
        dist_to_destination = math.sqrt(
            (self.destination[0] - current_x) ** 2
            + (self.destination[1] - current_y) ** 2
        )
        if dist_to_destination <= self.arrival_radius_m:
            if speed_kmh < 3.0:
                self.state = NavigationState.ARRIVED
                print(f"[navigation] ARRIVED at destination ({self.destination[0]:.1f}, {self.destination[1]:.1f})")
                return 0.0, 0.0, 1.0, False
            else:
                # Inside arrival zone: apply full stop brake to halt at destination
                return 0.0, 0.0, 1.0, False

        # 2. Advance progress along the route
        search_window = min(len(self.polyline_2d), self.current_index + 25)
        best_idx = self.current_index
        min_dist_sq = float("inf")
        for i in range(self.current_index, search_window):
            pt = self.polyline_2d[i]
            d_sq = (pt[0] - current_x) ** 2 + (pt[1] - current_y) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_idx = i

        self.current_index = max(self.current_index, best_idx)
        lateral_departure = math.sqrt(min_dist_sq)

        # 3. Check for route departure (needs replan)
        needs_replan = False
        if lateral_departure > self.replan_off_route_m:
            self.off_route_ticks += 1
            if self.off_route_ticks > 20 and self.time_since_last_replan > self.replan_cooldown_s:
                self.state = NavigationState.REPLANNING
                needs_replan = True
                self.time_since_last_replan = 0.0
                self.off_route_ticks = 0
                print(f"[navigation] Route departure ({lateral_departure:.1f}m > {self.replan_off_route_m:.1f}m); triggering REPLAN")
        else:
            self.off_route_ticks = max(0, self.off_route_ticks - 1)

        # 4. Lookahead waypoint selection for smooth Pure Pursuit steering
        lookahead_m = float(np.clip(0.35 * speed_kmh, 3.5, 8.5))
        accum_dist = 0.0
        target_idx = self.current_index
        for i in range(self.current_index, len(self.polyline_2d) - 1):
            p1 = self.polyline_2d[i]
            p2 = self.polyline_2d[i + 1]
            seg_len = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            accum_dist += seg_len
            if accum_dist >= lookahead_m:
                target_idx = i + 1
                break
        else:
            target_idx = len(self.polyline_2d) - 1

        target_wp = self.waypoints[target_idx]
        target_pt = self.polyline_2d[target_idx]

        # Calculate remaining distance to destination
        rem_dist = dist_to_destination
        self.remaining_distance_m = rem_dist

        # 5. Lateral Steering Control (Stanley / Pure Pursuit combination)
        target_yaw = float(target_wp.transform.rotation.yaw)
        car_yaw = (float(current_yaw_deg) + 180.0) % 360.0 - 180.0
        heading_err = (target_yaw - car_yaw + 180.0) % 360.0 - 180.0

        # Cross-track error relative to target segment forward direction
        target_forward = target_wp.transform.get_forward_vector()
        target_right = target_wp.transform.get_right_vector()
        dx_loc = current_x - target_pt[0]
        dy_loc = current_y - target_pt[1]
        cross_track = dx_loc * target_right.x + dy_loc * target_right.y

        # Steering command calculation (negative = left, positive = right in CARLA)
        # Heading component + cross track damping
        steer_target = (heading_err / 30.0) - (0.07 * cross_track)
        steer_target = float(np.clip(steer_target, -1.0, 1.0))

        # Smooth steering rate
        steer_command = float(np.clip(0.70 * self.previous_steer + 0.30 * steer_target, -1.0, 1.0))
        self.previous_steer = steer_command

        # 6. Longitudinal Speed Control
        # Speed profiling: slow down before sharp turns and near destination
        speed_target = self.cruise_speed_kmh
        if abs(steer_command) > 0.30:
            speed_target = max(8.0, self.cruise_speed_kmh * 0.40)
        elif abs(steer_command) > 0.15:
            speed_target = max(10.0, self.cruise_speed_kmh * 0.55)

        # Smooth deceleration ramp approaching destination
        if dist_to_destination < 20.0:
            ramp = max(
                0.0,
                (dist_to_destination - self.arrival_radius_m)
                / (20.0 - self.arrival_radius_m),
            )
            speed_target = min(speed_target, max(4.0 * ramp, 25.0 * ramp))

        speed_error = speed_target - speed_kmh
        if speed_error > 0.0:
            throttle = float(np.clip(speed_error * 0.05, 0.0, 0.55))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-speed_error * 0.06, 0.0, 0.8))

        return steer_command, throttle, brake, needs_replan
