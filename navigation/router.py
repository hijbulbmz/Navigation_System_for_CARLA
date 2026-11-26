"""CARLA Road Graph Router and Route Planning Adapter.

Wraps CARLA's GlobalRoutePlanner to compute legal, shortest-path road routes
from the car's current position to a selected destination.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_PATH = PROJECT_ROOT / "PythonAPI" / "carla"
if API_PATH.is_dir() and str(API_PATH) not in sys.path:
    sys.path.insert(0, str(API_PATH))


@dataclass
class RoutePlan:
    """Holds a validated road route from origin to destination."""

    success: bool
    total_distance_m: float
    waypoints: List[Any]  # List[carla.Waypoint]
    road_options: List[Any]  # List[RoadOption]
    polyline_2d: List[Tuple[float, float]]  # [(x, y), ...] for Minimap and trackers
    start_location: Tuple[float, float]
    destination_location: Tuple[float, float]
    error_message: str = ""


class RoadRouter:
    """Computes valid road routes using CARLA's OpenDRIVE topology graph."""

    def __init__(self, world_map: Any, resolution_m: float = 2.0):
        """Initializes the router with the active CARLA world map."""
        self.world_map = world_map
        self.resolution_m = max(0.5, float(resolution_m))
        self._planner = None
        self._init_planner()

    def _init_planner(self) -> None:
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            self._planner = GlobalRoutePlanner(self.world_map, self.resolution_m)
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize CARLA GlobalRoutePlanner: {exc}"
            ) from exc

    def match_to_road(
        self,
        location: Tuple[float, float] | Any,
        heading_deg: Optional[float] = None,
    ) -> Any:
        """Snaps arbitrary coordinates to a valid driving lane waypoint, favoring current heading."""
        import carla

        if isinstance(location, (tuple, list)):
            loc = carla.Location(x=float(location[0]), y=float(location[1]), z=0.5)
        else:
            loc = location

        wp = self.world_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            raise RuntimeError(f"Could not find valid driving road near {loc.x:.1f}, {loc.y:.1f}")

        # If heading is supplied, ensure lane direction matches the vehicle heading
        if heading_deg is not None:
            car_yaw = (float(heading_deg) + 180.0) % 360.0 - 180.0
            wp_yaw = (float(wp.transform.rotation.yaw) + 180.0) % 360.0 - 180.0
            yaw_diff = abs((car_yaw - wp_yaw + 180.0) % 360.0 - 180.0)
            if yaw_diff > 120.0:
                # Opposite direction: check if adjacent lane has same direction
                left_lane = wp.get_left_lane()
                if left_lane is not None and left_lane.lane_type == carla.LaneType.Driving:
                    left_yaw = (float(left_lane.transform.rotation.yaw) + 180.0) % 360.0 - 180.0
                    if abs((car_yaw - left_yaw + 180.0) % 360.0 - 180.0) < yaw_diff:
                        wp = left_lane

        return wp

    def plan_route(
        self,
        origin: Tuple[float, float] | Any,
        destination: Tuple[float, float] | Any,
        origin_heading: Optional[float] = None,
    ) -> RoutePlan:
        """Calculates the best valid road route between origin and destination."""
        import carla

        try:
            start_wp = self.match_to_road(origin, heading_deg=origin_heading)
            dest_wp = self.match_to_road(destination)
        except Exception as exc:
            return RoutePlan(
                success=False,
                total_distance_m=0.0,
                waypoints=[],
                road_options=[],
                polyline_2d=[],
                start_location=(0.0, 0.0),
                destination_location=(0.0, 0.0),
                error_message=str(exc),
            )

        start_loc = start_wp.transform.location
        dest_loc = dest_wp.transform.location

        try:
            raw_route = self._planner.trace_route(start_loc, dest_loc)
        except Exception as exc:
            return RoutePlan(
                success=False,
                total_distance_m=0.0,
                waypoints=[],
                road_options=[],
                polyline_2d=[],
                start_location=(start_loc.x, start_loc.y),
                destination_location=(dest_loc.x, dest_loc.y),
                error_message=f"A* route search failed: {exc}",
            )

        if not raw_route:
            return RoutePlan(
                success=False,
                total_distance_m=0.0,
                waypoints=[],
                road_options=[],
                polyline_2d=[],
                start_location=(start_loc.x, start_loc.y),
                destination_location=(dest_loc.x, dest_loc.y),
                error_message="Route planner returned 0 waypoints (unreachable destination)",
            )

        waypoints = [item[0] for item in raw_route]
        road_options = [item[1] for item in raw_route]

        # Extract 2D polyline points and compute exact accumulated road distance
        polyline_2d = []
        total_dist = 0.0
        prev_loc = None
        for wp in waypoints:
            loc = wp.transform.location
            pt = (round(float(loc.x), 2), round(float(loc.y), 2))
            if prev_loc is not None:
                step = math.sqrt((loc.x - prev_loc.x) ** 2 + (loc.y - prev_loc.y) ** 2)
                total_dist += step
            polyline_2d.append(pt)
            prev_loc = loc

        return RoutePlan(
            success=True,
            total_distance_m=round(total_dist, 1),
            waypoints=waypoints,
            road_options=road_options,
            polyline_2d=polyline_2d,
            start_location=(start_loc.x, start_loc.y),
            destination_location=(dest_loc.x, dest_loc.y),
        )


def check_route_departure(
    current_x: float,
    current_y: float,
    route_polyline: List[Tuple[float, float]],
    current_index: int = 0,
    search_ahead: int = 25,
) -> Tuple[bool, float, int]:
    """Calculates cross-track departure from the active route segment.

    Returns:
        (is_off_route, min_distance_m, closest_waypoint_index)
    """
    if not route_polyline:
        return False, 0.0, 0

    start_idx = max(0, current_index - 3)
    end_idx = min(len(route_polyline), current_index + search_ahead)

    min_dist_sq = float("inf")
    closest_idx = current_index

    for i in range(start_idx, end_idx):
        pt = route_polyline[i]
        d_sq = (pt[0] - current_x) ** 2 + (pt[1] - current_y) ** 2
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
            closest_idx = i

    min_dist = math.sqrt(min_dist_sq)
    # Off-route threshold: > 4.5 meters lateral deviation from nearest route segment
    is_off_route = min_dist > 4.5

    return is_off_route, min_dist, closest_idx
