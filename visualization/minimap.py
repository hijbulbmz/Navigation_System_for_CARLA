"""CARLA 2D Live Minimap with route markers and vehicle tracking.

Renders a top-down CARLA road network, Start and Destination markers, and the ego
vehicle on OpenCV camera frames.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

DEFAULT_ROAD_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "town03_roads_cache.json",
)
ROAD_CACHE_DIRECTORY = os.path.dirname(DEFAULT_ROAD_CACHE_PATH)
MAP_INTERIOR_FRACTION = 0.1


def map_cache_key(map_name: str) -> str:
    """Return a filesystem-safe key for a CARLA map path or map name."""
    basename = map_name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    key = re.sub(r"[^A-Za-z0-9_-]+", "_", basename).strip("_").lower()
    if not key:
        raise ValueError(f"Invalid CARLA map name: {map_name!r}")
    return key


def road_cache_path_for_map(map_name: str) -> str:
    """Return the road-cache path for one CARLA map."""
    key = map_cache_key(map_name)
    if key == "town03":
        return DEFAULT_ROAD_CACHE_PATH
    return os.path.join(ROAD_CACHE_DIRECTORY, f"{key}_roads_cache.json")


def select_start_destination(
    roads: List[List[Tuple[float, float]]],
    bounds: Tuple[float, float, float, float],
    preferred_start: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Choose distinct road points inside the map bounds, favoring a supplied start."""
    min_x, max_x, min_y, max_y = bounds
    span_x = max_x - min_x
    span_y = max_y - min_y
    margin_x = max(2.0, span_x * MAP_INTERIOR_FRACTION)
    margin_y = max(2.0, span_y * MAP_INTERIOR_FRACTION)

    road_points = list(dict.fromkeys(
        point for segment in roads for point in segment
    ))
    candidates = [
        point for point in road_points
        if min_x + margin_x <= point[0] <= max_x - margin_x
        and min_y + margin_y <= point[1] <= max_y - margin_y
    ]
    if len(candidates) < 2:
        raise RuntimeError(
            "The road cache needs at least two distinct road points away from "
            "the map border to place Start and Destination"
        )

    if preferred_start is None:
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        start = min(
            candidates,
            key=lambda point: (point[0] - center[0]) ** 2
            + (point[1] - center[1]) ** 2,
        )
    else:
        start = min(
            candidates,
            key=lambda point: (point[0] - preferred_start[0]) ** 2
            + (point[1] - preferred_start[1]) ** 2,
        )

    destination = max(
        (point for point in candidates if point != start),
        key=lambda point: (point[0] - start[0]) ** 2
        + (point[1] - start[1]) ** 2,
    )
    min_separation = max(5.0, min(span_x, span_y) * 0.1)
    distance_sq = (
        (destination[0] - start[0]) ** 2
        + (destination[1] - start[1]) ** 2
    )
    if distance_sq < min_separation ** 2:
        raise RuntimeError(
            "The map does not have two sufficiently separated interior road points"
        )
    return start, destination


# Color definitions in BGR format (for OpenCV)
COLOR_BACKGROUND = (30, 30, 30)       # Dark charcoal
COLOR_PANEL_BORDER = (80, 80, 80)     # Subtle border
COLOR_ROAD = (175, 175, 175)          # Light grey
COLOR_ROAD_OUTLINE = (100, 100, 100)  # Medium grey road casing
COLOR_START = (255, 100, 0)           # Blue
COLOR_DESTINATION = (0, 130, 245)     # Red-orange
COLOR_MARKER_BORDER = (20, 20, 20)
COLOR_VEHICLE = (0, 235, 0)           # Bright green
COLOR_VEHICLE_BORDER = (0, 60, 0)     # Dark green outline
COLOR_TEXT = (255, 255, 255)          # White
COLOR_LEGEND_BG = (20, 20, 20)        # Legend banner
COLOR_ROUTE = (0, 235, 255)           # High-contrast bright cyan for active route
COLOR_ROUTE_OUTLINE = (20, 30, 40)    # Dark contrast casing
COLOR_ROUTE_COMPLETED = (130, 110, 80)# Muted line for already traversed road
COLOR_ROUTE_STATUS = (0, 255, 200)    # Soft bright status indicator


class Minimap:
    """Top-down CARLA minimap with Start and Destination road markers."""

    def __init__(
        self,
        size: int = 260,
        position: str = "top-right",
        opacity: float = 0.85,
        carla_map: Optional[Any] = None,
        road_cache_path: Optional[str] = None,
        padding: int = 14,
        map_name: Optional[str] = None,
        preferred_start: Optional[Tuple[float, float]] = None,
    ):
        """Initializes the Minimap.

        Args:
            size: Square dimensions (width & height in pixels) of the minimap panel.
            position: Position on the camera frame ('top-right' or 'top-left').
            opacity: Transparency blend weight (0.0 to 1.0).
            carla_map: Optional live CARLA map instance (for extracting road network).
            road_cache_path: Path to cached road polylines.
            padding: Inner margin in pixels between roads and minimap border.
            map_name: CARLA map name used to select the correct road cache.
            preferred_start: Optional start hint, snapped to an interior road point.
        """
        self.size = max(80, int(size))
        self.position = position.lower()
        if self.position not in ("top-right", "top-left"):
            self.position = "top-right"
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.padding = padding
        source_map_name = map_name or getattr(carla_map, "name", None) or "Town03"
        self.map_name = source_map_name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        self.map_key = map_cache_key(self.map_name)
        self.road_cache_path = road_cache_path or road_cache_path_for_map(self.map_name)

        # 1. Load or extract road polylines
        self.roads: List[List[Tuple[float, float]]] = []
        self.min_x: float = -260.0
        self.max_x: float = 260.0
        self.min_y: float = -260.0
        self.max_y: float = 260.0

        if carla_map is not None:
            self._extract_from_carla_map(carla_map)
        else:
            self._load_from_cache_or_default()

        self.start_location, self.destination_location = select_start_destination(
            self.roads,
            (self.min_x, self.max_x, self.min_y, self.max_y),
            preferred_start=preferred_start,
        )

        # 2. Compute coordinate transformation parameters
        self._compute_transform_parameters()

        # 3. Dynamic navigation route and progress state
        self.route_points: List[Tuple[float, float]] = []
        self.route_pixel_pts: List[Tuple[int, int]] = []
        self.route_distance_m: Optional[float] = None
        self.remaining_distance_m: Optional[float] = None
        self.current_route_index: int = 0
        self.navigation_status: str = "IDLE"

        # 4. Render the static base image (background, roads, markers, and legend)
        self.base_image: Optional[np.ndarray] = None
        if np is not None and cv2 is not None:
            self._render_static_base()

    def set_route(
        self,
        route_points: List[Tuple[float, float]],
        route_distance_m: Optional[float] = None,
    ) -> None:
        """Sets the active navigation route waypoints in world coordinates.

        Renders cleanly on top of the road network and beneath the vehicle/markers.
        """
        self.route_points = list(route_points)
        self.route_pixel_pts = [self.world_to_pixel(pt[0], pt[1]) for pt in self.route_points]
        self.route_distance_m = float(route_distance_m) if route_distance_m is not None else None
        self.remaining_distance_m = self.route_distance_m
        self.current_route_index = 0
        self.navigation_status = "FOLLOWING" if self.route_points else "IDLE"

    def clear_route(self) -> None:
        """Clears the active navigation route."""
        self.route_points = []
        self.route_pixel_pts = []
        self.route_distance_m = None
        self.remaining_distance_m = None
        self.current_route_index = 0
        self.navigation_status = "IDLE"

    def set_destination(self, dest_x: float, dest_y: float) -> None:
        """Updates the destination marker and re-renders the base background."""
        self.destination_location = (float(dest_x), float(dest_y))
        if np is not None and cv2 is not None:
            self._render_static_base()

    def update_progress(
        self,
        current_x: float,
        current_y: float,
        remaining_distance_m: Optional[float] = None,
        status: str = "FOLLOWING",
    ) -> None:
        """Updates the vehicle's progress along the route and current navigation status."""
        self.navigation_status = status
        if remaining_distance_m is not None:
            self.remaining_distance_m = float(remaining_distance_m)
        if self.route_points:
            # Local window search around current index to advance progress along the route
            search_start = max(0, self.current_route_index - 3)
            search_end = min(len(self.route_points), self.current_route_index + 25)
            best_idx = self.current_route_index
            min_dist_sq = float("inf")
            for idx in range(search_start, search_end):
                pt = self.route_points[idx]
                d_sq = (pt[0] - current_x) ** 2 + (pt[1] - current_y) ** 2
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    best_idx = idx
            self.current_route_index = max(self.current_route_index, best_idx)

    def _extract_from_carla_map(self, carla_map: Any, step_m: float = 2.0) -> None:
        """Extract road polylines from the selected CARLA map and cache them."""
        import carla

        source_map_name = getattr(carla_map, "name", None)
        if source_map_name and map_cache_key(source_map_name) != self.map_key:
            raise ValueError(
                f"Minimap is configured for {self.map_name}, but received "
                f"CARLA map {source_map_name}"
            )

        waypoints = carla_map.generate_waypoints(step_m)
        roads: List[List[Tuple[float, float]]] = []

        for wp in waypoints:
            if getattr(wp, "lane_type", None) != carla.LaneType.Driving:
                continue
            loc1 = wp.transform.location
            for nxt in wp.next(step_m):
                loc2 = nxt.transform.location
                roads.append([(loc1.x, loc1.y), (loc2.x, loc2.y)])

        if not roads:
            raise RuntimeError("CARLA map returned no road waypoints")

        self.roads = roads
        all_x = [pt[0] for segment in roads for pt in segment]
        all_y = [pt[1] for segment in roads for pt in segment]
        self.min_x = min(all_x) - 10.0
        self.max_x = max(all_x) + 10.0
        self.min_y = min(all_y) - 10.0
        self.max_y = max(all_y) + 10.0

        # Save to cache for future fast loading
        self._save_cache()

    def _save_cache(self) -> None:
        """Save the selected map's road segments and bounds."""
        if not self.roads:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.road_cache_path)), exist_ok=True)
        data = {
            "map": self.map_name,
            "bounds": {
                "min_x": round(self.min_x, 1),
                "max_x": round(self.max_x, 1),
                "min_y": round(self.min_y, 1),
                "max_y": round(self.max_y, 1),
            },
            "roads_count": len(self.roads),
            "roads": [[[round(pt[0], 2), round(pt[1], 2)] for pt in seg] for seg in self.roads],
        }
        with open(self.road_cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load_from_cache_or_default(self) -> None:
        """Load and validate the cache for the selected CARLA map."""
        if not os.path.isfile(self.road_cache_path):
            raise FileNotFoundError(
                f"No road cache for {self.map_name} at {self.road_cache_path}; "
                "generate one from the live CARLA map first"
            )
        with open(self.road_cache_path, "r", encoding="utf-8") as cache_file:
            data = json.load(cache_file)

        cache_map_name = data.get("map")
        if not cache_map_name or map_cache_key(str(cache_map_name)) != self.map_key:
            raise ValueError(
                f"Road cache {self.road_cache_path} is for {cache_map_name!r}, "
                f"not {self.map_name}"
            )

        raw_roads = data.get("roads")
        if not isinstance(raw_roads, list) or not raw_roads:
            raise ValueError(f"Road cache for {self.map_name} contains no roads")
        self.roads = [
            [(float(point[0]), float(point[1])) for point in segment]
            for segment in raw_roads
        ]
        if any(len(segment) < 2 for segment in self.roads):
            raise ValueError(f"Road cache for {self.map_name} contains invalid segments")

        bounds = data.get("bounds", {})
        self.min_x = float(bounds["min_x"])
        self.max_x = float(bounds["max_x"])
        self.min_y = float(bounds["min_y"])
        self.max_y = float(bounds["max_y"])
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError(f"Road cache for {self.map_name} has invalid bounds")

    def _compute_transform_parameters(self) -> None:
        """Computes isotropic scaling and centering offsets for world-to-pixel mapping.

        Maintains exact aspect ratio so roundabouts remain circular and road geometry
        is not distorted. Inverted vertical image axis is preserved.
        """
        drawable_w = max(10, self.size - 2 * self.padding)
        drawable_h = max(10, self.size - 2 * self.padding)

        world_span_x = max(1.0, self.max_x - self.min_x)
        world_span_y = max(1.0, self.max_y - self.min_y)

        # Uniform scale factor to prevent spatial distortion
        self.scale = min(drawable_w / world_span_x, drawable_h / world_span_y)

        rendered_w = world_span_x * self.scale
        rendered_h = world_span_y * self.scale

        # Centering offsets
        self.offset_x = self.padding + (drawable_w - rendered_w) / 2.0
        self.offset_y = self.padding + (drawable_h - rendered_h) / 2.0

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Converts CARLA world coordinates (X, Y) to 2D Minimap pixel coordinates (px, py).

        Handles the difference between CARLA's Cartesian coordinate system and 2D image
        coordinates, including the inverted vertical image axis:
        - CARLA: +X = East/Right, +Y = North/Up
        - Minimap Image: +px = Right, +py = Down (0 at top, size at bottom)
        """
        px = int(round(self.offset_x + (x - self.min_x) * self.scale))
        # Note the inverted vertical axis: (self.max_y - y)
        py = int(round(self.offset_y + (self.max_y - y) * self.scale))

        # Clamp to panel boundaries
        px = max(0, min(self.size - 1, px))
        py = max(0, min(self.size - 1, py))
        return px, py

    def yaw_to_direction(self, yaw_deg: float) -> Tuple[float, float]:
        """Converts CARLA vehicle yaw angle (degrees) to a 2D pixel direction unit vector.

        In CARLA (Unreal Engine):
        - yaw = 0 deg: vehicle faces +X (East -> Right on screen)
        - yaw = 90 deg: vehicle faces +Y (North -> Up on screen)
        - yaw = 180 / -180 deg: vehicle faces -X (West -> Left on screen)
        - yaw = -90 deg: vehicle faces -Y (South -> Down on screen)

        In Minimap pixel coordinates (with inverted vertical axis):
        - dx_pixel = cos(yaw)
        - dy_pixel = -sin(yaw)  (pointing UP when yaw = 90)
        """
        rad = math.radians(yaw_deg)
        dx_pixel = math.cos(rad)
        dy_pixel = -math.sin(rad)
        return dx_pixel, dy_pixel

    def _render_static_base(self) -> None:
        """Pre-render the road network, Start/Destination markers, and legend.

        Executed only ONCE during initialization to ensure maximum runtime performance.
        """
        if np is None or cv2 is None:
            return

        base = np.full((self.size, self.size, 3), COLOR_BACKGROUND, dtype=np.uint8)

        # 1. Draw Roads
        # First draw thicker dark casing for road edges, then light grey core
        for segment in self.roads:
            if len(segment) < 2:
                continue
            pixel_pts = [self.world_to_pixel(pt[0], pt[1]) for pt in segment]
            pts_array = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
            # Road core
            cv2.polylines(base, [pts_array], isClosed=False, color=COLOR_ROAD, thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw Start and Destination on sampled road points
        marker_radius = max(3, int(round(self.size * 0.016)))
        font_scale = max(0.28, self.size * 0.0012)
        font_thick = 1

        for location, label, color in (
            (self.start_location, "Start", COLOR_START),
            (self.destination_location, "Destination", COLOR_DESTINATION),
        ):
            px, py = self.world_to_pixel(*location)
            cv2.circle(
                base, (px, py), marker_radius + 1, COLOR_MARKER_BORDER, -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                base, (px, py), marker_radius, color, -1, lineType=cv2.LINE_AA
            )
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

            text_x = px + marker_radius + 2
            text_y = py + text_h // 2
            if text_x + text_w > self.size - 4:
                text_x = px - marker_radius - text_w - 2
            text_x = max(4, text_x)
            text_y = max(text_h + 4, min(self.size - text_h - 18, text_y))

            cv2.putText(
                base,
                label,
                (text_x + 1, text_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thick,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                base,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                COLOR_TEXT,
                font_thick,
                lineType=cv2.LINE_AA,
            )

        # 3. Draw the legend at the bottom of the minimap panel
        legend_h = max(18, int(self.size * 0.08))
        legend_y1 = self.size - legend_h
        cv2.rectangle(base, (0, legend_y1), (self.size, self.size), COLOR_LEGEND_BG, -1)
        cv2.line(base, (0, legend_y1), (self.size, legend_y1), COLOR_PANEL_BORDER, 1)

        leg_font_scale = max(0.22, self.size * 0.0009)
        legend_y = self.size - legend_h // 2
        for x, label, color in (
            (int(self.size * 0.04), "Start", COLOR_START),
            (int(self.size * 0.38), "Destination", COLOR_DESTINATION),
            (int(self.size * 0.82), "Ego", COLOR_VEHICLE),
        ):
            cv2.circle(base, (x, legend_y), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                base,
                label,
                (x + 5, legend_y + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                leg_font_scale,
                COLOR_TEXT,
                1,
                lineType=cv2.LINE_AA,
            )

        # Panel border
        cv2.rectangle(base, (0, 0), (self.size - 1, self.size - 1), COLOR_PANEL_BORDER, 1)

        self.base_image = base

    def render(self, vehicle_x: float, vehicle_y: float, vehicle_yaw: float) -> Optional[np.ndarray]:
        """Renders the minimap with the planned route line and dynamic ego vehicle marker.

        Extremely fast per-frame operation: copies pre-rendered static base image,
        draws the route polyline beneath markers, and renders the vehicle directional arrow.
        """
        if self.base_image is None or np is None or cv2 is None:
            return None

        img = self.base_image.copy()

        # 1. Draw planned route polyline beneath markers and vehicle
        if len(self.route_pixel_pts) >= 2:
            idx = min(self.current_route_index, len(self.route_pixel_pts) - 1)
            # Completed portion of the route (muted)
            if idx > 1:
                pts_done = np.array(self.route_pixel_pts[:idx + 1], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts_done], isClosed=False, color=COLOR_ROUTE_COMPLETED, thickness=2, lineType=cv2.LINE_AA)
            # Remaining portion of the route (high-contrast cyan with dark casing)
            pts_rem = np.array(self.route_pixel_pts[idx:], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_rem], isClosed=False, color=COLOR_ROUTE_OUTLINE, thickness=4, lineType=cv2.LINE_AA)
            cv2.polylines(img, [pts_rem], isClosed=False, color=COLOR_ROUTE, thickness=2, lineType=cv2.LINE_AA)

            # Ensure Destination and Start markers remain crisp on top of the route line
            marker_radius = max(3, int(round(self.size * 0.016)))
            px_dst, py_dst = self.world_to_pixel(*self.destination_location)
            cv2.circle(img, (px_dst, py_dst), marker_radius + 1, COLOR_MARKER_BORDER, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (px_dst, py_dst), marker_radius, COLOR_DESTINATION, -1, lineType=cv2.LINE_AA)

        # 2. Navigation status overlay
        if self.navigation_status != "IDLE":
            status_text = self.navigation_status
            if self.remaining_distance_m is not None:
                status_text += f" {self.remaining_distance_m:.0f}m"
            font_scale = max(0.24, self.size * 0.001)
            cv2.putText(
                img,
                status_text,
                (self.padding, self.padding + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                COLOR_ROUTE_STATUS,
                1,
                lineType=cv2.LINE_AA,
            )

        # 3. Convert vehicle position to pixel
        px, py = self.world_to_pixel(vehicle_x, vehicle_y)

        # Convert vehicle yaw to direction vector
        dx, dy = self.yaw_to_direction(vehicle_yaw)
        nx, ny = -dy, dx  # Normal vector (orthogonal)

        # Arrow dimensions proportional to minimap size
        length = max(6.0, self.size * 0.038)
        width = length * 0.55

        # Vehicle triangle vertices: tip, left wing, rear notch, right wing
        tip_x = px + dx * length
        tip_y = py + dy * length

        left_x = px - dx * (length * 0.5) + nx * width
        left_y = py - dy * (length * 0.5) + ny * width

        notch_x = px - dx * (length * 0.2)
        notch_y = py - dy * (length * 0.2)

        right_x = px - dx * (length * 0.5) - nx * width
        right_y = py - dy * (length * 0.5) - ny * width

        triangle_pts = np.array(
            [[tip_x, tip_y], [left_x, left_y], [notch_x, notch_y], [right_x, right_y]],
            dtype=np.int32,
        )

        # Draw filled vehicle arrow with dark contrast border
        cv2.drawContours(img, [triangle_pts], 0, COLOR_VEHICLE, -1, lineType=cv2.LINE_AA)
        cv2.drawContours(img, [triangle_pts], 0, COLOR_VEHICLE_BORDER, 1, lineType=cv2.LINE_AA)
        # Small center locator dot
        cv2.circle(img, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return img

    def overlay_on_frame(
        self,
        display_frame: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        margin_px: int = 6,
    ) -> np.ndarray:
        """Blends the rendered minimap onto the simulation display frame.

        Args:
            display_frame: BGR image from the camera / simulation pipeline.
            vehicle_x: Vehicle world coordinate X.
            vehicle_y: Vehicle world coordinate Y.
            vehicle_yaw: Vehicle world rotation yaw in degrees.
            margin_px: Padding from the edge of the display frame.

        Returns:
            display_frame with the minimap cleanly overlaid at top-right or top-left.
        """
        if self.base_image is None or cv2 is None or np is None:
            return display_frame

        minimap_img = self.render(vehicle_x, vehicle_y, vehicle_yaw)
        if minimap_img is None:
            return display_frame

        frame_h, frame_w = display_frame.shape[:2]

        # Adaptive sizing: clamp minimap size if frame is smaller than requested size
        max_allowed_w = frame_w - 2 * margin_px
        max_allowed_h = frame_h - 2 * margin_px
        max_dim = min(max_allowed_w, max_allowed_h)

        if self.size > max_dim:
            target_size = max(50, max_dim)
            scaled_minimap = cv2.resize(
                minimap_img, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
        else:
            target_size = self.size
            scaled_minimap = minimap_img

        # Determine placement rectangle
        if self.position == "top-left":
            x1 = margin_px
            # Leave space for top-left status text if frame has room
            y1 = margin_px + 24 if frame_h > target_size + margin_px + 28 else margin_px
        else:  # top-right (default)
            x1 = frame_w - target_size - margin_px
            y1 = margin_px

        x2 = x1 + target_size
        y2 = y1 + target_size

        # Boundary checks
        if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            x1 = max(0, min(frame_w - target_size, x1))
            y1 = max(0, min(frame_h - target_size, y1))
            x2 = x1 + target_size
            y2 = y1 + target_size

        roi = display_frame[y1:y2, x1:x2]

        # Transparent alpha blending
        if self.opacity >= 0.999:
            display_frame[y1:y2, x1:x2] = scaled_minimap
        else:
            blended = cv2.addWeighted(
                scaled_minimap, self.opacity, roi, 1.0 - self.opacity, 0.0
            )
            display_frame[y1:y2, x1:x2] = blended

        return display_frame

