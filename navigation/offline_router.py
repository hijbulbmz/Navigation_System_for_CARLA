"""Offline Road-Cache Router for Edge Node 2 (Zero CARLA Dependency).

Builds a lightweight topological road graph from config/*_roads_cache.json on demand
(keeping only one map graph in RAM at a time) and produces RoutePlan objects fully
compatible with RouteFollower.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from navigation.router import RoutePlan

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class EdgeLocation:
    x: float
    y: float
    z: float = 0.5


@dataclass
class EdgeRotation:
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0


class EdgeTransform:
    """Lightweight CARLA-compatible Transform for RouteFollower."""

    __slots__ = ("location", "rotation")

    def __init__(self, x: float, y: float, yaw: float, z: float = 0.5):
        self.location = EdgeLocation(x=x, y=y, z=z)
        self.rotation = EdgeRotation(yaw=yaw)

    def get_forward_vector(self) -> EdgeLocation:
        rad = math.radians(self.rotation.yaw)
        return EdgeLocation(x=math.cos(rad), y=math.sin(rad), z=0.0)

    def get_right_vector(self) -> EdgeLocation:
        rad = math.radians(self.rotation.yaw + 90.0)
        return EdgeLocation(x=math.cos(rad), y=math.sin(rad), z=0.0)


class EdgeWaypoint:
    """Lightweight CARLA-compatible Waypoint for RouteFollower."""

    __slots__ = ("transform",)

    def __init__(self, x: float, y: float, yaw: float, z: float = 0.5):
        self.transform = EdgeTransform(x=x, y=y, yaw=yaw, z=z)


def _snap(coord: float, grid: float = 2.0) -> float:
    return round(coord / grid) * grid


def infer_maneuver(waypoints: List[EdgeWaypoint], idx: int, lookahead_steps: int = 4) -> str:
    """Infers a high-level maneuver label at waypoint index `idx`."""
    if not waypoints or idx >= len(waypoints) - 1:
        return "ARRIVED"
    ahead_idx = min(len(waypoints) - 1, idx + lookahead_steps)
    yaw_now = waypoints[idx].transform.rotation.yaw
    yaw_ahead = waypoints[ahead_idx].transform.rotation.yaw
    diff = (yaw_ahead - yaw_now + 180.0) % 360.0 - 180.0
    if diff > 22.0:
        return "RIGHT"
    if diff < -22.0:
        return "LEFT"
    return "LANEFOLLOW"


class OfflineCacheRouter:
    """Memory-frugal topological router backed by config/*_roads_cache.json.

    Keeps at most one map's NetworkX graph in memory at a time to minimize RAM usage.
    """

    def __init__(self, config_dir: Optional[Path] = None, snap_grid_m: float = 2.0):
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.snap_grid_m = max(0.5, float(snap_grid_m))
        self._cached_map_key: Optional[str] = None
        self._cached_graph: Optional[nx.Graph] = None
        self._cached_bounds: Dict[str, float] = {}

    @staticmethod
    def normalize_map_key(map_name: str) -> str:
        clean = map_name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return clean.lower().replace("_opt", "")

    def list_supported_maps(self) -> List[str]:
        if not self.config_dir.is_dir():
            return []
        maps = set()
        for p in self.config_dir.glob("*_roads_cache.json"):
            key = p.name.replace("_opt_roads_cache.json", "").replace("_roads_cache.json", "")
            maps.add(key.capitalize())
        return sorted(maps)

    def _load_map_graph(self, map_name: str) -> Tuple[nx.Graph, Dict[str, float]]:
        key = self.normalize_map_key(map_name)
        if self._cached_map_key == key and self._cached_graph is not None:
            return self._cached_graph, self._cached_bounds

        cache_file = self.config_dir / f"{key}_roads_cache.json"
        if not cache_file.is_file():
            cache_file = self.config_dir / f"{key}_opt_roads_cache.json"
        if not cache_file.is_file():
            raise FileNotFoundError(f"Road cache not found for map '{map_name}' in {self.config_dir}")

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = nx.Graph()
        grid = self.snap_grid_m
        for seg in data.get("roads", []):
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            s, e = seg[0], seg[1]
            n1 = (_snap(float(s[0]), grid), _snap(float(s[1]), grid))
            n2 = (_snap(float(e[0]), grid), _snap(float(e[1]), grid))
            if n1 != n2:
                graph.add_edge(n1, n2, weight=math.hypot(n1[0] - n2[0], n1[1] - n2[1]))

        # Evict previous map graph immediately to keep RAM usage minimal
        self._cached_map_key = key
        self._cached_graph = graph
        self._cached_bounds = data.get("bounds", {})
        return self._cached_graph, self._cached_bounds

    def get_map_roads_payload(self, map_name: str, max_segments: int = 3500) -> Dict[str, Any]:
        """Returns map bounds and road segments for browser/mobile canvas rendering."""
        key = self.normalize_map_key(map_name)
        cache_file = self.config_dir / f"{key}_roads_cache.json"
        if not cache_file.is_file():
            cache_file = self.config_dir / f"{key}_opt_roads_cache.json"
        if not cache_file.is_file():
            raise FileNotFoundError(f"Road cache not found for map '{map_name}' in {self.config_dir}")

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        roads = data.get("roads", [])
        if len(roads) > max_segments:
            step = max(1, len(roads) // max_segments)
            roads = roads[::step]

        return {
            "map_name": map_name,
            "bounds": data.get("bounds", {}),
            "roads": roads,
        }

    def pick_random_destination(
        self,
        map_name: str,
        origin: Tuple[float, float],
        min_distance_m: float = 25.0,
    ) -> Tuple[float, float]:
        """Picks a random reachable road node at least `min_distance_m` away from `origin`."""
        graph, _ = self._load_map_graph(map_name)
        nodes = list(graph.nodes)
        if not nodes:
            raise RuntimeError(f"Road graph for {map_name} is empty")
        ox, oy = float(origin[0]), float(origin[1])
        start_node = min(nodes, key=lambda n: (n[0] - ox) ** 2 + (n[1] - oy) ** 2)
        reachable = list(nx.node_connected_component(graph, start_node))
        far_nodes = [
            n for n in reachable
            if (n[0] - start_node[0]) ** 2 + (n[1] - start_node[1]) ** 2 >= min_distance_m ** 2
        ]
        choice = random.choice(far_nodes if far_nodes else reachable)
        return (float(choice[0]), float(choice[1]))

    def _default_destination(
        self,
        map_name: str,
        graph: nx.Graph,
        start_node: Tuple[float, float],
    ) -> Tuple[float, float]:
        key = self.normalize_map_key(map_name)
        route_file = self.config_dir / f"route_{key}.json"
        if route_file.is_file():
            with open(route_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            dest_d = data.get("destination", {})
            if "x" in dest_d and "y" in dest_d:
                cand = (float(dest_d["x"]), float(dest_d["y"]))
                # Ensure default destination is reasonably far from current standing point
                if math.hypot(cand[0] - start_node[0], cand[1] - start_node[1]) >= 10.0:
                    return cand

        # Otherwise pick a well-separated road point in the same connected component
        component = list(nx.node_connected_component(graph, start_node))
        far_node = max(
            component,
            key=lambda n: (n[0] - start_node[0]) ** 2 + (n[1] - start_node[1]) ** 2,
        )
        return (float(far_node[0]), float(far_node[1]))

    def plan_route(
        self,
        map_name: str,
        origin: Tuple[float, float],
        destination: Optional[Tuple[float, float]] = None,
        origin_heading: Optional[float] = None,
    ) -> RoutePlan:
        """Plans a road-following route from the vehicle's current standing `origin` to `destination`."""
        try:
            graph, _ = self._load_map_graph(map_name)
        except Exception as exc:
            return RoutePlan(
                success=False,
                total_distance_m=0.0,
                waypoints=[],
                road_options=[],
                polyline_2d=[],
                start_location=(float(origin[0]), float(origin[1])),
                destination_location=(0.0, 0.0),
                error_message=str(exc),
            )

        nodes = list(graph.nodes)
        if not nodes:
            return RoutePlan(
                success=False,
                total_distance_m=0.0,
                waypoints=[],
                road_options=[],
                polyline_2d=[],
                start_location=(float(origin[0]), float(origin[1])),
                destination_location=(0.0, 0.0),
                error_message=f"Road graph for {map_name} is empty",
            )

        ox, oy = float(origin[0]), float(origin[1])
        start_node = min(nodes, key=lambda n: (n[0] - ox) ** 2 + (n[1] - oy) ** 2)

        if destination is None:
            dx, dy = self._default_destination(map_name, graph, start_node)
        else:
            dx, dy = float(destination[0]), float(destination[1])

        # Restrict destination snap to nodes reachable from start_node
        reachable = list(nx.node_connected_component(graph, start_node))
        dest_node = min(reachable, key=lambda n: (n[0] - dx) ** 2 + (n[1] - dy) ** 2)

        try:
            path = nx.shortest_path(graph, source=start_node, target=dest_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            return RoutePlan(
                success=False,
                total_distance_m=0.0,
                waypoints=[],
                road_options=[],
                polyline_2d=[],
                start_location=start_node,
                destination_location=dest_node,
                error_message=str(exc),
            )

        # Convert path coordinates into EdgeWaypoint objects with computed heading yaw
        waypoints: List[EdgeWaypoint] = []
        polyline_2d: List[Tuple[float, float]] = []
        total_dist = 0.0
        default_yaw = float(origin_heading) if origin_heading is not None else 0.0

        for i, pt in enumerate(path):
            if i < len(path) - 1:
                nxt = path[i + 1]
                yaw = math.degrees(math.atan2(nxt[1] - pt[1], nxt[0] - pt[0]))
                total_dist += math.hypot(nxt[0] - pt[0], nxt[1] - pt[1])
            elif waypoints:
                yaw = waypoints[-1].transform.rotation.yaw
            else:
                yaw = default_yaw

            waypoints.append(EdgeWaypoint(x=float(pt[0]), y=float(pt[1]), yaw=round(yaw, 2)))
            polyline_2d.append((round(float(pt[0]), 2), round(float(pt[1]), 2)))

        road_options = [infer_maneuver(waypoints, i) for i in range(len(waypoints))]

        return RoutePlan(
            success=True,
            total_distance_m=round(total_dist, 1),
            waypoints=waypoints,
            road_options=road_options,
            polyline_2d=polyline_2d,
            start_location=(round(float(start_node[0]), 2), round(float(start_node[1]), 2)),
            destination_location=(round(float(dest_node[0]), 2), round(float(dest_node[1]), 2)),
        )
