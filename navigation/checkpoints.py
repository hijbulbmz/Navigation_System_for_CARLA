"""Checkpoint definitions and management for CARLA Town03 map.

Defines the 10 fixed checkpoints (P1 to P10), handles loading/saving JSON configuration,
and provides a deterministic selection procedure using CARLA's map API when regenerating
checkpoints from driving waypoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "checkpoints_town03.json",
)

# Reference definitions for Town03 regions matching the required layout:
TARGET_REGIONS: List[Dict[str, Any]] = [
    {
        "id": "P1",
        "name": "P1 - Central road near the roundabout",
        "region": "Central road near the roundabout",
        # Anchor target in Town03 world coords
        "target": (142.0, 195.0),
        "prefer_roundabout_approach": True,
    },
    {
        "id": "P2",
        "name": "P2 - Roundabout approach/exit",
        "region": "Roundabout approach/exit",
        "target": (125.0, 240.0),
        "prefer_roundabout_exit": True,
    },
    {
        "id": "P3",
        "name": "P3 - Upper-left road junction",
        "region": "Upper-left road junction",
        "relative": (-0.75, 0.75),
    },
    {
        "id": "P4",
        "name": "P4 - Upper-middle junction",
        "region": "Upper-middle junction",
        "relative": (0.0, 0.85),
    },
    {
        "id": "P5",
        "name": "P5 - Upper-right junction",
        "region": "Upper-right junction",
        "relative": (0.85, 0.85),
    },
    {
        "id": "P6",
        "name": "P6 - Right-middle road junction",
        "region": "Right-middle road junction",
        "relative": (0.85, 0.05),
    },
    {
        "id": "P7",
        "name": "P7 - Lower-right road section",
        "region": "Lower-right road section",
        "relative": (0.80, -0.70),
    },
    {
        "id": "P8",
        "name": "P8 - Bottom-right curved road",
        "region": "Bottom-right curved road",
        "relative": (0.45, -0.90),
    },
    {
        "id": "P9",
        "name": "P9 - Bottom-left road section",
        "region": "Bottom-left road section",
        "relative": (-0.80, -0.75),
    },
    {
        "id": "P10",
        "name": "P10 - Left-middle road junction",
        "region": "Left-middle road junction",
        "relative": (-0.85, 0.08),
    },
]


@dataclass
class Checkpoint:
    """Represents a single fixed map checkpoint."""

    id: str
    name: str
    region: str
    x: float
    y: float
    z: float
    road_id: int
    section_id: int
    lane_id: int
    s: float
    latitude: float
    longitude: float
    altitude: float

    @property
    def world_coordinates(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @property
    def gps(self) -> Dict[str, float]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "region": self.region,
            "world_coordinates": {
                "x": round(self.x, 2),
                "y": round(self.y, 2),
                "z": round(self.z, 2),
            },
            "road_id": int(self.road_id),
            "section_id": int(self.section_id),
            "lane_id": int(self.lane_id),
            "s": round(self.s, 2),
            "gps": {
                "latitude": round(self.latitude, 6),
                "longitude": round(self.longitude, 6),
                "altitude": round(self.altitude, 2),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Checkpoint:
        coords = data.get("world_coordinates", {})
        gps = data.get("gps", {})
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            region=str(data.get("region", "")),
            x=float(coords.get("x", 0.0)),
            y=float(coords.get("y", 0.0)),
            z=float(coords.get("z", 0.0)),
            road_id=int(data.get("road_id", 0)),
            section_id=int(data.get("section_id", 0)),
            lane_id=int(data.get("lane_id", 0)),
            s=float(data.get("s", 0.0)),
            latitude=float(gps.get("latitude", 0.0)),
            longitude=float(gps.get("longitude", 0.0)),
            altitude=float(gps.get("altitude", coords.get("z", 0.0))),
        )


class CheckpointManager:
    """Manages loading, validation, and generation of Town03 checkpoints."""

    def __init__(self, checkpoints: Optional[List[Checkpoint]] = None):
        self.checkpoints: List[Checkpoint] = checkpoints or []

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __iter__(self):
        return iter(self.checkpoints)

    def get_by_id(self, cp_id: str) -> Optional[Checkpoint]:
        for cp in self.checkpoints:
            if cp.id.upper() == cp_id.upper():
                return cp
        return None

    @classmethod
    def load(cls, config_path: str = DEFAULT_CONFIG_PATH) -> CheckpointManager:
        """Loads checkpoints from a JSON file."""
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Checkpoint config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_list = data.get("checkpoints", [])
        if not isinstance(raw_list, list) or len(raw_list) == 0:
            raise ValueError(f"No valid 'checkpoints' list found in {config_path}")

        checkpoints = [Checkpoint.from_dict(item) for item in raw_list]
        manager = cls(checkpoints)
        is_valid, msg = manager.validate()
        if not is_valid:
            raise ValueError(f"Invalid checkpoint configuration in {config_path}: {msg}")
        return manager

    def save(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        """Saves checkpoints to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        payload = {
            "map": "Town03",
            "count": len(self.checkpoints),
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def validate(self) -> Tuple[bool, str]:
        """Validates that exactly 10 checkpoints exist with unique IDs and coordinates."""
        if len(self.checkpoints) != 10:
            return False, f"Expected exactly 10 checkpoints, found {len(self.checkpoints)}"

        seen_ids = set()
        seen_coords = set()
        expected_ids = {f"P{i}" for i in range(1, 11)}

        for cp in self.checkpoints:
            if not cp.id:
                return False, "Checkpoint with empty ID detected"
            if cp.id in seen_ids:
                return False, f"Duplicate checkpoint ID detected: {cp.id}"
            seen_ids.add(cp.id)

            coord_key = (round(cp.x, 1), round(cp.y, 1))
            if coord_key in seen_coords:
                return False, f"Duplicate coordinates detected for {cp.id} at {coord_key}"
            seen_coords.add(coord_key)

        missing_ids = expected_ids - seen_ids
        if missing_ids:
            return False, f"Missing required checkpoint IDs: {sorted(missing_ids)}"

        return True, "OK"


def load_checkpoints(config_path: str = DEFAULT_CONFIG_PATH) -> List[Checkpoint]:
    """Helper function to load checkpoints list from file."""
    manager = CheckpointManager.load(config_path)
    return manager.checkpoints


def save_checkpoints(
    checkpoints: List[Checkpoint], config_path: str = DEFAULT_CONFIG_PATH
) -> None:
    """Helper function to save checkpoints list to file."""
    manager = CheckpointManager(checkpoints)
    manager.save(config_path)


def generate_town03_checkpoints(
    carla_map: Any,
    save_path: Optional[str] = DEFAULT_CONFIG_PATH,
    min_separation_m: float = 35.0,
) -> List[Checkpoint]:
    """Deterministic checkpoint placement procedure using CARLA map data.

    Picks 10 spatially distributed driving waypoints across Town03 matching
    the required regions (P1 to P10), ensuring no checkpoints are inside junctions.
    """
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("CARLA Python API is required to generate checkpoints from map") from exc

    # Generate waypoints every 2 meters
    all_waypoints = carla_map.generate_waypoints(2.0)

    # Filter: valid driving lane only, strictly avoid middle of intersections (junctions)
    candidates = [
        wp
        for wp in all_waypoints
        if not wp.is_junction
        and getattr(wp, "lane_type", None) == carla.LaneType.Driving
    ]

    if not candidates:
        raise RuntimeError("No valid non-junction driving waypoints found in the current map")

    # Map bounds
    xs = [wp.transform.location.x for wp in candidates]
    ys = [wp.transform.location.y for wp in candidates]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    # Locate roundabout center if present, otherwise default to map center
    roundabout_center = (142.0, 241.0)
    # Check if waypoints exist near this typical Town03 roundabout coordinate
    has_roundabout_near_typical = any(
        (wp.transform.location.x - 142.0) ** 2 + (wp.transform.location.y - 241.0) ** 2 < 400.0
        for wp in all_waypoints if wp.is_junction
    )
    if not has_roundabout_near_typical:
        roundabout_center = (mid_x, mid_y)

    chosen_checkpoints: List[Checkpoint] = []
    chosen_positions: List[Tuple[float, float]] = []

    for region_info in TARGET_REGIONS:
        cp_id = region_info["id"]
        cp_name = region_info["name"]
        region_desc = region_info["region"]

        # Determine target point
        if "target" in region_info:
            target_x, target_y = region_info["target"]
            # If map bounds are drastically shifted, translate relative to roundabout center
            if not has_roundabout_near_typical:
                target_x = mid_x + (region_info["target"][0] - 142.0)
                target_y = mid_y + (region_info["target"][1] - 240.0)
        else:
            rel_x, rel_y = region_info["relative"]
            target_x = mid_x + rel_x * (span_x / 2.0)
            target_y = mid_y + rel_y * (span_y / 2.0)

        # Filter candidates separated from previously chosen checkpoints
        available = []
        for wp in candidates:
            loc = wp.transform.location
            dist_to_chosen = [
                math.hypot(loc.x - px, loc.y - py) for px, py in chosen_positions
            ]
            if not dist_to_chosen or min(dist_to_chosen) >= min_separation_m:
                available.append(wp)

        if not available:
            # Fallback to lower separation if map density is low
            available = candidates

        # Deterministic sorting: distance to target + tie breaking by road_id, lane_id, s
        best_wp = min(
            available,
            key=lambda wp: (
                (wp.transform.location.x - target_x) ** 2
                + (wp.transform.location.y - target_y) ** 2,
                int(wp.road_id),
                int(wp.lane_id),
                round(float(wp.s), 2),
            ),
        )

        loc = best_wp.transform.location
        chosen_positions.append((loc.x, loc.y))

        # Geolocation conversion via CARLA Map API
        try:
            geo = carla_map.transform_to_geolocation(loc)
            lat, lon, alt = float(geo.latitude), float(geo.longitude), float(geo.altitude)
        except Exception:
            lat = loc.y / 111319.49
            lon = loc.x / (111319.49 * max(0.1, math.cos(math.radians(lat))))
            alt = loc.z

        cp = Checkpoint(
            id=cp_id,
            name=cp_name,
            region=region_desc,
            x=round(float(loc.x), 2),
            y=round(float(loc.y), 2),
            z=round(float(loc.z), 2),
            road_id=int(best_wp.road_id),
            section_id=int(getattr(best_wp, "section_id", 0)),
            lane_id=int(best_wp.lane_id),
            s=round(float(best_wp.s), 2),
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            altitude=round(alt, 2),
        )
        chosen_checkpoints.append(cp)

    if save_path:
        save_checkpoints(chosen_checkpoints, save_path)
        print(f"[checkpoints] Saved {len(chosen_checkpoints)} Town03 checkpoints to: {save_path}")

    return chosen_checkpoints


def load_or_generate_checkpoints(
    carla_map: Any = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    force_regenerate: bool = False,
) -> List[Checkpoint]:
    """Loads checkpoints from configuration file if it exists, otherwise generates them from CARLA map."""
    if not force_regenerate and os.path.isfile(config_path):
        try:
            return load_checkpoints(config_path)
        except Exception as exc:
            print(f"[checkpoints] Warning: Could not load existing config ({exc}), regenerating...")

    if carla_map is not None:
        return generate_town03_checkpoints(carla_map, save_path=config_path)

    raise RuntimeError(
        f"Checkpoint configuration file not found at '{config_path}' and no CARLA map provided to generate one"
    )


def print_checkpoints_table(checkpoints: List[Checkpoint]) -> None:
    """Pretty prints the checkpoints in a formatted CLI table."""
    header = (
        f"{'ID':<4} | {'Region':<34} | {'X':>7} | {'Y':>7} | {'Z':>5} | "
        f"{'Road':>4} | {'Lane':>4} | {'S':>6} | {'Lat':>10} | {'Lon':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for cp in checkpoints:
        print(
            f"{cp.id:<4} | {cp.region:<34} | {cp.x:>7.1f} | {cp.y:>7.1f} | {cp.z:>5.1f} | "
            f"{cp.road_id:>4} | {cp.lane_id:>4} | {cp.s:>6.1f} | {cp.latitude:>10.6f} | {cp.longitude:>10.6f}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Town03 Checkpoint Tool")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Path to checkpoints JSON configuration"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Connect to CARLA and generate new checkpoints"
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--town", default="Town03", help="CARLA town map name")
    args = parser.parse_args()

    if args.generate:
        print(f"[connect] Connecting to CARLA at {args.host}:{args.port}...")
        try:
            import carla
        except ImportError:
            print("Error: CARLA Python API not installed in this Python environment.")
            sys.exit(1)

        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        world = client.get_world()
        if args.town and args.town not in world.get_map().name:
            print(f"[load] Loading map {args.town}...")
            world = client.load_world(args.town)
        carla_map = world.get_map()
        print(f"[generate] Extracting checkpoints from {carla_map.name}...")
        checkpoints = generate_town03_checkpoints(carla_map, save_path=args.config)
        print_checkpoints_table(checkpoints)
    else:
        print(f"[load] Loading checkpoints from {args.config}...")
        manager = CheckpointManager.load(args.config)
        print_checkpoints_table(manager.checkpoints)
        print(f"[ok] Verified {len(manager.checkpoints)} checkpoints successfully.")


if __name__ == "__main__":
    main()
