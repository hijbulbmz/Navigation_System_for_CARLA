"""Autonomous road-following route planning between Start and Destination in Town03."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

ROUTE_TOWN = "Town03"
ROUTE_START_SPAWN_INDEX = 0
ROUTE_RESOLUTION_M = 2.0
ROUTE_MAX_DISTANCE_M = 3000.0
MODEL_STEER_WEIGHT = 0.25

DEFAULT_ROUTE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "route_town03.json",
)


def load_start_destination_config(
    config_path: Optional[str] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Loads start and destination coordinates from configuration file.

    Returns:
        ((start_x, start_y, start_z), (dest_x, dest_y, dest_z))
    """
    path = config_path or DEFAULT_ROUTE_CONFIG_PATH
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        start_d = data.get("start", {})
        dest_d = data.get("destination", {})
        start = (
            float(start_d.get("x", -84.79)),
            float(start_d.get("y", 27.59)),
            float(start_d.get("z", 1.0)),
        )
        dest = (
            float(dest_d.get("x", 216.19)),
            float(dest_d.get("y", 169.45)),
            float(dest_d.get("z", 1.0)),
        )
        return start, dest

    # Default Town03 Start and Destination from interior road samples
    return (-84.79, 27.59, 1.0), (216.19, 169.45, 1.0)


def build_route_from_locations(planner, start_location, destinations):
    """Traces sequential route segments through destination targets."""
    try:
        import networkx as nx
        no_path_exceptions = (nx.NetworkXNoPath, nx.NodeNotFound)
    except ImportError:
        no_path_exceptions = ()

    route = []
    route_distance = 0.0
    previous_location = None
    previous_label = f"spawn {ROUTE_START_SPAWN_INDEX}"
    current_location = start_location
    route_limit_reached = False

    for destination_label, destination in destinations:
        try:
            segment = planner.trace_route(current_location, destination)
        except no_path_exceptions as exc:
            raise RuntimeError(
                f"Could not plan route from {previous_label} to {destination_label}"
            ) from exc
        if not segment:
            raise RuntimeError(
                f"Route planner returned no waypoints for {destination_label}"
            )

        for waypoint, road_option in segment:
            location = waypoint.transform.location
            if previous_location is not None:
                step_distance = previous_location.distance(location)
                if step_distance < 0.05:
                    continue
                if route_distance + step_distance > ROUTE_MAX_DISTANCE_M:
                    route_limit_reached = True
                    break
                route_distance += step_distance
            route.append((waypoint, road_option))
            previous_location = location

        if route_limit_reached or route_distance >= ROUTE_MAX_DISTANCE_M:
            break
        current_location = destination
        previous_label = destination_label

    if not route:
        raise RuntimeError("The planned route contains no waypoints")
    return route, route_distance


def build_route_to_destination(
    world_map: Any,
    start_location: Optional[Any] = None,
    destination_location: Optional[Any] = None,
    start_spawn_index: int = ROUTE_START_SPAWN_INDEX,
    resolution: float = ROUTE_RESOLUTION_M,
    config_path: Optional[str] = None,
) -> Tuple[Any, float, Any, Any, Any]:
    """Generates a road-following route from Start to Destination using CARLA's GlobalRoutePlanner.

    Snaps both locations to valid driving-lane waypoints and validates the route.

    Returns:
        (route, route_distance, planner, start_waypoint, destination_waypoint)
    """
    api_path = Path(__file__).resolve().parents[1] / "PythonAPI" / "carla"
    if api_path.is_dir() and str(api_path) not in sys.path:
        sys.path.insert(0, str(api_path))

    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    spawn_points = world_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("The current CARLA map has no vehicle spawn points")

    # Determine start location
    if start_location is None:
        cfg_start, cfg_dest = load_start_destination_config(config_path)
        if start_spawn_index < len(spawn_points):
            start_location = spawn_points[start_spawn_index].location
        else:
            start_location = carla.Location(x=cfg_start[0], y=cfg_start[1], z=cfg_start[2])
    elif not isinstance(start_location, carla.Location):
        start_location = carla.Location(
            x=float(start_location[0]),
            y=float(start_location[1]),
            z=float(start_location[2]) if len(start_location) > 2 else 0.5,
        )

    # Determine destination location
    if destination_location is None:
        cfg_start, cfg_dest = load_start_destination_config(config_path)
        destination_location = carla.Location(x=cfg_dest[0], y=cfg_dest[1], z=cfg_dest[2])
    elif not isinstance(destination_location, carla.Location):
        destination_location = carla.Location(
            x=float(destination_location[0]),
            y=float(destination_location[1]),
            z=float(destination_location[2]) if len(destination_location) > 2 else 0.5,
        )

    # Snap both points to valid driving-lane waypoints on the road network
    start_wp = world_map.get_waypoint(
        start_location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    dest_wp = world_map.get_waypoint(
        destination_location, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    if start_wp is None:
        raise RuntimeError(f"Could not snap start location {start_location} to a driving lane")
    if dest_wp is None:
        raise RuntimeError(f"Could not snap destination location {destination_location} to a driving lane")

    planner = GlobalRoutePlanner(world_map, resolution)
    start_pos = start_wp.transform.location
    dest_pos = dest_wp.transform.location

    route, route_distance = build_route_from_locations(
        planner, start_pos, [("Destination", dest_pos)]
    )

    if not route:
        raise RuntimeError(
            f"Failed to generate route from start ({start_pos.x:.1f}, {start_pos.y:.1f}) "
            f"to destination ({dest_pos.x:.1f}, {dest_pos.y:.1f})"
        )

    return route, route_distance, planner, start_wp, dest_wp


def build_fixed_route(world_map: Any, checkpoints_path: Optional[str] = None):
    """Backward-compatible entry point for live_demo and our_model_demo.

    Routes from Start to Destination.
    """
    spawn_points = world_map.get_spawn_points()
    route, route_distance, planner, start_wp, dest_wp = build_route_to_destination(
        world_map=world_map,
        start_spawn_index=ROUTE_START_SPAWN_INDEX,
        resolution=ROUTE_RESOLUTION_M,
        config_path=checkpoints_path,
    )
    return route, route_distance, planner, spawn_points


def build_navigation_agent(vehicle, world_map, planner, route, target_speed):
    """Builds a CARLA BasicAgent pre-loaded with a planned route."""
    from agents.navigation.basic_agent import BasicAgent

    agent = BasicAgent(
        vehicle,
        target_speed=target_speed,
        opt_dict={"sampling_resolution": ROUTE_RESOLUTION_M},
        map_inst=world_map,
        grp_inst=planner,
    )
    agent.set_global_plan(route, stop_waypoint_creation=True)
    return agent


def blend_model_and_navigation_steering(model_steer: float, navigation_steer: float) -> float:
    """Blends PilotNet steering with navigation steering command."""
    return float(max(
        -1.0,
        min(
            1.0,
            MODEL_STEER_WEIGHT * model_steer
            + (1.0 - MODEL_STEER_WEIGHT) * navigation_steer,
        ),
    ))
