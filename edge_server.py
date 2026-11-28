"""Edge Node 2: Lightweight Distributed Navigation & Mission Server.

Receives the vehicle's current standing location + destination from Node 1,
computes shortest-path road routes from road caches, serves real-time
navigation instructions (steer, speed hints, maneuver, arrival FSM, replanning),
and hosts a live mobile/web dashboard (/dashboard) with tap-anywhere destination picking.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from navigation.gps_tracker import GPSTracker
from navigation.offline_router import OfflineCacheRouter, infer_maneuver
from navigation.route_follower import NavigationState, RouteFollower
from schemas import (
    PlanRouteRequest,
    PlanRouteResponse,
    SetDestinationRequest,
    StepNavigationRequest,
    StepNavigationResponse,
)

MAX_ACTIVE_SESSIONS = 4  # Keep RAM footprint tiny on resource-constrained devices
DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"


@dataclass
class NavigationSession:
    route_id: str
    map_name: str
    follower: RouteFollower
    tracker: GPSTracker
    latest_maneuver: str = "LANEFOLLOW"
    pending_replan_waypoints: Optional[List[List[float]]] = field(default=None)
    pending_new_destination: Optional[List[float]] = field(default=None)


app = FastAPI(
    title="CARLA Edge Node 2 - Navigation Server",
    version="1.1.0",
    description="Distributed Edge AI navigation planner and controller for autonomous vehicles.",
)

router = OfflineCacheRouter()
sessions: OrderedDict[str, NavigationSession] = OrderedDict()


def _serialize_waypoints(waypoints: list) -> List[List[float]]:
    return [
        [
            round(float(wp.transform.location.x), 2),
            round(float(wp.transform.location.y), 2),
            round(float(wp.transform.rotation.yaw), 2),
        ]
        for wp in waypoints
    ]


def _get_target_session(route_id: Optional[str]) -> NavigationSession:
    if not sessions:
        raise HTTPException(status_code=404, detail="No active navigation session")
    if route_id is not None:
        session = sessions.get(route_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown route_id: {route_id}")
        return session
    # Return most recently active session
    last_key = next(reversed(sessions))
    return sessions[last_key]


@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "Node 2 Edge Navigation Server",
        "loaded_map": router._cached_map_key,
        "active_sessions": len(sessions),
        "dashboard": "/dashboard",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ONLINE",
        "service": "Node 2 Edge Navigation Server",
        "loaded_map": router._cached_map_key,
        "active_sessions": len(sessions),
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    if not DASHBOARD_HTML_PATH.is_file():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return HTMLResponse(content=DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/maps")
def list_maps():
    return {"supported_maps": router.list_supported_maps()}


@app.get("/map_roads/{map_name}")
def get_map_roads(map_name: str):
    try:
        return router.get_map_roads_payload(map_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/plan_route", response_model=PlanRouteResponse)
def plan_route(req: PlanRouteRequest):
    origin_heading = req.origin[2] if len(req.origin) > 2 else None
    dest = (req.destination[0], req.destination[1]) if req.destination else None

    plan = router.plan_route(
        map_name=req.map_name,
        origin=(req.origin[0], req.origin[1]),
        destination=dest,
        origin_heading=origin_heading,
    )
    if not plan.success:
        raise HTTPException(status_code=400, detail=plan.error_message or "Route planning failed")

    route_id = uuid.uuid4().hex[:12]
    follower = RouteFollower(
        target_speed_kmh=req.target_speed_kmh,
        arrival_radius_m=req.arrival_radius_m,
    )
    follower.set_route(plan)

    tracker = GPSTracker(
        destination=plan.destination_location,
        planned_distance_m=plan.total_distance_m,
    )
    tracker.update(
        current_location=(req.origin[0], req.origin[1], 0.5),
        current_speed_kmh=0.0,
        heading_yaw_deg=origin_heading or 0.0,
    )

    while len(sessions) >= MAX_ACTIVE_SESSIONS:
        sessions.popitem(last=False)

    sessions[route_id] = NavigationSession(
        route_id=route_id,
        map_name=req.map_name,
        follower=follower,
        tracker=tracker,
    )

    return PlanRouteResponse(
        status="SUCCESS",
        route_id=route_id,
        map_name=req.map_name,
        origin=[plan.start_location[0], plan.start_location[1]],
        destination=[plan.destination_location[0], plan.destination_location[1]],
        total_distance_m=plan.total_distance_m,
        waypoint_count=len(plan.waypoints),
        waypoints=_serialize_waypoints(plan.waypoints),
        maneuvers=[str(m) for m in plan.road_options],
    )


@app.post("/set_destination", response_model=PlanRouteResponse)
def set_destination(req: SetDestinationRequest):
    """Dynamically updates destination from tapped map coordinates (snapped to nearest road)."""
    session = _get_target_session(req.route_id)

    cur_x = session.tracker.x
    cur_y = session.tracker.y
    cur_yaw = session.tracker.heading_yaw_deg

    if req.random_dest or req.destination is None:
        target_dest = router.pick_random_destination(session.map_name, (cur_x, cur_y))
    else:
        target_dest = (float(req.destination[0]), float(req.destination[1]))

    plan = router.plan_route(
        map_name=session.map_name,
        origin=(cur_x, cur_y),
        destination=target_dest,
        origin_heading=cur_yaw,
    )
    if not plan.success:
        raise HTTPException(status_code=400, detail=plan.error_message or "Could not plan to selected point")

    session.follower.set_route(plan)
    session.tracker.set_destination(plan.destination_location)
    session.tracker.set_planned_distance(plan.total_distance_m)
    session.tracker.driven_distance_m = 0.0

    serialized = _serialize_waypoints(plan.waypoints)
    session.pending_replan_waypoints = serialized
    session.pending_new_destination = [plan.destination_location[0], plan.destination_location[1]]
    session.latest_maneuver = str(plan.road_options[0]) if plan.road_options else "LANEFOLLOW"

    return PlanRouteResponse(
        status="SUCCESS",
        route_id=session.route_id,
        map_name=session.map_name,
        origin=[plan.start_location[0], plan.start_location[1]],
        destination=[plan.destination_location[0], plan.destination_location[1]],
        total_distance_m=plan.total_distance_m,
        waypoint_count=len(plan.waypoints),
        waypoints=serialized,
        maneuvers=[str(m) for m in plan.road_options],
    )


@app.post("/navigation/step", response_model=StepNavigationResponse)
def step_navigation(req: StepNavigationRequest):
    session = sessions.get(req.route_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {req.route_id}")

    sessions.move_to_end(req.route_id)

    steer, throttle, brake, needs_replan = session.follower.step(
        current_x=req.x,
        current_y=req.y,
        current_yaw_deg=req.yaw_deg,
        speed_kmh=req.speed_kmh,
        delta_s=req.delta_s,
        failsafe_blocked=req.failsafe_blocked,
    )

    new_waypoints = None
    new_destination = None

    if session.pending_replan_waypoints is not None:
        needs_replan = True
        new_waypoints = session.pending_replan_waypoints
        new_destination = session.pending_new_destination
        session.pending_replan_waypoints = None
        session.pending_new_destination = None
    elif needs_replan and session.follower.destination is not None:
        replan = router.plan_route(
            map_name=session.map_name,
            origin=(req.x, req.y),
            destination=session.follower.destination,
            origin_heading=req.yaw_deg,
        )
        if replan.success:
            session.follower.set_route(replan)
            session.tracker.set_planned_distance(replan.total_distance_m)
            new_waypoints = _serialize_waypoints(replan.waypoints)

    session.tracker.update(
        current_location=(req.x, req.y, 0.5),
        current_speed_kmh=req.speed_kmh,
        heading_yaw_deg=req.yaw_deg,
    )

    maneuver = (
        "ARRIVED"
        if session.follower.state == NavigationState.ARRIVED
        else infer_maneuver(session.follower.waypoints, session.follower.current_index)
    )
    session.latest_maneuver = maneuver

    return StepNavigationResponse(
        route_id=req.route_id,
        nav_steer=round(steer, 4),
        throttle_hint=round(throttle, 4),
        brake_hint=round(brake, 4),
        target_speed_kmh=round(session.follower.cruise_speed_kmh, 1),
        maneuver=maneuver,
        nav_state=session.follower.state.value,
        remaining_distance_m=round(session.follower.remaining_distance_m, 2),
        route_progress_pct=round(session.tracker.route_progress_pct, 1),
        needs_replan=needs_replan,
        new_waypoints=new_waypoints,
        new_destination=new_destination,
    )


@app.get("/telemetry/latest")
def get_latest_telemetry() -> Dict:
    if not sessions:
        return {"active": False}
    last_key = next(reversed(sessions))
    session = sessions[last_key]
    data = session.tracker.get_telemetry()
    data["active"] = True
    data["route_id"] = session.route_id
    data["map_name"] = session.map_name
    data["nav_state"] = session.follower.state.value
    data["maneuver"] = session.latest_maneuver
    data["remaining_distance_m"] = round(session.follower.remaining_distance_m, 2)
    data["destination"] = list(session.follower.destination) if session.follower.destination else None
    data["polyline_2d"] = session.follower.polyline_2d
    return data


@app.get("/telemetry/{route_id}")
def get_telemetry(route_id: str) -> Dict:
    session = sessions.get(route_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {route_id}")
    data = session.tracker.get_telemetry()
    data["nav_state"] = session.follower.state.value
    data["maneuver"] = session.latest_maneuver
    data["remaining_distance_m"] = round(session.follower.remaining_distance_m, 2)
    return data
