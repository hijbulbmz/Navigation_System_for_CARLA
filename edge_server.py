"""Edge Node 2: Lightweight Distributed Navigation & Mission Server.

Receives the vehicle's current standing location + destination from Node 1,
computes shortest-path road routes from road caches, and serves real-time
navigation instructions (steer, speed hints, maneuver, arrival FSM, replanning).
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

from fastapi import FastAPI, HTTPException

from navigation.gps_tracker import GPSTracker
from navigation.offline_router import OfflineCacheRouter, infer_maneuver
from navigation.route_follower import NavigationState, RouteFollower
from schemas import (
    PlanRouteRequest,
    PlanRouteResponse,
    StepNavigationRequest,
    StepNavigationResponse,
)

MAX_ACTIVE_SESSIONS = 4  # Keep RAM footprint tiny on resource-constrained devices


@dataclass
class NavigationSession:
    route_id: str
    map_name: str
    follower: RouteFollower
    tracker: GPSTracker


app = FastAPI(
    title="CARLA Edge Node 2 - Navigation Server",
    version="1.0.0",
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


@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "Node 2 Edge Navigation Server",
        "loaded_map": router._cached_map_key,
        "active_sessions": len(sessions),
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


@app.get("/maps")
def list_maps():
    return {"supported_maps": router.list_supported_maps()}


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

    # Evict oldest session if over limit to bound memory usage
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


@app.post("/navigation/step", response_model=StepNavigationResponse)
def step_navigation(req: StepNavigationRequest):
    session = sessions.get(req.route_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {req.route_id}")

    # Mark session as recently used
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
    if needs_replan and session.follower.destination is not None:
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
    )


@app.get("/telemetry/{route_id}")
def get_telemetry(route_id: str) -> Dict:
    session = sessions.get(route_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {route_id}")
    data = session.tracker.get_telemetry()
    data["nav_state"] = session.follower.state.value
    data["remaining_distance_m"] = round(session.follower.remaining_distance_m, 2)
    return data
