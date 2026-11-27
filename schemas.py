"""Pydantic schemas for Edge Node 2 Navigation Server."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PlanRouteRequest(BaseModel):
    map_name: str = Field(..., description="Active CARLA map name (e.g. Town03, Town01)")
    origin: List[float] = Field(
        ...,
        min_length=2,
        description="Vehicle's current standing location [x, y] or [x, y, yaw_deg]",
    )
    destination: Optional[List[float]] = Field(
        default=None,
        description="Optional destination [x, y]. If omitted, uses map default destination.",
    )
    target_speed_kmh: float = Field(default=25.0, ge=5.0, le=80.0)
    arrival_radius_m: float = Field(default=2.5, ge=1.0, le=10.0)


class PlanRouteResponse(BaseModel):
    status: str
    route_id: str
    map_name: str
    origin: List[float]
    destination: List[float]
    total_distance_m: float
    waypoint_count: int
    waypoints: List[List[float]]  # [[x, y, yaw_deg], ...]
    maneuvers: List[str]


class StepNavigationRequest(BaseModel):
    route_id: str
    x: float
    y: float
    yaw_deg: float
    speed_kmh: float
    delta_s: float = Field(default=0.05, gt=0.0, le=1.0)
    failsafe_blocked: bool = False


class StepNavigationResponse(BaseModel):
    route_id: str
    nav_steer: float
    throttle_hint: float
    brake_hint: float
    target_speed_kmh: float
    maneuver: str
    nav_state: str
    remaining_distance_m: float
    route_progress_pct: float
    needs_replan: bool
    new_waypoints: Optional[List[List[float]]] = None
