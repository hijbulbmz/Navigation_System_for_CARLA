# Navigation System for CARLA

A self-contained navigation stack for CARLA Simulator autonomous vehicle projects.
Extracted from the **Autonomous Vehicle Training** project.

---

## Folder Structure

```
Node2/
├── navigation/
│   ├── router.py           # Road route planner (CARLA GlobalRoutePlanner wrapper)
│   ├── route_follower.py   # Per-tick steering/throttle/brake controller
│   ├── arrival_manager.py  # Destination arrival FSM (approach → stop → arrived)
│   ├── gps_tracker.py      # Vehicle location, speed, heading, progress tracker
│   ├── checkpoints.py      # Town03 checkpoint definitions (P1–P10)
│   ├── fixed_route.py      # Legacy Town03 spawn-point route builder
│   └── __init__.py         # Package exports
│
├── visualization/
│   ├── minimap.py          # 2D top-down road map + route overlay (OpenCV)
│   ├── hud.py              # On-screen telemetry HUD
│   └── __init__.py
│
├── config/
│   ├── route_town03.json         # Predefined start/destination for Town03
│   ├── checkpoints_town03.json   # 10 checkpoint coordinates for Town03
│   ├── town01_roads_cache.json   # Road geometry for Town01 (offline minimap)
│   ├── town02_roads_cache.json   # Road geometry for Town02
│   ├── town03_roads_cache.json   # Road geometry for Town03
│   ├── town04_roads_cache.json   # Road geometry for Town04
│   ├── town05_roads_cache.json   # Road geometry for Town05
│   ├── town10hd_roads_cache.json # Road geometry for Town10HD
│   └── opendrivemap_roads_cache.json
│
├── tests/                  # Offline unit tests (no CARLA server needed)
│   ├── test_router_and_follower.py
│   ├── test_arrival_manager.py
│   ├── test_gps_tracker.py
│   ├── test_minimap_offline.py
│   └── test_minimap_map_selection.py
│
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `numpy >= 1.22`
- `opencv-python >= 4.6`
- `torch >= 2.0`
- `networkx >= 2.8`
- **CARLA Python API** — installed from your CARLA simulator's `PythonAPI/` folder

---

## How to Use in Your Project

### Step 1: Import the navigation modules

```python
from navigation.router import RoadRouter, RoutePlan
from navigation.route_follower import RouteFollower, NavigationState
from navigation.arrival_manager import ArrivalManager
from navigation.gps_tracker import GPSTracker
```

### Step 2: Set up at mission start (once)

```python
# Connect to CARLA
import carla
client = carla.Client("localhost", 2000)
world = client.get_world()

# Initialize router and planner
router = RoadRouter(world.get_map(), resolution_m=2.0)
route_follower = RouteFollower(target_speed_kmh=25.0, arrival_radius_m=2.5)
gps_tracker = GPSTracker(carla_map=world.get_map())

# Plan a route: (start_x, start_y) → (dest_x, dest_y)
plan = router.plan_route(
    origin=(start_x, start_y),
    destination=(dest_x, dest_y)
)

if plan.success:
    route_follower.set_route(plan)
    gps_tracker.set_destination(plan.destination_location)
    print(f"Route planned: {plan.total_distance_m:.0f} m, {len(plan.waypoints)} waypoints")
else:
    print(f"Route planning failed: {plan.error_message}")
```

### Step 3: Call every simulation tick

```python
# Inside your CARLA simulation loop:
transform = vehicle.get_transform()
velocity = vehicle.get_velocity()

# Update GPS tracker
gps_tracker.update(transform, velocity)

# Get navigation control command
steer, throttle, brake, needs_replan = route_follower.step(
    current_x=gps_tracker.x,
    current_y=gps_tracker.y,
    current_yaw_deg=gps_tracker.heading_yaw,
    speed_kmh=gps_tracker.speed_kmh,
    delta_s=0.05,            # simulation timestep in seconds
    failsafe_blocked=False   # set True if obstacle detector triggers
)

# (Optional) Auto-replan if vehicle goes off-route
if needs_replan and route_follower.destination is not None:
    replan = router.plan_route(
        origin=(gps_tracker.x, gps_tracker.y),
        destination=route_follower.destination,
        origin_heading=gps_tracker.heading_yaw
    )
    route_follower.set_route(replan)

# Apply to vehicle
vehicle.apply_control(carla.VehicleControl(
    steer=steer,
    throttle=throttle,
    brake=brake
))

# Check arrival
if route_follower.state == NavigationState.ARRIVED:
    print("Destination reached!")
```

---

## Navigation States

| State | Meaning |
|---|---|
| `IDLE` | No route loaded |
| `FOLLOWING` | Actively tracking waypoints |
| `REPLANNING` | Off-route deviation detected, requesting new plan |
| `BLOCKED` | Emergency stop active (obstacle/failsafe) |
| `ARRIVED` | Vehicle stopped at destination |
| `FAILED` | Route failed or planner error |

---

## Run Offline Tests (No CARLA Needed)

```bash
cd C:\Users\hijbu\Desktop\Node2
python -m unittest discover -s tests
```

Tests cover: route planning logic, steering direction, arrival braking, FSM transitions, minimap rendering, GPS tracking, and progress calculation.

---

## Supported CARLA Maps

Road geometry caches are pre-built for all standard CARLA maps:

| Map | Cache File |
|---|---|
| Town01 | `config/town01_roads_cache.json` |
| Town02 | `config/town02_roads_cache.json` |
| Town03 | `config/town03_roads_cache.json` |
| Town04 | `config/town04_roads_cache.json` |
| Town05 | `config/town05_roads_cache.json` |
| Town10HD | `config/town10hd_roads_cache.json` |
| OpenDriveMap | `config/opendrivemap_roads_cache.json` |

---

## Architecture

```
CARLA World Map
      │
      ▼
 RoadRouter.plan_route()
      │  (A* search on OpenDRIVE graph)
      ▼
 RoutePlan (waypoints + polyline_2d)
      │
      ▼
 RouteFollower.set_route()
      │
      ▼ (every tick)
 RouteFollower.step()
      │
      ├── steer (Pure Pursuit + Stanley)
      ├── throttle (speed profile)
      └── brake (deceleration / arrival stop)
            │
            ▼
      vehicle.apply_control()
```
