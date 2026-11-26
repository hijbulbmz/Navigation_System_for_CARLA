import math
import unittest
from types import SimpleNamespace

from navigation.router import RoutePlan, check_route_departure
from navigation.route_follower import RouteFollower, NavigationState
from visualization.minimap import Minimap


class FakeLocation:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeTransform:
    def __init__(self, x, y, yaw=0.0):
        self.location = FakeLocation(x, y)
        self.rotation = SimpleNamespace(yaw=yaw)

    def get_forward_vector(self):
        rad = math.radians(self.rotation.yaw)
        return FakeLocation(math.cos(rad), math.sin(rad), 0.0)

    def get_right_vector(self):
        rad = math.radians(self.rotation.yaw + 90.0)
        return FakeLocation(math.cos(rad), math.sin(rad), 0.0)


class FakeWaypoint:
    def __init__(self, x, y, yaw=0.0):
        self.transform = FakeTransform(x, y, yaw)


class TestRouterAndFollower(unittest.TestCase):
    def setUp(self):
        # Create a straight line route along x-axis from (0,0) to (100,0)
        self.straight_waypoints = [
            FakeWaypoint(float(x), 0.0, 0.0) for x in range(0, 105, 5)
        ]
        self.road_options = [SimpleNamespace(name="LANEFOLLOW") for _ in self.straight_waypoints]
        self.polyline_2d = [(float(x), 0.0) for x in range(0, 105, 5)]

        self.straight_plan = RoutePlan(
            success=True,
            total_distance_m=100.0,
            waypoints=self.straight_waypoints,
            road_options=self.road_options,
            polyline_2d=self.polyline_2d,
            start_location=(0.0, 0.0),
            destination_location=(100.0, 0.0),
        )

    def test_route_plan_properties(self):
        self.assertEqual(len(self.straight_plan.polyline_2d), len(self.straight_waypoints))
        self.assertAlmostEqual(self.straight_plan.total_distance_m, 100.0, places=1)
        self.assertAlmostEqual(self.straight_plan.polyline_2d[0][0], 0.0)
        self.assertAlmostEqual(self.straight_plan.polyline_2d[-1][0], 100.0)

    def test_route_departure_detection(self):
        polyline = self.straight_plan.polyline_2d
        # Close to route: 1.0m off in y
        is_departed, dist, idx = check_route_departure(20.0, 1.0, polyline, current_index=4)
        self.assertFalse(is_departed)
        self.assertAlmostEqual(dist, 1.0, places=2)

        # Far from route: 6.0m off in y
        is_departed_far, dist_far, _ = check_route_departure(20.0, 6.0, polyline, current_index=4)
        self.assertTrue(is_departed_far)
        self.assertAlmostEqual(dist_far, 6.0, places=2)

    def test_follower_state_transitions(self):
        follower = RouteFollower(target_speed_kmh=30.0, arrival_radius_m=2.5)
        self.assertEqual(follower.state, NavigationState.IDLE)

        follower.set_route(self.straight_plan)
        self.assertEqual(follower.state, NavigationState.FOLLOWING)
        self.assertGreater(follower.remaining_distance_m, 90.0)

        # Step along route: normal driving
        steer, throttle, brake, replan = follower.step(
            current_x=0.0, current_y=0.0, current_yaw_deg=0.0, speed_kmh=15.0
        )
        self.assertEqual(follower.state, NavigationState.FOLLOWING)
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)
        self.assertAlmostEqual(steer, 0.0, delta=0.05)
        self.assertFalse(replan)

        # Step when failsafe obstacle is detected
        steer, throttle, brake, replan = follower.step(
            current_x=10.0, current_y=0.0, current_yaw_deg=0.0, speed_kmh=10.0,
            failsafe_blocked=True
        )
        self.assertEqual(follower.state, NavigationState.BLOCKED)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)
        self.assertFalse(replan)

        # Release failsafe -> resumes following
        steer, throttle, brake, replan = follower.step(
            current_x=10.0, current_y=0.0, current_yaw_deg=0.0, speed_kmh=0.0,
            failsafe_blocked=False
        )
        self.assertEqual(follower.state, NavigationState.FOLLOWING)

        # Destination reached: within arrival radius at slow speed
        steer, throttle, brake, replan = follower.step(
            current_x=99.5, current_y=0.0, current_yaw_deg=0.0, speed_kmh=1.0
        )
        self.assertEqual(follower.state, NavigationState.ARRIVED)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)
        self.assertFalse(replan)

    def test_destination_arrival_braking_and_full_stop(self):
        follower = RouteFollower(target_speed_kmh=25.0, arrival_radius_m=2.5)
        follower.set_route(self.straight_plan)

        # 1. Inside arrival radius (distance = 1.0m <= 2.5m) but carrying speed (8 km/h) -> must command full brake
        steer, throttle, brake, replan = follower.step(
            current_x=99.0, current_y=0.0, current_yaw_deg=0.0, speed_kmh=8.0
        )
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)

        # 2. Vehicle halts (speed drops below 3.0 km/h) -> transitions to ARRIVED
        steer, throttle, brake, replan = follower.step(
            current_x=99.2, current_y=0.0, current_yaw_deg=0.0, speed_kmh=0.5
        )
        self.assertEqual(follower.state, NavigationState.ARRIVED)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)

        # 3. Permanently holds full stop at destination
        steer, throttle, brake, replan = follower.step(
            current_x=99.2, current_y=0.0, current_yaw_deg=0.0, speed_kmh=0.0
        )
        self.assertEqual(follower.state, NavigationState.ARRIVED)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 1.0)

    def test_lateral_steering_correction(self):
        follower = RouteFollower(target_speed_kmh=30.0)
        follower.set_route(self.straight_plan)

        # Vehicle is at y = 2.0 (target is at y=0, dy=-2). Should steer left (negative).
        steer, _, _, _ = follower.step(
            current_x=20.0, current_y=2.0, current_yaw_deg=0.0, speed_kmh=20.0
        )
        self.assertLess(steer, 0.0)

        # Vehicle is at y = -2.0 (target is at y=0, dy=+2). Should steer right (positive).
        steer_right, _, _, _ = follower.step(
            current_x=20.0, current_y=-2.0, current_yaw_deg=0.0, speed_kmh=20.0
        )
        self.assertGreater(steer_right, 0.0)

    def test_minimap_route_drawing_integration(self):
        minimap = Minimap(size=260)
        route_pts = self.straight_plan.polyline_2d
        minimap.set_route(route_pts, route_distance_m=self.straight_plan.total_distance_m)

        self.assertEqual(len(minimap.route_points), len(route_pts))
        self.assertAlmostEqual(minimap.route_distance_m, 100.0, places=1)
        self.assertEqual(minimap.navigation_status, "FOLLOWING")

        # Update progress
        minimap.update_progress(current_x=50.0, current_y=0.0, remaining_distance_m=50.0, status="FOLLOWING")
        self.assertGreaterEqual(minimap.current_route_index, 5)
        self.assertAlmostEqual(minimap.remaining_distance_m, 50.0)

        # Clear route
        minimap.clear_route()
        self.assertEqual(len(minimap.route_points), 0)
        self.assertEqual(minimap.navigation_status, "IDLE")


if __name__ == "__main__":
    unittest.main()
