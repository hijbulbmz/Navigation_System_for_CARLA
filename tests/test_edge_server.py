"""Unit and integration tests for Edge Node 2 server and offline cache router."""

import unittest
from fastapi.testclient import TestClient

from edge_server import app, router


class TestEdgeNavigationServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_and_maps(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "ONLINE")

        maps_res = self.client.get("/maps")
        self.assertEqual(maps_res.status_code, 200)
        maps = maps_res.json()["supported_maps"]
        self.assertIn("Town01", maps)
        self.assertIn("Town03", maps)

    def test_plan_route_from_vehicle_standing_location_default_dest(self):
        # Vehicle standing at (-84.79, 27.59) in Town03, no destination specified -> uses default
        res = self.client.post(
            "/plan_route",
            json={
                "map_name": "Town03",
                "origin": [-84.79, 27.59, 0.0],
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertGreater(data["waypoint_count"], 20)
        self.assertGreater(data["total_distance_m"], 50.0)
        self.assertEqual(len(data["waypoints"][0]), 3)  # [x, y, yaw]
        self.assertEqual(len(data["maneuvers"]), data["waypoint_count"])

    def test_plan_route_custom_destination_and_step_streaming(self):
        # Vehicle standing at (92.0, 76.0) in Town01 heading toward (334.0, 252.0)
        res = self.client.post(
            "/plan_route",
            json={
                "map_name": "Town01",
                "origin": [92.0, 76.0, 0.0],
                "destination": [334.0, 252.0],
            },
        )
        self.assertEqual(res.status_code, 200)
        plan = res.json()
        route_id = plan["route_id"]
        self.assertGreater(plan["waypoint_count"], 50)

        # Single-map RAM cache check: Town01 replaced Town03 in memory
        self.assertEqual(router._cached_map_key, "town01")

        # 1. Step at first waypoint
        w0 = plan["waypoints"][0]
        step1 = self.client.post(
            "/navigation/step",
            json={
                "route_id": route_id,
                "x": w0[0],
                "y": w0[1],
                "yaw_deg": w0[2],
                "speed_kmh": 15.0,
            },
        )
        self.assertEqual(step1.status_code, 200)
        s1 = step1.json()
        self.assertEqual(s1["nav_state"], "FOLLOWING")
        self.assertGreaterEqual(s1["nav_steer"], -1.0)
        self.assertLessEqual(s1["nav_steer"], 1.0)
        self.assertFalse(s1["needs_replan"])

        # 2. Step with obstacle failsafe active -> BLOCKED & full brake
        step_blocked = self.client.post(
            "/navigation/step",
            json={
                "route_id": route_id,
                "x": w0[0],
                "y": w0[1],
                "yaw_deg": w0[2],
                "speed_kmh": 10.0,
                "failsafe_blocked": True,
            },
        )
        self.assertEqual(step_blocked.status_code, 200)
        sb = step_blocked.json()
        self.assertEqual(sb["nav_state"], "BLOCKED")
        self.assertEqual(sb["throttle_hint"], 0.0)
        self.assertEqual(sb["brake_hint"], 1.0)

        # 3. Step at destination -> ARRIVED
        dest = plan["destination"]
        step_arrived = self.client.post(
            "/navigation/step",
            json={
                "route_id": route_id,
                "x": dest[0],
                "y": dest[1],
                "yaw_deg": 0.0,
                "speed_kmh": 0.5,
                "failsafe_blocked": False,
            },
        )
        self.assertEqual(step_arrived.status_code, 200)
        sa = step_arrived.json()
        self.assertEqual(sa["nav_state"], "ARRIVED")
        self.assertEqual(sa["maneuver"], "ARRIVED")
        self.assertEqual(sa["brake_hint"], 1.0)

        # 4. Telemetry endpoint
        tel = self.client.get(f"/telemetry/{route_id}")
        self.assertEqual(tel.status_code, 200)
        self.assertEqual(tel.json()["nav_state"], "ARRIVED")


if __name__ == "__main__":
    unittest.main()
