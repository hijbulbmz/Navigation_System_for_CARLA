"""Unit tests for GPSTracker odometry, geolocation, and distance calculations."""

import unittest
from types import SimpleNamespace

from navigation.arrival_manager import NavigationState
from navigation.gps_tracker import GPSTracker


class FakeCarlaGeoLocation:
    def __init__(self, latitude, longitude, altitude=0.0):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude


class FakeCarlaMap:
    def __init__(self, lat_offset=0.0001, lon_offset=0.0002):
        self.lat_offset = lat_offset
        self.lon_offset = lon_offset

    def transform_to_geolocation(self, location):
        return FakeCarlaGeoLocation(
            latitude=location.x * self.lat_offset,
            longitude=location.y * self.lon_offset,
            altitude=getattr(location, "z", 0.0),
        )


class TestGPSTracker(unittest.TestCase):
    def setUp(self):
        self.destination = (100.0, 100.0, 0.0)
        self.fake_map = FakeCarlaMap()
        self.tracker = GPSTracker(
            destination=self.destination,
            planned_distance_m=200.0,
            carla_map=self.fake_map,
        )

    def test_initial_values(self):
        self.assertEqual(self.tracker.driven_distance_m, 0.0)
        self.assertEqual(self.tracker.route_progress_pct, 0.0)
        self.assertEqual(self.tracker.navigation_state, NavigationState.NAVIGATING)

    def test_update_tracks_odometry(self):
        # Step 1: at (0, 0, 0)
        self.tracker.update(
            current_location=(0.0, 0.0, 0.0),
            current_speed_kmh=10.0,
            heading_yaw_deg=45.0,
            sim_time_s=1.0,
        )
        self.assertEqual(self.tracker.driven_distance_m, 0.0)
        # Distance to destination from (0, 0) to (100, 100) is sqrt(20000) ~ 141.42m
        self.assertAlmostEqual(self.tracker.distance_to_destination_m, 141.42, places=1)

        # Step 2: move to (10, 0, 0) -> +10m driven
        self.tracker.update(
            current_location=(10.0, 0.0, 0.0),
            current_speed_kmh=15.0,
            heading_yaw_deg=45.0,
            sim_time_s=2.0,
        )
        self.assertAlmostEqual(self.tracker.driven_distance_m, 10.0)
        # Progress: 10m / 200m = 5%
        self.assertAlmostEqual(self.tracker.route_progress_pct, 5.0)

        # Step 3: move to (10, 30, 0) -> +30m driven -> total 40m
        self.tracker.update(
            current_location=(10.0, 30.0, 0.0),
            current_speed_kmh=20.0,
            heading_yaw_deg=90.0,
            sim_time_s=3.0,
        )
        self.assertAlmostEqual(self.tracker.driven_distance_m, 40.0)
        # Progress: 40m / 200m = 20%
        self.assertAlmostEqual(self.tracker.route_progress_pct, 20.0)

    def test_route_progress_clamps_between_0_and_100(self):
        # Drive over the planned distance
        self.tracker.update((0.0, 0.0, 0.0), 20.0, 0.0, 1.0)
        self.tracker.update((300.0, 0.0, 0.0), 20.0, 0.0, 2.0)
        self.assertEqual(self.tracker.driven_distance_m, 300.0)
        self.assertEqual(self.tracker.route_progress_pct, 100.0)

    def test_geolocation_conversion_with_map(self):
        self.tracker.update(
            current_location=(50.0, -25.0, 1.0),
            current_speed_kmh=10.0,
            heading_yaw_deg=0.0,
            sim_time_s=1.0,
        )
        # 50 * 0.0001 = 0.005
        self.assertAlmostEqual(self.tracker.current_gps_lat, 0.005)
        # -25 * 0.0002 = -0.005
        self.assertAlmostEqual(self.tracker.current_gps_lon, -0.005)

    def test_geolocation_fallback_without_map(self):
        tracker_no_map = GPSTracker(
            destination=(50.0, 50.0, 0.0),
            planned_distance_m=100.0,
            carla_map=None,
        )
        tracker_no_map.update(
            current_location=(0.0, 0.0, 0.0),
            current_speed_kmh=0.0,
            heading_yaw_deg=0.0,
            sim_time_s=1.0,
        )
        # Fallback uses approximate CARLA Town03 reference lat/lon
        self.assertIsInstance(tracker_no_map.current_gps_lat, float)
        self.assertIsInstance(tracker_no_map.current_gps_lon, float)
        self.assertNotEqual(tracker_no_map.current_gps_lat, 0.0)

    def test_telemetry_dict(self):
        self.tracker.update(
            current_location=(10.0, 20.0, 1.0),
            current_speed_kmh=24.5,
            heading_yaw_deg=35.0,
            sim_time_s=4.5,
        )
        self.tracker.set_navigation_state(NavigationState.APPROACHING)
        telemetry = self.tracker.get_telemetry()
        self.assertEqual(telemetry["nav_state"], "APPROACHING")
        self.assertAlmostEqual(telemetry["speed_kmh"], 24.5)
        self.assertAlmostEqual(telemetry["heading_yaw_deg"], 35.0)
        self.assertAlmostEqual(telemetry["x"], 10.0)
        self.assertAlmostEqual(telemetry["y"], 20.0)
        self.assertFalse(telemetry["is_arrived"])


if __name__ == "__main__":
    unittest.main()

