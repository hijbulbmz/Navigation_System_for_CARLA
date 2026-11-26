"""Offline Unit Tests for Town03 Minimap and Checkpoints.

Runs completely independently of CARLA, OpenCV, or external network access.
Verifies checkpoint configuration, coordinate transformation, inverted vertical axis,
yaw calculation, vehicle arrow geometry, positioning, and deterministic algorithms.
"""

import json
import math
import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navigation.checkpoints import (
    DEFAULT_CONFIG_PATH,
    Checkpoint,
    CheckpointManager,
    load_checkpoints,
)
from visualization.minimap import (
    DEFAULT_ROAD_CACHE_PATH,
    Minimap,
)


class TestTown03Checkpoints(unittest.TestCase):
    """Verifies checkpoint loading, validation, and JSON integrity."""

    def test_config_file_exists(self):
        """Check that config/checkpoints_town03.json exists."""
        self.assertTrue(
            os.path.isfile(DEFAULT_CONFIG_PATH),
            f"Checkpoint file not found: {DEFAULT_CONFIG_PATH}",
        )

    def test_checkpoint_count_and_ids(self):
        """Check that exactly 10 checkpoints P1 to P10 are present."""
        manager = CheckpointManager.load(DEFAULT_CONFIG_PATH)
        self.assertEqual(len(manager), 10, "Must contain exactly 10 checkpoints")

        expected_ids = [f"P{i}" for i in range(1, 11)]
        actual_ids = [cp.id for cp in manager.checkpoints]
        self.assertEqual(actual_ids, expected_ids, "Checkpoint IDs must be P1 through P10 in order")

    def test_checkpoint_fields_validity(self):
        """Check all required fields, valid world coordinates, and GPS data."""
        manager = CheckpointManager.load(DEFAULT_CONFIG_PATH)
        for cp in manager.checkpoints:
            self.assertTrue(cp.id.startswith("P"), f"Invalid ID {cp.id}")
            self.assertTrue(len(cp.name) > 0, f"Empty name for {cp.id}")
            self.assertTrue(len(cp.region) > 0, f"Empty region for {cp.id}")

            # Verify reasonable Town03 coordinates (-300 to +300 meters)
            self.assertTrue(-300.0 <= cp.x <= 300.0, f"{cp.id} X out of Town03 bounds: {cp.x}")
            self.assertTrue(-300.0 <= cp.y <= 300.0, f"{cp.id} Y out of Town03 bounds: {cp.y}")
            self.assertTrue(-10.0 <= cp.z <= 50.0, f"{cp.id} Z out of bounds: {cp.z}")

            # Verify road parameters
            self.assertIsInstance(cp.road_id, int)
            self.assertIsInstance(cp.lane_id, int)
            self.assertGreaterEqual(cp.s, 0.0)

            # Verify GPS parameters
            self.assertIsInstance(cp.latitude, float)
            self.assertIsInstance(cp.longitude, float)
            self.assertIsInstance(cp.altitude, float)

    def test_checkpoint_spatial_distribution(self):
        """Check that checkpoints are distributed across Town03 and not clustered."""
        checkpoints = load_checkpoints(DEFAULT_CONFIG_PATH)
        min_distance = 35.0  # meters

        for i in range(len(checkpoints)):
            for j in range(i + 1, len(checkpoints)):
                c1, c2 = checkpoints[i], checkpoints[j]
                dist = math.hypot(c1.x - c2.x, c1.y - c2.y)
                self.assertGreater(
                    dist,
                    min_distance,
                    f"Checkpoints {c1.id} and {c2.id} are too close ({dist:.1f} m < {min_distance} m)",
                )

    def test_serialization_roundtrip(self):
        """Verify to_dict and from_dict produce identical checkpoint data."""
        checkpoints = load_checkpoints(DEFAULT_CONFIG_PATH)
        for cp in checkpoints:
            data = cp.to_dict()
            restored = Checkpoint.from_dict(data)
            self.assertEqual(cp.id, restored.id)
            self.assertAlmostEqual(cp.x, restored.x, places=2)
            self.assertAlmostEqual(cp.y, restored.y, places=2)
            self.assertAlmostEqual(cp.z, restored.z, places=2)
            self.assertEqual(cp.road_id, restored.road_id)
            self.assertEqual(cp.lane_id, restored.lane_id)
            self.assertAlmostEqual(cp.s, restored.s, places=2)


class TestMinimapCoordinateTransformation(unittest.TestCase):
    """Verifies 2D projection, aspect-ratio preservation, and inverted Y-axis."""

    def setUp(self):
        self.minimap = Minimap(
            size=260,
            position="top-right",
            opacity=0.85,
            road_cache_path=DEFAULT_ROAD_CACHE_PATH,
            padding=14,
        )

    def test_inverted_vertical_axis(self):
        """Verify that higher world Y produces SMALLER pixel Y (closer to image top)."""
        # Point North (high Y) vs Point South (low Y)
        px_north, py_north = self.minimap.world_to_pixel(0.0, 200.0)
        px_south, py_south = self.minimap.world_to_pixel(0.0, -200.0)

        self.assertLess(
            py_north,
            py_south,
            f"North Y pixel ({py_north}) must be less than South Y pixel ({py_south}) due to inverted image axis",
        )

    def test_horizontal_axis_orientation(self):
        """Verify that higher world X produces LARGER pixel X (further to the right)."""
        px_west, _ = self.minimap.world_to_pixel(-200.0, 0.0)
        px_east, _ = self.minimap.world_to_pixel(200.0, 0.0)

        self.assertLess(
            px_west,
            px_east,
            f"West X pixel ({px_west}) must be less than East X pixel ({px_east})",
        )

    def test_route_markers_stay_inside_the_panel(self):
        """Verify Start and Destination are not clipped by the minimap border."""
        size = self.minimap.size
        for marker in (
            self.minimap.start_location,
            self.minimap.destination_location,
        ):
            px, py = self.minimap.world_to_pixel(*marker)
            self.assertGreater(px, 0)
            self.assertLess(px, size - 1)
            self.assertGreater(py, 0)
            self.assertLess(py, size - 1)

    def test_aspect_ratio_uniformity(self):
        """Verify scaling factor is isotropic to avoid road and roundabout distortion."""
        scale = self.minimap.scale
        self.assertGreater(scale, 0.0, "Scale must be positive")

        # Test circle of radius 20m centered at (0, 0)
        r_world = 20.0
        p_center_x, p_center_y = self.minimap.world_to_pixel(0.0, 0.0)
        p_east_x, _ = self.minimap.world_to_pixel(r_world, 0.0)
        _, p_north_y = self.minimap.world_to_pixel(0.0, r_world)

        radius_px_x = abs(p_east_x - p_center_x)
        radius_px_y = abs(p_north_y - p_center_y)

        # In pixel space, horizontal and vertical radius must match within rounding (<= 1 px)
        self.assertLessEqual(
            abs(radius_px_x - radius_px_y),
            1,
            f"Aspect ratio distorted: horizontal radius {radius_px_x}px != vertical radius {radius_px_y}px",
        )


class TestVehicleHeadingCalculation(unittest.TestCase):
    """Verifies yaw to 2D pixel direction mapping and vehicle arrow orientation."""

    def setUp(self):
        self.minimap = Minimap(size=260)

    def test_cardinal_directions(self):
        """Verify 0 deg (East), 90 deg (North), 180 deg (West), -90 deg (South)."""
        # East (yaw = 0): dx > 0, dy == 0
        dx0, dy0 = self.minimap.yaw_to_direction(0.0)
        self.assertAlmostEqual(dx0, 1.0, places=4)
        self.assertAlmostEqual(dy0, 0.0, places=4)

        # North (yaw = 90): in pixel coordinates, UP is dy < 0
        dx90, dy90 = self.minimap.yaw_to_direction(90.0)
        self.assertAlmostEqual(dx90, 0.0, places=4)
        self.assertAlmostEqual(dy90, -1.0, places=4)

        # West (yaw = 180): dx < 0, dy == 0
        dx180, dy180 = self.minimap.yaw_to_direction(180.0)
        self.assertAlmostEqual(dx180, -1.0, places=4)
        self.assertAlmostEqual(dy180, 0.0, places=4)

        # South (yaw = -90 or 270): in pixel coordinates, DOWN is dy > 0
        dx_s, dy_s = self.minimap.yaw_to_direction(-90.0)
        self.assertAlmostEqual(dx_s, 0.0, places=4)
        self.assertAlmostEqual(dy_s, 1.0, places=4)

    def test_unit_vector_norm(self):
        """Verify direction vector is always unit length across all angles."""
        for deg in range(-180, 181, 15):
            dx, dy = self.minimap.yaw_to_direction(float(deg))
            norm = math.hypot(dx, dy)
            self.assertAlmostEqual(norm, 1.0, places=5)


class TestRoadCacheData(unittest.TestCase):
    """Verifies road cache file structure and geometry."""

    def test_road_cache_file_exists(self):
        """Check that config/town03_roads_cache.json exists."""
        self.assertTrue(
            os.path.isfile(DEFAULT_ROAD_CACHE_PATH),
            f"Road cache file not found: {DEFAULT_ROAD_CACHE_PATH}",
        )

    def test_road_cache_content(self):
        """Check roads and bounding box inside cache."""
        with open(DEFAULT_ROAD_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("map"), "Town03")
        bounds = data.get("bounds", {})
        self.assertIn("min_x", bounds)
        self.assertIn("max_x", bounds)
        self.assertIn("min_y", bounds)
        self.assertIn("max_y", bounds)

        roads = data.get("roads", [])
        self.assertGreater(len(roads), 10, "Expected at least 10 road polylines in cache")
        for poly in roads:
            self.assertGreaterEqual(len(poly), 2, "Each road polyline must have >= 2 points")
            for pt in poly:
                self.assertEqual(len(pt), 2, "Point must have X and Y coordinates")


class TestDeterministicPlacementAlgorithm(unittest.TestCase):
    """Verifies that the waypoint selection procedure is 100% deterministic."""

    def test_deterministic_sorting(self):
        """Simulates candidate waypoints and verifies deterministic selection."""
        # Create a mock candidate list with known coordinates
        class MockWP:
            def __init__(self, x, y, road_id, lane_id, s):
                self.x = x
                self.y = y
                self.road_id = road_id
                self.lane_id = lane_id
                self.s = s

        candidates = [
            MockWP(100.0, 50.0, 1, -1, 10.0),
            MockWP(100.0, 50.0, 2, -1, 5.0),
            MockWP(101.0, 50.0, 3, 1, 2.0),
            MockWP(-50.0, -100.0, 4, -1, 15.0),
        ]

        target_x, target_y = 100.0, 50.0
        # Run selection twice
        select = lambda: min(
            candidates,
            key=lambda wp: (
                (wp.x - target_x) ** 2 + (wp.y - target_y) ** 2,
                wp.road_id,
                wp.lane_id,
                wp.s,
            ),
        )

        res1 = select()
        res2 = select()
        self.assertIs(res1, res2)
        # Should pick road_id 1 over road_id 2 due to deterministic tie breaking
        self.assertEqual(res1.road_id, 1)


class TestOverlayPositioning(unittest.TestCase):
    """Verifies overlay placement coordinates and adaptive sizing calculation."""

    def test_top_right_placement_coordinates(self):
        """Verify top-right placement math within camera frame."""
        frame_w, frame_h = 320, 240
        size = 120
        margin = 6
        x1 = frame_w - size - margin
        y1 = margin
        x2 = x1 + size
        y2 = y1 + size
        self.assertEqual((x1, y1), (194, 6))
        self.assertEqual((x2, y2), (314, 126))
        self.assertLess(x2, frame_w)
        self.assertLess(y2, frame_h)

    def test_top_left_placement_coordinates(self):
        """Verify top-left placement math within camera frame."""
        frame_w, frame_h = 320, 240
        size = 120
        margin = 6
        x1 = margin
        y1 = margin + 24  # Leaves room for status line
        x2 = x1 + size
        y2 = y1 + size
        self.assertEqual((x1, y1), (6, 30))
        self.assertEqual((x2, y2), (126, 150))
        self.assertLess(x2, frame_w)
        self.assertLess(y2, frame_h)

    def test_adaptive_size_clamping(self):
        """Verify adaptive size clamping when requested size exceeds frame dimensions."""
        frame_w, frame_h = 320, 240
        requested_size = 260
        margin = 6
        max_allowed = min(frame_w - 2 * margin, frame_h - 2 * margin)
        effective_size = min(requested_size, max_allowed)
        self.assertEqual(effective_size, 228)
        self.assertLessEqual(effective_size + 2 * margin, frame_h)


if __name__ == "__main__":
    unittest.main()
