import json
import os
import tempfile
import unittest

from visualization.minimap import (
    DEFAULT_ROAD_CACHE_PATH,
    Minimap,
    map_cache_key,
    road_cache_path_for_map,
    select_start_destination,
)


class TestMinimapMapSelection(unittest.TestCase):
    def test_map_paths_are_normalized_and_map_specific(self):
        self.assertEqual(map_cache_key("Carla/Maps/Town03"), "town03")
        self.assertEqual(
            road_cache_path_for_map("Carla/Maps/Town03"),
            DEFAULT_ROAD_CACHE_PATH,
        )
        self.assertTrue(
            road_cache_path_for_map("Carla/Maps/Town10HD_Opt").endswith(
                "town10hd_opt_roads_cache.json"
            )
        )

    def test_preferred_start_snaps_to_an_interior_road_point(self):
        roads = [
            [(-80, 0), (0, 0), (80, 0)],
            [(0, -80), (0, 80)],
        ]
        start, destination = select_start_destination(
            roads,
            (-100, 100, -100, 100),
            preferred_start=(77, 2),
        )

        self.assertEqual(start, (80, 0))
        self.assertIn(destination, [point for segment in roads for point in segment])
        self.assertGreater(
            (destination[0] - start[0]) ** 2 + (destination[1] - start[1]) ** 2,
            20 ** 2,
        )

    def test_route_markers_are_selected_from_interior_road_points(self):
        cache = {
            "map": "Town01",
            "bounds": {"min_x": -100, "max_x": 100, "min_y": -100, "max_y": 100},
            "roads": [
                [[-80, 0], [0, 0], [80, 0]],
                [[0, -80], [0, 80]],
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "town01_roads_cache.json")
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(cache, cache_file)

            minimap = Minimap(
                map_name="Carla/Maps/Town01",
                road_cache_path=cache_path,
            )

        self.assertEqual(minimap.start_location, (0, 0))
        self.assertNotEqual(minimap.start_location, minimap.destination_location)
        for marker in (minimap.start_location, minimap.destination_location):
            self.assertTrue(
                any(marker in segment for segment in minimap.roads),
                f"marker {marker} is not on the road geometry",
            )
            self.assertTrue(
                -90 < marker[0] < 90 and -90 < marker[1] < 90,
                f"marker {marker} is too close to the map border",
            )

    def test_wrong_map_cache_is_rejected(self):
        cache = {
            "map": "Town03",
            "bounds": {"min_x": -10, "max_x": 10, "min_y": -10, "max_y": 10},
            "roads": [[[-5, 0], [5, 0]]],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "wrong_map.json")
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(cache, cache_file)

            with self.assertRaisesRegex(ValueError, "not Town01"):
                Minimap(
                    map_name="Town01",
                    road_cache_path=cache_path,
                )


if __name__ == "__main__":
    unittest.main()
