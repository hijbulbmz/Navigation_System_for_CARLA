"""Unit tests for ArrivalManager FSM, deceleration profile, and arrival locking."""

import math
import unittest

from navigation.arrival_manager import ArrivalManager, NavigationState


class TestArrivalManager(unittest.TestCase):
    def setUp(self):
        self.dest = (100.0, 100.0, 0.0)
        self.manager = ArrivalManager(
            destination=self.dest,
            arrival_radius=3.0,
            stop_speed_threshold_kmh=0.5,
            approach_distance=30.0,
            deceleration=1.5,
            cruise_speed_kmh=25.0,
            confirm_frames=3,
        )

    def test_initial_state_far_away(self):
        # 70 meters away -> NAVIGATING
        state, target_speed = self.manager.update(
            current_position=(50.0, 50.0, 0.0),
            current_speed_kmh=25.0,
        )
        self.assertEqual(state, NavigationState.NAVIGATING)
        self.assertAlmostEqual(target_speed, 25.0)
        self.assertFalse(self.manager.is_stopped())
        self.assertFalse(self.manager.is_arrived())

    def test_transition_to_approaching(self):
        # Move to 20 meters away (< 30.0m) -> APPROACHING
        pos_20m = (100.0 - 20.0, 100.0, 0.0)
        state, target_speed = self.manager.update(
            current_position=pos_20m,
            current_speed_kmh=20.0,
        )
        self.assertEqual(state, NavigationState.APPROACHING)
        # v = min(25/3.6, sqrt(2 * 1.5 * 20.0)) * 3.6 = min(6.94, sqrt(60)) * 3.6 = min(6.94, 7.74) * 3.6 = 25.0
        # At 10m away: sqrt(2 * 1.5 * 10) = sqrt(30) = 5.477 m/s = 19.72 km/h
        pos_10m = (100.0 - 10.0, 100.0, 0.0)
        state, target_speed = self.manager.update(
            current_position=pos_10m,
            current_speed_kmh=18.0,
        )
        self.assertEqual(state, NavigationState.APPROACHING)
        expected_speed_kmh = math.sqrt(2.0 * 1.5 * 10.0) * 3.6
        self.assertAlmostEqual(target_speed, expected_speed_kmh, places=1)
        self.assertLess(target_speed, 25.0)

    def test_deceleration_profile_clamping(self):
        # Target speed must be >= 0 and <= cruise speed
        profile_speed = self.manager.compute_target_speed(remaining_distance=50.0)
        self.assertEqual(profile_speed, 25.0)

        profile_zero = self.manager.compute_target_speed(remaining_distance=0.0)
        self.assertEqual(profile_zero, 0.0)

        profile_neg = self.manager.compute_target_speed(remaining_distance=-5.0)
        self.assertEqual(profile_neg, 0.0)

    def test_transition_to_stopping_and_high_speed_protection(self):
        # Inside arrival radius (2.0m away) but still travelling fast (10 km/h)
        pos_inside = (100.0 - 2.0, 100.0, 0.0)
        state, target_speed = self.manager.update(
            current_position=pos_inside,
            current_speed_kmh=10.0,
        )
        # Must be in STOPPING, not ARRIVED
        self.assertEqual(state, NavigationState.STOPPING)
        self.assertEqual(target_speed, 0.0)
        self.assertTrue(self.manager.is_stopped())
        self.assertFalse(self.manager.is_arrived())

        # Overrides must apply braking
        override = self.manager.get_control_override()
        self.assertEqual(override["throttle"], 0.0)
        self.assertGreater(override["brake"], 0.7)
        self.assertFalse(override["hand_brake"])

    def test_consecutive_frames_required_for_arrived(self):
        # In arrival radius, speed low (0.2 km/h < 0.5 km/h)
        pos_inside = (100.0 - 1.5, 100.0, 0.0)

        # Frame 1: Low speed, frame_count becomes 1
        state, _ = self.manager.update(pos_inside, current_speed_kmh=0.2)
        self.assertEqual(state, NavigationState.STOPPING)
        self.assertFalse(self.manager.is_arrived())

        # Frame 2: Low speed, frame_count becomes 2
        state, _ = self.manager.update(pos_inside, current_speed_kmh=0.1)
        self.assertEqual(state, NavigationState.STOPPING)
        self.assertFalse(self.manager.is_arrived())

        # Frame 3: Low speed, reaches 3 frames -> transitions to ARRIVED
        state, _ = self.manager.update(pos_inside, current_speed_kmh=0.0)
        self.assertEqual(state, NavigationState.ARRIVED)
        self.assertTrue(self.manager.is_arrived())
        self.assertTrue(self.manager.is_stopped())

    def test_speed_blip_resets_confirmation_counter(self):
        pos_inside = (100.0 - 1.5, 100.0, 0.0)

        # Frame 1: Low speed (1 frame)
        self.manager.update(pos_inside, current_speed_kmh=0.2)
        # Frame 2: Sudden roll/movement above threshold (e.g. 1.2 km/h) -> counter resets
        self.manager.update(pos_inside, current_speed_kmh=1.2)
        self.assertEqual(self.manager.state, NavigationState.STOPPING)
        self.assertEqual(self.manager._stopped_frame_count, 0)

        # Frame 3: Low speed again -> counter is only 1, not arrived
        state, _ = self.manager.update(pos_inside, current_speed_kmh=0.1)
        self.assertEqual(state, NavigationState.STOPPING)
        self.assertFalse(self.manager.is_arrived())

    def test_arrived_state_locks_permanently(self):
        pos_inside = (100.0, 100.0, 0.0)
        # Drive through to ARRIVED
        for _ in range(3):
            self.manager.update(pos_inside, current_speed_kmh=0.0)

        self.assertEqual(self.manager.state, NavigationState.ARRIVED)

        # Check control lockout
        override = self.manager.get_control_override()
        self.assertEqual(override["throttle"], 0.0)
        self.assertEqual(override["brake"], 1.0)
        self.assertTrue(override["hand_brake"])

        # Even if position moves or high speed is reported, it cannot exit ARRIVED
        state, target_speed = self.manager.update(
            current_position=(0.0, 0.0, 0.0),  # far away
            current_speed_kmh=30.0,             # high speed
        )
        self.assertEqual(state, NavigationState.ARRIVED)
        self.assertEqual(target_speed, 0.0)
        override = self.manager.get_control_override()
        self.assertEqual(override["throttle"], 0.0)
        self.assertEqual(override["brake"], 1.0)
        self.assertTrue(override["hand_brake"])

    def test_reset_functionality(self):
        pos_inside = (100.0, 100.0, 0.0)
        for _ in range(3):
            self.manager.update(pos_inside, current_speed_kmh=0.0)
        self.assertTrue(self.manager.is_arrived())

        # Reset FSM
        self.manager.reset()
        self.assertEqual(self.manager.state, NavigationState.NAVIGATING)
        self.assertFalse(self.manager.is_arrived())


if __name__ == "__main__":
    unittest.main()

