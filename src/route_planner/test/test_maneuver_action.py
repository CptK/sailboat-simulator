"""Unit tests for the ManeuverAction class and its subclasses"""

import unittest

from route_planner.maneuver_action import JibeManeuver
from route_planner.maneuver_action import ManeuverAction
from route_planner.maneuver_action import TackManeuver


class ManeuverForTesting(ManeuverAction):  # this is line 5
    """Concrete implementation for testing"""

    def is_stuck(self, current_heading: float) -> bool:
        return False

    def _turn_clockwise(self) -> bool:
        return True  # Always turn clockwise for testing


class TestManeuverAction(unittest.TestCase):
    """Unit tests for the ManeuverAction class"""

    def setUp(self):
        # Common test parameters
        self.target_heading = 90.0
        self.current_heading = 0.0
        self.heading_step_size = 10.0
        self.wind_direction = 45.0
        self.command_multiplier = 1.5
        self.heading_tolerance = 2.0

        self.maneuver = ManeuverForTesting(
            target_heading=self.target_heading,
            current_heading=self.current_heading,
            heading_step_size=self.heading_step_size,
            wind_direction=self.wind_direction,
            command_step_size_multiplier=self.command_multiplier,
            heading_tolerance=self.heading_tolerance,
        )

    def test_properties(self):
        self.assertEqual(self.maneuver.target_heading, self.target_heading)
        self.assertEqual(self.maneuver.current_heading, self.current_heading)
        self.assertEqual(self.maneuver.heading_step_size, self.heading_step_size)
        self.assertEqual(self.maneuver.command_step_size_multiplier, self.command_multiplier)
        self.assertEqual(self.maneuver.heading_tolerance, self.heading_tolerance)

        self.maneuver._current_command = 10.0
        self.assertEqual(self.maneuver.current_command, 10.0)
        self.maneuver._last_step_heading = 20.0
        self.assertEqual(self.maneuver.last_step_heading, 20.0)
        self.maneuver._current_step_heading = 30.0
        self.assertEqual(self.maneuver.current_step_heading, 30.0)
        self.maneuver._clockwise = True
        self.assertTrue(self.maneuver.clockwise)

    def test_has_reached_current_step_clockwise(self):
        # Should not have reached step yet
        self.maneuver._current_heading = 5.0
        self.assertFalse(self.maneuver._has_reached_current_step())

        # Should have reached step (exactly)
        self.maneuver._current_heading = 10.0
        self.assertTrue(self.maneuver._has_reached_current_step())

        # Should have reached step (within tolerance)
        self.maneuver._current_heading = 8.5  # 10 - 1.5 tolerance
        self.assertTrue(self.maneuver._has_reached_current_step())

    def test_has_reached_current_step_counterclockwise(self):
        # Setup counterclockwise turn
        self.maneuver._clockwise = False
        self.maneuver._last_step_heading = 360.0

        # Should not have reached step yet
        self.maneuver._current_heading = 355.0
        self.assertFalse(self.maneuver._has_reached_current_step())

        # Should have reached step (exactly)
        self.maneuver._current_heading = 350.0
        self.assertTrue(self.maneuver._has_reached_current_step())

    def test_would_overshoot_target(self):
        # Test clockwise overshoot
        self.assertTrue(self.maneuver._would_overshoot_target(95.0))  # Past target
        self.assertFalse(self.maneuver._would_overshoot_target(85.0))  # Before target

        # Test counterclockwise overshoot
        self.maneuver._clockwise = False
        self.assertTrue(self.maneuver._would_overshoot_target(85.0))  # Past target
        self.assertFalse(self.maneuver._would_overshoot_target(95.0))  # Before target

    def test_amplify_command(self):
        # Test clockwise amplification
        base_heading = 10.0
        expected = (base_heading + self.heading_step_size * (self.command_multiplier - 1)) % 360
        self.assertEqual(self.maneuver._amplify_command(base_heading), expected)

        # Test counterclockwise amplification
        self.maneuver._clockwise = False
        expected = (base_heading - self.heading_step_size * (self.command_multiplier - 1)) % 360
        self.assertEqual(self.maneuver._amplify_command(base_heading), expected)

    def test_calculate_next_command(self):
        # Test reaching target
        self.maneuver._current_step_heading = 85.0
        self.assertEqual(self.maneuver._calculate_next_command(), self.target_heading)

        # Test normal progression
        self.maneuver._current_step_heading = 0.0
        self.maneuver._target_heading = 180.0
        expected = self.maneuver._amplify_command(10.0)  # One step size forward
        self.assertEqual(self.maneuver._calculate_next_command(), expected)
        # Verify last_step_heading was updated
        self.assertEqual(self.maneuver._current_step_heading, 10.0)

    def test_calculate_next_heading_step_integration(self):
        # Test not reached step yet
        self.maneuver._current_heading = 5.0
        cmd1 = self.maneuver._calculate_next_command()
        self.assertEqual(cmd1, self.maneuver._get_current_command())

        # Test reached step, calculate next
        self.maneuver._current_heading = 12.0  # Past first step
        cmd2 = self.maneuver._calculate_next_command()
        self.assertNotEqual(cmd1, cmd2)  # Should get different command

    def test_step(self):
        # Test normal step
        next_heading = self.maneuver.step(5.0)
        self.assertIsNotNone(next_heading)
        self.assertEqual(self.maneuver._current_heading, 5.0)

        # Test completion
        self.maneuver._complete = True
        self.assertIsNone(self.maneuver.step(10.0))

    def test_complete_property(self):
        self.assertFalse(self.maneuver.complete)
        self.maneuver._complete = True
        self.assertTrue(self.maneuver.complete)

    def test_get_current_command(self):
        expected = self.maneuver._amplify_command(self.maneuver._current_step_heading)
        self.assertEqual(self.maneuver._get_current_command(), expected)

        self.maneuver._current_step_heading = self.maneuver._target_heading
        new_command = self.maneuver._get_current_command()
        self.assertEqual(new_command, self.maneuver._target_heading)


class TestTackManeuver(unittest.TestCase):
    """Unit tests for the TackManeuver class"""

    def setUp(self):
        # Common test parameters
        self.target_heading = 240.0
        self.current_heading = 90.0
        self.heading_step_size = 10.0
        self.wind_direction = 180.0  # Wind from the south
        self.command_multiplier = 1.5
        self.heading_tolerance = 2.0
        self.max_steps = 40

        self.tack = TackManeuver(
            target_heading=self.target_heading,
            current_heading=self.current_heading,
            heading_step_size=self.heading_step_size,
            wind_direction=self.wind_direction,
            command_step_size_multiplier=self.command_multiplier,
            heading_tolerance=self.heading_tolerance,
            max_steps_to_cross_wind=self.max_steps,
        )

    def test_is_on_port_tack(self):
        # When heading = 270 (facing west) and wind from south (180)
        # Wind angle = (180 - 270) % 360 = 270, meaning wind from port
        self.assertTrue(self.tack._is_on_port_tack(270.0))  # Wind should be from port

        # When heading = 90 (facing east) and wind from south (180)
        # Wind angle = (180 - 90) % 360 = 90, meaning wind from starboard
        self.assertFalse(self.tack._is_on_port_tack(90.0))  # Wind should be from starboard

        # Edge cases
        # When heading directly south (180), wind from south (180)
        # Wind angle = (180 - 180) % 360 = 0, meaning wind from ahead/starboard
        self.assertFalse(self.tack._is_on_port_tack(180.0))

        # When heading directly north (0), wind from south (180)
        # Wind angle = (180 - 0) % 360 = 180, meaning wind from behind
        self.assertFalse(self.tack._is_on_port_tack(0.0))

    def test_turn_clockwise(self):
        # Test clockwise turn determination
        tack1 = TackManeuver(
            target_heading=270.0,  # Target far counterclockwise
            current_heading=0.0,
            heading_step_size=10.0,
            wind_direction=180.0,
            command_step_size_multiplier=1.5,
            heading_tolerance=2.0,
        )
        self.assertFalse(tack1._turn_clockwise())

        tack2 = TackManeuver(
            target_heading=0.0,  # Target far clockwise
            current_heading=270.0,
            heading_step_size=10.0,
            wind_direction=180.0,
            command_step_size_multiplier=1.5,
            heading_tolerance=2.0,
        )
        self.assertTrue(tack2._turn_clockwise())

    def test_step_counter(self):
        # Test step counter increment
        initial_steps = self.tack._steps_taken
        self.tack.step(95.0)
        self.assertEqual(self.tack._steps_taken, initial_steps + 1)

    def test_is_stuck(self):
        # Should not be stuck initially
        self.assertFalse(self.tack.is_stuck(self.current_heading))

        # Should be stuck after max steps without crossing wind
        self.tack._steps_taken = self.max_steps
        self.assertTrue(self.tack.is_stuck(self.current_heading))

        # Should not be stuck if crossed wind, even after max steps
        self.tack._steps_taken = self.max_steps
        self.assertFalse(self.tack.is_stuck(270.0))  # Crossed to starboard tack

    def test_tack_completion(self):
        # Test completion through a full maneuver sequence
        self.tack._current_heading = 230.0  # Almost at target
        self.tack._current_step_heading = 230.0
        self.tack._last_step_heading = 220.0

        # Step should give us the target heading since we're close
        next_heading = self.tack.step(235.0)  # Progress towards target
        self.assertEqual(next_heading, self.target_heading)

        next_heading = self.tack.step(244.0)
        self.assertEqual(next_heading, self.target_heading)
        self.assertTrue(self.tack.complete)

        # Further steps should return None
        self.assertIsNone(self.tack.step(self.target_heading))


class TestJibeManeuver(unittest.TestCase):
    """Unit tests for the JibeManeuver class"""

    def setUp(self):
        self.target_heading = 270.0
        self.current_heading = 90.0
        self.heading_step_size = 10.0
        self.wind_direction = 180.0  # Wind from the south
        self.command_multiplier = 1.5
        self.heading_tolerance = 2.0

        self.jibe = JibeManeuver(
            target_heading=self.target_heading,
            current_heading=self.current_heading,
            heading_step_size=self.heading_step_size,
            wind_direction=self.wind_direction,
            command_step_size_multiplier=self.command_multiplier,
            heading_tolerance=self.heading_tolerance,
        )

    def test_is_stuck(self):
        # Jibe maneuver should never be stuck
        self.assertFalse(self.jibe.is_stuck(self.current_heading))
        self.assertFalse(self.jibe.is_stuck(0.0))
        self.assertFalse(self.jibe.is_stuck(359.0))

    def test_turn_clockwise(self):
        # Test jibe direction based on wind angle
        # Should turn clockwise when wind is on port side
        jibe1 = JibeManeuver(
            target_heading=270.0,
            current_heading=90.0,  # Wind angle = 90 (on starboard)
            heading_step_size=10.0,
            wind_direction=180.0,
            command_step_size_multiplier=1.5,
            heading_tolerance=2.0,
        )
        self.assertFalse(jibe1._turn_clockwise())

        # Should turn counterclockwise when wind is on starboard side
        jibe2 = JibeManeuver(
            target_heading=90.0,
            current_heading=270.0,  # Wind angle = 270 (on port)
            heading_step_size=10.0,
            wind_direction=180.0,
            command_step_size_multiplier=1.5,
            heading_tolerance=2.0,
        )
        self.assertTrue(jibe2._turn_clockwise())

    def test_jibe_completion(self):
        # Test completion through a full maneuver sequence
        self.jibe._current_heading = 275.0  # Almost at target
        self.jibe._current_step_heading = 275.0
        self.jibe._last_step_heading = 290.0

        # Step should give us the target heading since we're close
        next_heading = self.jibe.step(275.0)  # Progress towards target
        self.assertEqual(next_heading, self.target_heading)

        next_heading = self.jibe.step(275.0)
        self.assertEqual(next_heading, self.target_heading)

        next_heading = self.jibe.step(268.0)
        self.assertEqual(next_heading, self.target_heading)
        self.assertTrue(self.jibe.complete)

        next_heading = self.jibe.step(270.0)
        self.assertIsNone(next_heading)

        # Further steps should return None
        self.assertIsNone(self.jibe.step(self.target_heading))

    def test_edge_case_wind_angles(self):
        # Case 1: Wind on starboard side (should turn clockwise to avoid wind)
        jibe1 = JibeManeuver(
            target_heading=180.0,  # Want to head South
            current_heading=0.0,  # Currently heading North
            heading_step_size=10.0,
            wind_direction=270.0,  # Wind from West (on starboard)
            command_step_size_multiplier=1.5,
            heading_tolerance=2.0,
        )
        # Wind on starboard => should turn clockwise (away from wind)
        self.assertTrue(jibe1._turn_clockwise())

        # Case 2: Wind on port side (should turn counterclockwise to avoid wind)
        jibe2 = JibeManeuver(
            target_heading=0.0,  # Want to head North
            current_heading=180.0,  # Currently heading South
            heading_step_size=10.0,
            wind_direction=270.0,  # Wind from East (on port)
            command_step_size_multiplier=1.5,
            heading_tolerance=2.0,
        )
        # Wind on port => should turn counterclockwise (away from wind)
        self.assertFalse(jibe2._turn_clockwise())

    def test(self):
        current_heading = 5.0
        jibe = JibeManeuver(
            target_heading=355.0,
            current_heading=current_heading,
            heading_step_size=10.0,
            wind_direction=0.0,
            command_step_size_multiplier=1.5,
            heading_tolerance=0.0,
        )

        # Test completion through a full maneuver sequence
        expected = 20
        while current_heading != 355:
            next_heading = jibe.step(current_heading)
            self.assertEqual(next_heading, expected)
            next_heading = jibe.step(current_heading + 1.0)
            self.assertEqual(next_heading, expected)
            current_heading = next_heading  # type: ignore
            expected = (
                expected + 10 if expected != 350 else (355 if expected != 355 else None)  # type: ignore
            )

        current_heading = 355.0
        jibe = JibeManeuver(
            target_heading=5.0,
            current_heading=current_heading,
            heading_step_size=10.0,
            wind_direction=0.0,
            command_step_size_multiplier=1.5,
            heading_tolerance=0.0,
        )
        expected = 340
        while current_heading != 5:
            next_heading = jibe.step(current_heading)
            self.assertEqual(next_heading, expected)
            next_heading = jibe.step(current_heading - 1.0)
            self.assertEqual(next_heading, expected)
            current_heading = next_heading  # type: ignore
            expected = expected - 10 if expected != 10 else (5 if expected != 5 else None)  # type: ignore
