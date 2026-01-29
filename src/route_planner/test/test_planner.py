"""Tests for the Mission class."""

from datetime import datetime
import numpy as np
import random
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from route_planner.maneuver_action import JibeManeuver
from route_planner.maneuver_action import TackManeuver
from route_planner.planner import Mission
from route_planner.segment_action import StraightWayAction
from route_planner.segment_action import TackingWayAction
from route_planner.utils import bearing
from route_planner.waypoint import WayPoint


class TestMission(unittest.TestCase):
    """Tests for the Mission class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create standard waypoints for a simple square mission
        self.waypoints = [
            WayPoint.from_coordinates(east=1.0, north=1.0),
            WayPoint.from_coordinates(east=1.0, north=5.0),
            WayPoint.from_coordinates(east=5.0, north=5.0),
            WayPoint.from_coordinates(east=5.0, north=1.0),
            WayPoint.from_coordinates(east=1.0, north=1.0),  # Back to start
        ]

        # Location and environment for testing
        self.start_location = WayPoint(0.0, 0.0)
        self.current_location = WayPoint(0.8, 0.8)
        self.wind_direction=0.0  # North
        self.wind_speed=5.0  # m/s

        # Create mission object
        self.mission = Mission(
            waypoints=self.waypoints,
            dist_threshold=0.5,
            off_course_threshold=3.0,
            min_tack_angle=44.0,
            safe_jibe_threshold=25.0,
        )

    def test_initialization(self):
        """Test that the mission is initialized correctly."""
        self.assertEqual(self.mission.hard_waypoints, self.waypoints)
        self.assertEqual(self.mission.dist_threshold, 0.5)
        self.assertEqual(self.mission.off_course_threshold, 3.0)
        self.assertEqual(self.mission.min_tack_angle, 44.0)
        self.assertEqual(self.mission.safe_jibe_threshold, 25.0)
        self.assertFalse(self.mission.completed)
        self.assertIsNone(self.mission.maneuver_action)

    def test_plan(self):
        """Test planning the mission."""
        act1 = StraightWayAction(WayPoint.from_coordinates(0.0, 0.0), WayPoint.from_coordinates(1.0, 1.0))
        act2 = StraightWayAction(WayPoint.from_coordinates(1.0, 1.0), WayPoint.from_coordinates(1.0, 5.0))
        location = WayPoint(0.0, 0.1)

        # Mock the _build_submissions method to avoid actual submission creation
        with patch.object(self.mission, "_build_submissions") as mock_build:
            # Mock the return value of the method
            mock_build.return_value = [act1, act2]

            # Plan the mission
            route = self.mission.plan(location, self.wind_direction)
            self.assertEqual(self.mission.previous_waypoint, location)
            self.assertEqual(len(route), 2)

            true_route = [act1.target, act2.target]
            for i in range(len(route)):
                true = true_route[i]
                pred = route[i]
                self.assertAlmostEqual(true.east, pred.east, 5)
                self.assertAlmostEqual(true.north, pred.north, 5)

    def test_step_completed_and_wind_value_added(self):
        """Test step execution when the mission is already completed."""
        # Set the mission as completed
        self.mission.completed = True

        # Execute a step - should return zero heading change
        new_heading = self.mission.step(self.current_location, 30.0, self.wind_direction, self.wind_speed)

        # Should return zero heading change
        self.assertEqual(new_heading, 0.0)

    def test_step_with_active_maneuver(self):
        """Test step execution with an active maneuver."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        # Create a mock maneuver
        mock_maneuver = Mock()
        mock_maneuver.complete = False
        mock_maneuver.is_stuck.return_value = False
        mock_maneuver.step.return_value = 10.0

        # Set the maneuver
        self.mission.maneuver_action = mock_maneuver

        # Execute a step - should continue the maneuver
        heading_change = self.mission.step(self.current_location, 30.0, self.wind_direction, self.wind_speed)

        # Should return the heading change from the maneuver
        self.assertEqual(heading_change, 10.0)

        # Mock maneuver step should have been called
        mock_maneuver.step.assert_called_once_with(30.0)

    def test_step_maneuver_complete(self):
        """Test step execution when a maneuver completes."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        # Create a mock maneuver that is complete
        mock_maneuver = Mock()
        mock_maneuver.complete = True

        # Set the maneuver
        self.mission.maneuver_action = mock_maneuver

        # Mock other functions to avoid re-planning
        self.mission._off_course = Mock(return_value=False)  # type: ignore
        self.mission._critical_wind_change = Mock(return_value=False)  # type: ignore

        # Mock bearing to return a fixed heading
        with patch("route_planner.planner.bearing", return_value=45.0):
            # Execute a step - should complete the maneuver and continue navigation
            heading_change = self.mission.step(self.current_location, 30.0, self.wind_direction, self.wind_speed)

            # Should return the heading from bearing
            self.assertEqual(heading_change, 45.0)

            # Maneuver should be cleared
            self.assertIsNone(self.mission.maneuver_action)

    def test_step_in_maneuver_not_stuck(self):
        """Test step execution when in a maneuver that is not stuck."""
        # Create a mock maneuver
        self.mission.plan(self.start_location, self.wind_direction)
        mock_maneuver = Mock()
        mock_maneuver.complete = False
        mock_maneuver.is_stuck.return_value = False
        mock_maneuver.step.return_value = 10.0

        # Set the maneuver
        self.mission.maneuver_action = mock_maneuver

        # Execute a step - should return the heading change from the maneuver
        new_heading = self.mission.step(self.current_location, 30.0, self.wind_direction, self.wind_speed)

        # Should return the heading change from the maneuver
        self.assertEqual(new_heading, 10.0)

        # Mock maneuver step should have been called
        mock_maneuver.step.assert_called_once_with(30.0)

    def test_step_in_maneuver_stuck(self):
        """Test step execution when in a maneuver that is stuck."""
        self.mission.plan(self.start_location, self.wind_direction)

        # Create a mock maneuver that is stuck
        mock_maneuver = Mock()
        mock_maneuver.complete = False
        mock_maneuver.is_stuck.return_value = True
        mock_maneuver._target_heading = 0.0  # Old target (will be ignored in new logic)

        wind_direction=180.0
        wind_speed=5.0

        # Set the maneuver
        self.mission.maneuver_action = mock_maneuver

        # Calculate expected target heading (bearing from current location to current waypoint)
        expected_target_heading = bearing(self.current_location, self.mission.current_waypoint)

        # Execute a step - should create new jibe maneuver
        new_heading = self.mission.step(self.current_location, 30.0, wind_direction, wind_speed)

        # Should have switched to jibe maneuver
        self.assertIsInstance(self.mission.maneuver_action, JibeManeuver)

        # New behavior: target heading should be current desired heading, not old mock value
        self.assertAlmostEqual(
            self.mission.maneuver_action._target_heading, expected_target_heading, places=1
        )

        # Should return a valid heading
        self.assertIsNotNone(new_heading)

    def test_step_replan_on_off_course(self):
        """Test replanning when robot goes off course."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)
            original_waypoints = list(self.mission.route)  # Copy

        # Mock off_course to return True
        self.mission._off_course = Mock(return_value=True)  # type: ignore

        # Execute a step - should replan
        with patch("route_planner.planner.bearing", return_value=45.0):
            heading_change = self.mission.step(self.current_location, 30.0, self.wind_direction, self.wind_speed)

            # Should return the heading from bearing
            self.assertEqual(heading_change, 45.0)

            # The route should have been replanned (different from original)
            self.assertNotEqual(id(self.mission.route), id(original_waypoints))

    def test_step_replan_on_critical_wind_change(self):
        """Test replanning when wind changes critically."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)
            original_waypoints = list(self.mission.route)  # Copy

        # Mock critical_wind_change to return True
        self.mission._critical_wind_change = Mock(return_value=True)  # type: ignore
        self.mission._off_course = Mock(return_value=False)  # type: ignore

        # Execute a step - should replan
        with patch("route_planner.planner.bearing", return_value=45.0):
            heading_change = self.mission.step(self.current_location, 30.0, self.wind_direction, self.wind_speed)
            # Should return the heading from bearing
            self.assertEqual(heading_change, 45.0)

            # The route should have been replanned (different from original)
            self.assertNotEqual(id(self.mission.route), id(original_waypoints))

    def test_step_normal_navigation(self):
        """Test step execution during normal navigation."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        # Mock bearing to return a fixed heading
        with patch("route_planner.planner.bearing", return_value=45.0):
            # Also need to mock _off_course and _critical_wind_change to return False
            self.mission._off_course = Mock(return_value=False)  # type: ignore
            self.mission._critical_wind_change = Mock(return_value=False)  # type: ignore

            # Execute a step - should return a heading
            heading_change = self.mission.step(self.current_location, 30.0, self.wind_direction, self.wind_speed)
            # Should return the heading from bearing
            self.assertEqual(heading_change, 45.0)

            # Should not have completed the mission
            self.assertFalse(self.mission.completed)

            # Should not have started a maneuver
            self.assertIsNone(self.mission.maneuver_action)

    def test_step_reach_waypoint(self):
        """Test step execution when reaching a waypoint with new heading-based maneuver logic."""
        # Create test waypoints
        first_waypoint = WayPoint.from_coordinates(east=1.0, north=1.0)
        second_waypoint = WayPoint.from_coordinates(east=3.0, north=1.5)  # Eastward - easily sailable

        # Set BOTH route and hard_waypoints for consistent test
        self.mission.hard_waypoints = [first_waypoint, second_waypoint]

        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)
            # Now route should contain the test waypoints

        # Create a location very close to the first waypoint
        close_location = WayPoint(0.9, 0.9)
        current_heading = 30.0

        # Mock off_course and critical_wind_change to avoid re-planning during waypoint-reached logic
        self.mission._off_course = Mock(return_value=False)  # type: ignore
        self.mission._critical_wind_change = Mock(return_value=False)  # type: ignore

        # Calculate expected bearing: (0.9, 0.9) → (3.0, 1.5) ≈ 73° (eastward, sailable)
        expected_bearing_to_second = bearing(close_location, second_waypoint)

        # Execute the step
        heading_change = self.mission.step(close_location, current_heading, self.wind_direction, self.wind_speed)

        # Should have advanced to second waypoint (after reaching first)
        self.assertEqual(self.mission.current_waypoint.east, second_waypoint.east)
        self.assertEqual(self.mission.current_waypoint.north, second_waypoint.north)

        # Should NOT have started a maneuver (heading difference should be < 70°)
        self.assertIsNone(self.mission.maneuver_action)

        # Should return direct bearing to second waypoint
        self.assertAlmostEqual(heading_change, expected_bearing_to_second, delta=1.0)

    def test_step_reach_waypoint_needs_maneuver(self):
        """Test step execution when reaching waypoint requires a large heading change."""
        # Create test waypoints
        first_waypoint = WayPoint.from_coordinates(east=1.0, north=1.0)
        second_waypoint = WayPoint.from_coordinates(east=-1.0, north=1.0)

        # Set BOTH route and hard_waypoints
        self.mission.hard_waypoints = [first_waypoint, second_waypoint]

        # Setup scenario where next waypoint requires >70° turn but is still sailable
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        close_location = WayPoint(0.9, 0.9)
        current_heading = 30.0  # Heading northeast

        # Mock to avoid replanning during test
        self.mission._off_course = Mock(return_value=False)  # type: ignore
        self.mission._critical_wind_change = Mock(return_value=False)  # type: ignore

        # Calculate expected bearing: (0.9, 0.9) → (0.0, 3.0) ≈ 341° (northwest)
        # Heading difference: |341° - 30°| ≈ 91° > 70° → Should trigger maneuver
        expected_bearing = bearing(close_location, second_waypoint)
        heading_diff = abs((expected_bearing - current_heading + 180) % 360 - 180)

        print(f"Expected bearing: {expected_bearing:.1f}, Heading diff: {heading_diff:.1f}")

        if heading_diff > 70.0:
            # Mock maneuver
            mock_maneuver = Mock()
            mock_maneuver.step.return_value = 15.0

            with patch("route_planner.planner.Mission._get_maneuver_action", return_value=mock_maneuver):
                heading_change = self.mission.step(close_location, current_heading, self.wind_direction, self.wind_speed)

                # Should have advanced to second waypoint
                self.assertEqual(self.mission.current_waypoint.east, second_waypoint.east)
                self.assertEqual(self.mission.current_waypoint.north, second_waypoint.north)

                # Should have started a maneuver (large heading difference)
                self.assertIsNotNone(self.mission.maneuver_action)

                # Should return maneuver's heading command
                self.assertEqual(heading_change, 15.0)
        else:
            # If heading difference isn't large enough, adjust the test
            self.fail(f"Test setup error: heading difference {heading_diff:.1f}° < 70°")

    def test_step_reach_waypoint_eastward(self):
        """Test with eastward waypoint that's definitely sailable."""
        # Create test waypoints
        first_waypoint = WayPoint.from_coordinates(east=1.0, north=1.0)
        second_waypoint = WayPoint.from_coordinates(east=2.0, north=1.0)  # Due east

        # Set BOTH route and hard_waypoints
        self.mission.hard_waypoints = [first_waypoint, second_waypoint]

        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        self.mission._off_course = Mock(return_value=False)  # type: ignore
        self.mission._critical_wind_change = Mock(return_value=False)  # type: ignore

        # Should advance to second waypoint without maneuver
        self.mission.step(first_waypoint, 45.0, self.wind_direction, self.wind_speed)
        self.assertEqual(self.mission.current_waypoint.east, second_waypoint.east)
        self.assertEqual(self.mission.current_waypoint.north, second_waypoint.north)
        self.assertIsNone(self.mission.maneuver_action)

    def test_step_complete_mission(self):
        """Test step execution when completing the mission."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        # Manually set up the mission to have only one waypoint left
        self.mission.route = [self.waypoints[-1]]

        # Create a location very close to the last waypoint
        close_location = WayPoint(self.waypoints[-1].east - 0.1, self.waypoints[-1].north - 0.1)

        # Execute a step - should complete the mission
        heading_change = self.mission.step(close_location, 30.0, self.wind_direction, self.wind_speed)

        # Should have completed the mission
        self.assertTrue(self.mission.completed)

        # Should return zero heading change
        self.assertEqual(heading_change, 0.0)

    def test_next_waypoint(self):
        """Test the `next_waypoint` property."""
        # Should be the first waypoint initially
        self.assertIsNone(self.mission.next_waypoint)
        self.mission.route = [Mock(spec=WayPoint)]
        self.assertIsNone(self.mission.next_waypoint)
        next_waypoint = WayPoint.from_coordinates(east=1.0, north=2.0)
        self.mission.route.append(next_waypoint)
        self.assertEqual(self.mission.next_waypoint, next_waypoint)

    def test_current_submission(self):
        """Test the `current_submission` property."""
        # Should be None initially
        self.mission.way_actions = []
        self.assertIsNone(self.mission.current_submission)

        # Add a waypoint action
        self.mission.way_actions = [Mock(spec=StraightWayAction), Mock(spec=TackingWayAction)]
        self.assertIsInstance(self.mission.current_submission, StraightWayAction)

    def test_move_to_next_waypoint(self):
        """Test moving to the next waypoint."""
        # Set up route with multiple waypoints
        waypoint1 = Mock(spec=WayPoint)
        waypoint2 = Mock(spec=WayPoint)
        self.mission.route = [waypoint1, waypoint2]

        # Move to next waypoint
        next_waypoint = self.mission.move_to_next_waypoint()

        # Should have removed the first waypoint
        self.assertEqual(len(self.mission.route), 1)

        # Should have returned the second waypoint
        self.assertEqual(next_waypoint, waypoint2)

        # Current waypoint should now be the second waypoint
        self.assertEqual(self.mission.current_waypoint, waypoint2)

    def test_build_submissions(self):
        """Test building the list of submissions."""
        # Set up route with multiple waypoints
        wayactions = self.mission._build_submissions(self.start_location, self.wind_direction)
        self.assertEqual(len(wayactions), 5)

        start_waypoint = WayPoint.from_coordinates(self.start_location.east, self.start_location.north, True)
        ground_truth = [
            (TackingWayAction, start_waypoint, self.waypoints[0]),
            (TackingWayAction, self.waypoints[0], self.waypoints[1]),
            (StraightWayAction, self.waypoints[1], self.waypoints[2]),
            (StraightWayAction, self.waypoints[2], self.waypoints[3]),
            (StraightWayAction, self.waypoints[3], self.waypoints[4]),
        ]

        for created_action, (cls, start, target) in zip(wayactions, ground_truth):
            self.assertIsInstance(created_action, cls)
            self.assertAlmostEqual(created_action.start.east, start.east, 5)
            self.assertAlmostEqual(created_action.start.north, start.north, 5)
            self.assertAlmostEqual(created_action.target.east, target.east, 5)
            self.assertAlmostEqual(created_action.target.north, target.north, 5)

    def test_get_way_action(self):
        """Test selecting the appropriate way action based on wind conditions."""
        # Test with favorable wind conditions
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            action = self.mission._get_way_action(self.waypoints[0], self.waypoints[1], self.wind_direction)
            self.assertIsInstance(action, StraightWayAction)

        # Test with unfavorable wind conditions
        with patch("route_planner.planner.is_sailable_direction", return_value=False):
            action = self.mission._get_way_action(self.waypoints[0], self.waypoints[1], self.wind_direction)
            self.assertIsInstance(action, TackingWayAction)

        # Test with variable wind - we need a different approach
        # The logic works by adjusting min_tack_angle and then calling is_sailable_direction
        # Since the wind variance check happens inside the method,
        # we need to prepare a test that isolates just this behavior

        # Mock is_sailable_direction to return False to simulate the effect of
        # increased min_tack_angle making the path no longer sailable
        with patch("route_planner.planner.is_sailable_direction", return_value=False):
            action = self.mission._get_way_action(self.waypoints[0], self.waypoints[1], self.wind_direction)
            self.assertIsInstance(action, TackingWayAction)

    def test_get_maneuver_action_tack(self):
        """Test selecting tack maneuver."""
        # Set up conditions where tacking is preferable
        current_heading = 0.0  # North
        target_heading = 45.0  # Northeast
        wind_direction = 45.0  # Wind from Northeast (close to the wind)

        # Mock bearing to return target heading
        with patch("route_planner.planner.bearing", return_value=target_heading):
            maneuver = self.mission._get_maneuver_action(
                self.current_location, self.waypoints[0], current_heading, wind_direction, wind_speed=10.0
            )

            # Should select a tack maneuver
            self.assertIsInstance(maneuver, TackManeuver)

    def test_get_maneuver_action_jibe(self):
        """Test selecting jibe maneuver."""
        # Set up conditions where jibing is preferable - specifically for this implementation:
        # 1. Both wind angles need to be > 120 degrees (running with the wind)
        # 2. Wind speed must be below safe threshold
        current_heading = 180.0  # South
        target_heading = 230.0  # Southwest
        wind_direction = 0.0  # Wind from North (directly behind)

        # Mock bearing to return target heading
        with patch("route_planner.planner.bearing", return_value=target_heading):
            # Now test the method
            maneuver = self.mission._get_maneuver_action(
                self.current_location, self.waypoints[0], current_heading, wind_direction, wind_speed=10.0
            )

            # Should select a jibe maneuver in the running condition when wind is low
            self.assertIsInstance(maneuver, JibeManeuver)
            self.assertEqual(maneuver._target_heading, target_heading)
            self.assertEqual(maneuver._current_heading, current_heading)
            self.assertEqual(maneuver._wind_direction, wind_direction)

    def test_get_tack_when_windspeed_high(self):
        """Test selecting tack maneuver instead of jibe in high wind conditions."""
        # Set up conditions where jibing would normally be preferable
        current_heading = 180.0
        target_heading = 230.0
        wind_direction = 0.0

        # Mock bearing to return target heading
        with patch("route_planner.planner.bearing", return_value=target_heading):
            self.mission.safe_jibe_threshold = 15.0  # Lower threshold for test
            maneuver = self.mission._get_maneuver_action(
                self.current_location, self.waypoints[0], current_heading, wind_direction, wind_speed=30.0
            )

            # Should select a tack maneuver despite the running conditions
            self.assertIsInstance(maneuver, TackManeuver)
            self.assertEqual(maneuver._target_heading, target_heading)
            self.assertEqual(maneuver._current_heading, current_heading)
            self.assertEqual(maneuver._wind_direction, wind_direction)

    def test_get_maneuver_action_high_wind(self):
        """Test selecting tack instead of jibe in high wind conditions."""
        # Set up conditions where jibing would normally be preferable
        current_heading = 180.0  # South
        target_heading = 270.0  # West
        wind_direction = 0.0  # Wind from North (running with the wind)

        # Mock bearing to return target heading
        with patch("route_planner.planner.bearing", return_value=target_heading):
            maneuver = self.mission._get_maneuver_action(
                self.current_location, self.waypoints[0], current_heading, wind_direction, wind_speed=10.0
            )

            # Should select a tack maneuver despite the running conditions
            self.assertIsInstance(maneuver, TackManeuver)

    def test_off_course_detection(self):
        """Test detecting when the robot is off course."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        # Set the previous waypoint and current waypoint to create a line
        self.mission._previous_waypoint = WayPoint.from_coordinates(east=0.0, north=0.0)
        self.mission.route = [WayPoint.from_coordinates(east=10.0, north=0.0)]  # Horizontal line

        # Test with a point that is on course
        on_course_location = WayPoint(5.0, 0.5)  # Close to the line
        self.assertFalse(self.mission._off_course(on_course_location))

        # Test with a point that is off course
        off_course_location = WayPoint(5.0, 4.0)  # Far from the line
        self.assertTrue(self.mission._off_course(off_course_location))

    def test_critical_wind_change(self):
        """Test detecting critical wind changes."""
        # First plan the mission
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.mission.plan(self.start_location, self.wind_direction)

        # Test with favorable wind conditions
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            self.assertFalse(self.mission._critical_wind_change(self.current_location, self.wind_direction))

        # Test with unfavorable wind conditions
        with patch("route_planner.planner.is_sailable_direction", return_value=False):
            self.assertTrue(self.mission._critical_wind_change(self.current_location, self.wind_direction))
        

    def test_plan_with_favorable_wind(self):
        """Test planning with favorable wind conditions."""
        # Mock is_sailable_direction to always return True for favorable wind
        with patch("route_planner.planner.is_sailable_direction", return_value=True):
            # Plan the mission
            route = self.mission.plan(self.start_location, self.wind_direction)

            # Check that all way actions are StraightWayAction
            for action in self.mission.way_actions:
                self.assertIsInstance(action, StraightWayAction)

            # Check that the number of waypoints is at least the number of hard waypoints
            # The actual number may vary based on implementation details
            self.assertGreaterEqual(len(self.mission.route), len(self.waypoints))

            # Check that the route starts with the first waypoint
            self.assertEqual(self.mission.route[0].east, self.waypoints[0].east)
            self.assertEqual(self.mission.route[0].north, self.waypoints[0].north)

            # Check that the route ends at the last waypoint
            self.assertEqual(self.mission.route[-1], self.waypoints[-1])

            # Check that the timed route is not empty
            self.assertTrue(len(route) > 0)


# class TestMissionSimulation(unittest.TestCase):
#     """Test the Mission class with simulated waypoints and wind conditions."""

#     def setUp(self):
#         # Initialize WindMonitor with the same parameters from your node
#         self.wind_monitor = WindMonitor(window_size=5, min_measurements=3, measurement_max_age=20)

#         # Add initial wind measurement to ensure the monitor has data
#         self.wind_speed=5.0,
#         self.wind_direction=100.0

#         # Define origin and locations
#         self.origin = PolarLocation(latitude=49.877481, longitude=8.651811)
#         self.first_pos = self.origin  # Start at origin
#         self.wp1 = PolarLocation(latitude=49.877338, longitude=8.651025)
#         self.wp2 = PolarLocation(latitude=49.877804, longitude=8.650953)

#         # Convert waypoints to WayPoint objects
#         wp1_cartesian = self.wp1.to_cartesian(self.origin)
#         wp2_cartesian = self.wp2.to_cartesian(self.origin)
#         self.wp1_waypoint = WayPoint.from_coordinates(wp1_cartesian.east, wp1_cartesian.north)
#         self.wp2_waypoint = WayPoint.from_coordinates(wp2_cartesian.east, wp2_cartesian.north)

#         # Create Mission planner with the same parameters from your test
#         self.mission = Mission(
#             waypoints=[self.wp1_waypoint, self.wp2_waypoint],
#             origin=self.origin,
#             wind_monitor=self.wind_monitor,
#             dist_threshold=0.1,
#             off_course_threshold=5,
#             min_tack_angle=45,
#             safe_jibe_threshold=15,
#             maneuver_heading_step_size=10,
#             command_step_size_multiplier=1.5,
#             heading_tolerance=1,
#             max_steps_to_cross_wind_when_tacking=100,
#         )

#         np.random.seed(42)  # Set seed for reproducibility
#         random.seed(42)  # Set seed for reproducibility

#     def _to_cartesian_no_origin(self, polar_location: PolarLocation):
#         """Convert PolarLocation to CartesianLocation without without the new location having a set origin"""
#         temp = polar_location.to_cartesian(self.origin)
#         return CartesianLocation(
#             east=temp.east,
#             north=temp.north,
#         )

#     def test_simulated_route(self):
#         """Test the planner directly without ROS2 messaging"""
#         # Initial position in Cartesian coordinates
#         first_pos_cartesian = self._to_cartesian_no_origin(self.first_pos)

#         # Plan the route
#         self.mission.plan(first_pos_cartesian)
#         self.mission.step(first_pos_cartesian, 250.0)  # Initialize the mission

#         # Generate points to waypoint 1
#         points_to_wp1 = self._interpolate(
#             (self.first_pos.latitude, self.first_pos.longitude), (self.wp1.latitude, self.wp1.longitude), 20
#         )

#         # Simulate movement toward first waypoint
#         print("\nSimulating movement to waypoint 1...")
#         current_heading = self.first_pos.bearing(self.wp1) + random.uniform(-5, 5)

#         for point in points_to_wp1[1:-1]:
#             # Convert point to Cartesian for the planner
#             cartesian_point = self._to_cartesian_no_origin(point)

#             # Get desired heading from planner
#             desired_heading = self.mission.step(cartesian_point, current_heading)
#             current_heading = desired_heading + random.uniform(-5, 5)
#             expected_heading = point.bearing(self.wp1)

#             # Verify heading is close to expected
#             self.assertAlmostEqual(desired_heading, expected_heading, delta=5)

#         # Reach first waypoint
#         wp1_cartesian = self._to_cartesian_no_origin(self.wp1)
#         desired_heading = self.mission.step(wp1_cartesian, self.wp1.bearing(self.wp2) + random.uniform(-5, 5))
#         current_heading = desired_heading

#         # Generate points to waypoint 2
#         points_to_wp2 = self._interpolate(
#             (self.wp1.latitude, self.wp1.longitude), (self.wp2.latitude, self.wp2.longitude), 20
#         )

#         # Simulate movement toward second waypoint
#         print("\nSimulating movement to waypoint 2...")
#         for point in points_to_wp2[:-1]:
#             cartesian_point = self._to_cartesian_no_origin(point)
#             desired_heading = self.mission.step(cartesian_point, current_heading)
#             current_heading = desired_heading
#             expected_heading = point.bearing(self.wp2) + random.uniform(-5, 5)

#             self.assertAlmostEqual(desired_heading, expected_heading, delta=5)

#         # Reach second waypoint
#         wp2_cartesian = self._to_cartesian_no_origin(self.wp2)
#         final_heading = self.mission.step(wp2_cartesian, current_heading)
#         print(f"Reached waypoint 2, final heading: {final_heading:.2f}")

#         # Verify mission completion
#         self.assertTrue(self.mission.completed, "Mission was not marked as completed")
#         self.assertEqual(final_heading, 0.0, "Final heading should be 0.0 when mission completed")

#     def _interpolate(self, point1, point2, n_points, noise_level=0.000005):
#         """Linearly interpolate n points between two geographical coordinates and add noise.

#         Parameters:
#         point1 (tuple): Starting point in (lat, lon) format
#         point2 (tuple): Ending point in (lat, lon) format
#         n_points (int): Number of points to interpolate (including start and end points)
#         noise_level (float): Standard deviation of the Gaussian noise to add

#         Returns:
#         list: List of interpolated PolarLocation objects
#         """
#         # Extract coordinates
#         lat1, lon1 = point1
#         lat2, lon2 = point2

#         # Create n evenly spaced values from 0 to 1 (including 0 and 1)
#         t = np.linspace(0, 1, n_points)  # pylint: disable=C0103

#         # Linear interpolation formula: p = p1 + t * (p2 - p1)
#         lats = lat1 + t * (lat2 - lat1)
#         lons = lon1 + t * (lon2 - lon1)

#         # Add Gaussian noise
#         lat_noise = np.random.normal(0, noise_level, n_points)
#         lon_noise = np.random.normal(0, noise_level, n_points)

#         lats += lat_noise
#         lons += lon_noise

#         # Combine lat and lon into PolarLocation objects
#         return [PolarLocation(latitude=lat, longitude=lon) for lat, lon in zip(lats, lons)]
