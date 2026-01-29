"""Improved test cases for the utility functions in route_planner.utils."""

import unittest
from unittest.mock import Mock
from unittest.mock import patch

from route_planner.utils import angular_distance
from route_planner.utils import bearing
from route_planner.utils import is_sailable_direction
from route_planner.utils import point_to_line_distance
from route_planner.waypoint import WayPoint


class TestAngularDistance(unittest.TestCase):
    """Test cases specifically for the angular_distance function."""

    def test_angular_distance_simple_cases(self):
        """Test angular distance for simple cases."""
        # Same angles
        self.assertEqual(angular_distance(0, 0), 0)
        self.assertEqual(angular_distance(180, 180), 0)

        # 90 degree differences
        self.assertEqual(angular_distance(0, 90), 90)
        self.assertEqual(angular_distance(90, 0), 90)

        # 180 degree difference (maximum)
        self.assertEqual(angular_distance(0, 180), 180)
        self.assertEqual(angular_distance(180, 0), 180)

    def test_angular_distance_circular_boundary_cases(self):
        """Test angular distance across the 0°/360° boundary - these would fail with the old bug!"""
        # These are the critical test cases that would catch the bug
        self.assertAlmostEqual(angular_distance(10, 350), 20)
        self.assertAlmostEqual(angular_distance(350, 10), 20)

        self.assertAlmostEqual(angular_distance(5, 355), 10)
        self.assertAlmostEqual(angular_distance(355, 5), 10)

        self.assertAlmostEqual(angular_distance(1, 359), 2)
        self.assertAlmostEqual(angular_distance(359, 1), 2)

    def test_angular_distance_edge_cases(self):
        """Test angular distance edge cases."""
        # Just under 180 degrees
        self.assertAlmostEqual(angular_distance(0, 179), 179)
        self.assertAlmostEqual(angular_distance(0, 181), 179)  # Wraps around to shorter path

        # Various angles crossing boundary
        self.assertAlmostEqual(angular_distance(30, 330), 60)
        self.assertAlmostEqual(angular_distance(45, 315), 90)

    def test_angular_distance_symmetry(self):
        """Test that angular distance is symmetric."""
        test_cases = [(10, 350), (45, 315), (1, 359), (90, 270), (0, 180)]
        for angle1, angle2 in test_cases:
            self.assertEqual(angular_distance(angle1, angle2), angular_distance(angle2, angle1))


class TestBearing(unittest.TestCase):
    """Test cases for the bearing function."""

    def test_bearing_cardinal_directions(self):
        """Test bearing for cardinal directions."""
        wp_north = WayPoint(0, 10)
        wp_south = WayPoint(0, -10)
        wp_east = WayPoint(10, 0)
        wp_west = WayPoint(-10, 0)

        self.assertEqual(bearing(WayPoint(0, 0), wp_north), 0)
        self.assertEqual(bearing(WayPoint(0, 0), wp_east), 90)
        self.assertEqual(bearing(WayPoint(0, 0), wp_south), 180)
        self.assertEqual(bearing(WayPoint(0, 0), wp_west), 270)

    def test_bearing_intercardinal_directions(self):
        """Test bearing for intercardinal directions."""
        wp_ne = WayPoint(10, 10)
        wp_se = WayPoint(10, -10)
        wp_sw = WayPoint(-10, -10)
        wp_nw = WayPoint(-10, 10)

        self.assertEqual(bearing(WayPoint(0, 0), wp_ne), 45)
        self.assertEqual(bearing(WayPoint(0, 0), wp_se), 135)
        self.assertEqual(bearing(WayPoint(0, 0), wp_sw), 225)
        self.assertEqual(bearing(WayPoint(0, 0), wp_nw), 315)


class TestIsSailableDirection(unittest.TestCase):
    """Test cases for is_sailable_direction with real angle calculations."""

    def test_is_sailable_direction_circular_cases(self):
        """Test is_sailable_direction with circular angle cases."""
        with patch("route_planner.utils.bearing") as mock_bearing:
            start = Mock()
            end = Mock()

            # Test case 1: Wind at 350°, bearing at 10° → angular distance = 20° < 45° (not sailable)
            mock_bearing.return_value = 10.0
            wind_direction = 350.0
            result = is_sailable_direction(start, end, wind_direction, minimum_tack_angle=45)
            self.assertFalse(result)  # 20° < 45° minimum

            # Test case 2: Wind at 350°, bearing at 10° → angular distance = 20° < 25° (not sailable)
            result = is_sailable_direction(start, end, wind_direction, minimum_tack_angle=25)
            self.assertFalse(result)  # 20° < 25° minimum

            # Test case 3: Wind at 350°, bearing at 10° → angular distance = 20° >= 15° (sailable)
            result = is_sailable_direction(start, end, wind_direction, minimum_tack_angle=15)
            self.assertTrue(result)  # 20° >= 15° minimum

            # Test case 4: Wind at 10°, bearing at 350° → angular distance = 20° < 45° (not sailable)
            mock_bearing.return_value = 350.0
            wind_direction = 10.0
            result = is_sailable_direction(start, end, wind_direction, minimum_tack_angle=45)
            self.assertFalse(result)  # Same 20° difference

    def test_is_sailable_direction_real_sailing_scenarios(self):
        """Test realistic sailing scenarios."""
        with patch("route_planner.utils.bearing") as mock_bearing:
            start = Mock()
            end = Mock()

            # Scenario 1: Sailing close to wind (not sailable)
            mock_bearing.return_value = 45.0  # Bearing northeast
            wind_direction = 0.0  # Wind from north
            result = is_sailable_direction(start, end, wind_direction, minimum_tack_angle=45)
            self.assertEqual(result, True)  # 45° exactly at limit

            # Scenario 2: Sailing beam reach (sailable)
            mock_bearing.return_value = 90.0  # Bearing east
            wind_direction = 0.0  # Wind from north
            result = is_sailable_direction(start, end, wind_direction, minimum_tack_angle=45)
            self.assertTrue(result)  # 90° > 45° minimum

            # Scenario 3: Sailing downwind (sailable)
            mock_bearing.return_value = 180.0  # Bearing south
            wind_direction = 0.0  # Wind from north
            result = is_sailable_direction(start, end, wind_direction, minimum_tack_angle=45)
            self.assertTrue(result)  # 180° > 45° minimum

    def test_is_sailable_direction_integration_with_bearing(self):
        """Test is_sailable_direction without mocking bearing function."""
        # Create actual test points
        start_loc = WayPoint(0.0, 0.0)
        end_loc = WayPoint(0.0, 100.0)  # Due north

        # Test with wind from northeast (45°) - should not be sailable
        wind_direction = 45.0
        result = is_sailable_direction(
            start_loc, end_loc, wind_direction, minimum_tack_angle=45
        )
        self.assertEqual(result, True)  # 45° exactly at minimum


class TestGeometryFunctions(unittest.TestCase):
    """Test cases for the existing geometry functions with some improvements."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.waypoint1 = WayPoint(10.0, 20.0)
        self.waypoint2 = WayPoint(30.0, 40.0)

    def test_bearing(self):
        """Test the bearing function."""

        result1 = bearing(self.waypoint1, self.waypoint2)
        self.assertEqual(result1, 45.0)

    def test_point_to_line_distance(self):
        """Test the point_to_line_distance function."""
        # Test case 1: Point is on the line
        line_start = WayPoint(0.0, 0.0)
        line_end = WayPoint(10.0, 10.0)
        point = WayPoint(5.0, 5.0)

        distance1 = point_to_line_distance(line_start, line_end, point)
        self.assertAlmostEqual(distance1, 0.0)

        # Test case 2: Point is not on the line but projection falls within segment
        point2 = WayPoint(7.0, 4.0)
        distance2 = point_to_line_distance(line_start, line_end, point2)
        self.assertAlmostEqual(distance2, 2.12, places=2)

        # Test case 3: Projection falls outside the segment
        point3 = WayPoint(15.0, 12.0)
        distance3 = point_to_line_distance(line_start, line_end, point3)
        self.assertEqual(distance3, float("inf"))

        # Test case 4: Line start and end are the same point
        line_start5 = WayPoint(5.0, 5.0)
        line_end5 = WayPoint(5.0, 5.0)
        point5 = WayPoint(7.0, 7.0)
        distance5 = point_to_line_distance(line_start5, line_end5, point5)
        self.assertEqual(distance5, float("inf"))
