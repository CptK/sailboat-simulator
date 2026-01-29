"""Test cases for the WayPoint class."""

import numpy as np
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from route_planner.waypoint import WayPoint


class TestWayPoint(unittest.TestCase):
    """Test cases for the WayPoint class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.waypoint = WayPoint(10.0, 20.0)
        self.soft_waypoint = WayPoint(10.0, 20.0, is_soft=True)

    def test_init(self):
        """Test initialization of WayPoint class."""
        self.assertEqual(self.waypoint.east, 10.0)
        self.assertEqual(self.waypoint.north, 20.0)
        self.assertFalse(self.waypoint.is_soft)
        self.assertTrue(self.soft_waypoint.is_soft)

    def test_from_coordinates(self):
        """Test creation of WayPoint from coordinates."""
        wp = WayPoint.from_coordinates(10.0, 20.0)  # pylint: disable=invalid-name
        self.assertEqual(wp.east, 10.0)
        self.assertEqual(wp.north, 20.0)
        self.assertFalse(wp.is_soft)

        # Test with all parameters
        wp2 = WayPoint.from_coordinates(
            15.0,
            25.0,
            is_soft=True,
            name="test",
            identifier=42,
        )
        self.assertEqual(wp2.east, 15.0)
        self.assertEqual(wp2.north, 25.0)
        self.assertTrue(wp2.is_soft)
        self.assertEqual(wp2.name, "test")
        self.assertEqual(wp2.identifier, 42)

    def test_distance(self):
        """Test distance calculation between waypoints and locations."""
        # Create another waypoint for comparison
        other_waypoint = WayPoint(13.0, 24.0)

        # Test distance to another WayPoint
        dist1 = self.waypoint.distance(other_waypoint)
        self.assertEqual(dist1, 5.0)

        # Test distance to another WayPoint
        dist2 = self.waypoint.distance(other_waypoint)
        self.assertEqual(dist2, 5.0)

    def test_translate_cardinal_directions(self):
        wp = WayPoint(east=0.0, north=0.0, is_soft=False)

        wp_north, vec_north = wp.translate(0.0, 1.0)
        np.testing.assert_allclose(
            [wp_north.east, wp_north.north],
            [0.0, 1.0],
            atol=1e-8,
        )
        np.testing.assert_allclose(vec_north, [0.0, 1.0], atol=1e-8)

        wp_east, vec_east = wp.translate(90.0, 1.0)
        np.testing.assert_allclose(
            [wp_east.east, wp_east.north],
            [1.0, 0.0],
            atol=1e-8,
        )
        np.testing.assert_allclose(vec_east, [1.0, 0.0], atol=1e-8)

    def test_eq(self):
        """Test equality comparison."""
        # Same values should be equal
        wp1 = WayPoint.from_coordinates(10.0, 20.0)
        wp2 = WayPoint.from_coordinates(10.0, 20.0)
        self.assertEqual(wp1, wp2)

        # Different values should not be equal
        wp3 = WayPoint.from_coordinates(11.0, 20.0)
        self.assertNotEqual(wp1, wp3)

        # Different is_soft value should make them not equal
        wp4 = WayPoint.from_coordinates(10.0, 20.0, is_soft=True)
        self.assertNotEqual(wp1, wp4)

        # Non-WayPoint objects should not be equal
        self.assertNotEqual(wp1, "not a waypoint")

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        np_array = self.waypoint.to_numpy()
        self.assertIsInstance(np_array, np.ndarray)
        self.assertEqual(np_array.shape, (2,))
        np.testing.assert_array_equal(np_array, np.array([10.0, 20.0]))

    def test_add(self):
        """Test addition operator."""
        # Test adding two WayPoints
        wp1 = WayPoint.from_coordinates(10.0, 20.0)
        wp2 = WayPoint.from_coordinates(5.0, 7.0)
        result1 = wp1 + wp2
        self.assertEqual(result1.east, 15.0)
        self.assertEqual(result1.north, 27.0)

        # Test adding numpy array
        result2 = wp1 + np.array([3.0, 4.0])
        self.assertEqual(result2.east, 13.0)
        self.assertEqual(result2.north, 24.0)

        # Test adding tuple
        result3 = wp1 + (2.0, 3.0)
        self.assertEqual(result3.east, 12.0)
        self.assertEqual(result3.north, 23.0)

        # Test adding list
        result4 = wp1 + [1.0, 2.0]
        self.assertEqual(result4.east, 11.0)
        self.assertEqual(result4.north, 22.0)

        # Test that is_soft is preserved
        soft_wp = WayPoint.from_coordinates(10.0, 20.0, is_soft=True)
        result5 = soft_wp + wp2
        self.assertTrue(result5.is_soft)

    def test_sub(self):
        """Test subtraction operator."""
        # Test subtracting two WayPoints
        wp1 = WayPoint.from_coordinates(10.0, 20.0)
        wp2 = WayPoint.from_coordinates(5.0, 7.0)
        result1 = wp1 - wp2
        self.assertEqual(result1.east, 5.0)
        self.assertEqual(result1.north, 13.0)

        # Test subtracting numpy array
        result2 = wp1 - np.array([3.0, 4.0])
        self.assertEqual(result2.east, 7.0)
        self.assertEqual(result2.north, 16.0)

        # Test subtracting tuple
        result3 = wp1 - (2.0, 3.0)
        self.assertEqual(result3.east, 8.0)
        self.assertEqual(result3.north, 17.0)

        # Test subtracting list
        result4 = wp1 - [1.0, 2.0]
        self.assertEqual(result4.east, 9.0)
        self.assertEqual(result4.north, 18.0)

        # Test that is_soft is preserved
        soft_wp = WayPoint.from_coordinates(10.0, 20.0, is_soft=True)
        result5 = soft_wp - wp2
        self.assertTrue(result5.is_soft)

    def test_mul(self):
        """Test multiplication operator."""
        wp = WayPoint.from_coordinates(10.0, 20.0)  # pylint: disable=invalid-name

        # Test multiplying by another WayPoint
        wp2 = WayPoint.from_coordinates(2.0, 3.0)
        result1 = wp * wp2
        self.assertEqual(result1.east, 20.0)
        self.assertEqual(result1.north, 60.0)

        # Test multiplying by a scalar
        result2 = wp * 2.0
        self.assertEqual(result2.east, 20.0)
        self.assertEqual(result2.north, 40.0)

        # Test multiplying by numpy array
        result3 = wp * np.array([2.0, 3.0])
        self.assertEqual(result3.east, 20.0)
        self.assertEqual(result3.north, 60.0)

        # Test multiplying by tuple
        result4 = wp * (2.0, 3.0)
        self.assertEqual(result4.east, 20.0)
        self.assertEqual(result4.north, 60.0)

        # Test multiplying by list
        result5 = wp * [2.0, 3.0]
        self.assertEqual(result5.east, 20.0)
        self.assertEqual(result5.north, 60.0)

        # Test that is_soft is preserved
        soft_wp = WayPoint.from_coordinates(10.0, 20.0, is_soft=True)
        result6 = soft_wp * 2.0
        self.assertTrue(result6.is_soft)

    def test_str(self):
        """Test string representation."""
        self.assertEqual(str(self.waypoint), "WayPoint((10.0, 20.0), False)")
