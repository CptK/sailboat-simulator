"""Utility functions for the route planner package."""

from typing import cast
import numpy as np

from route_planner.waypoint import WayPoint


def is_sailable_direction(
    start: WayPoint,
    end: WayPoint,
    wind_direction: float,
    minimum_tack_angle: float = 45,
) -> bool:
    """Returns True if the direction from start to end is sailable given the wind direction.

    Args:
        start: The start location of the path
        end: The end location of the path
        wind_direction: The direction of the wind in [0, 360] degrees
        origin: The global reference to be used for back-projection
        minimum_tack_angle: The minimum angle between the wind direction and the path direction
            for the path to be considered sailable

    Returns:
        bool: True if the path is sailable, False otherwise
    """
    direction = bearing(start, end)  # in [0, 360] degrees
    return angular_distance(direction, wind_direction) >= minimum_tack_angle


def bearing(start: WayPoint, end: WayPoint) -> float:
    """Returns the bearing from the start location to the end location.

    Args:
        start: The start location
        end: The end location

    Returns:
        float: The bearing from the start location to the end location, ranged from 0 to 360 degrees
    """
    delta_east = end.east - start.east
    delta_north = end.north - start.north

    angle_rad = np.arctan2(delta_east, delta_north)
    angle_deg = np.degrees(angle_rad)
    bearing_deg = angle_deg % 360

    return cast(float, bearing_deg)


def angular_distance(angle1: float, angle2: float) -> float:
    """Calculate the angular distance between two angles.

    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        float: The angular distance in degrees
    """
    diff = (angle2 - angle1) % 360
    return min(diff, 360 - diff)


def point_to_line_distance(
    line_start: WayPoint,
    line_end: WayPoint,
    point: WayPoint,
) -> float:
    """Calculate the orthogonal distance from a point to a line segment.

    Args:
        line_start: tuple (WayPoint) representing the start point of the line
        line_end: tuple (WayPoint) representing the end point of the line
        point: tuple (WayPoint) representing the point to measure distance from

    Returns:
        float: The orthogonal distance if the projection falls within the line segment,
            float('inf') otherwise
    """
    x1, y1 = line_start.east, line_start.north  # pylint: disable=invalid-name
    x2, y2 = line_end.east, line_end.north  # pylint: disable=invalid-name
    x0, y0 = point.east, point.north  # pylint: disable=invalid-name

    # Calculate the line vector
    dx = x2 - x1  # pylint: disable=invalid-name
    dy = y2 - y1  # pylint: disable=invalid-name

    # If start and end points are the same, return infinity
    if dx == 0 and dy == 0:
        return float("inf")

    # Calculate the squared length of the line segment
    line_length_squared = dx * dx + dy * dy

    # Calculate the parameter t that minimizes the distance
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / line_length_squared  # pylint: disable=invalid-name

    # If t is outside [0,1], the orthogonal projection falls outside the segment
    if t < 0 or t > 1:
        return float("inf")

    # Calculate the projection point
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    # Calculate the distance between the point and its projection
    distance = ((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2) ** 0.5

    return cast(float, distance)
