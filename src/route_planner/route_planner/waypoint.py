"""A waypoint that can be used in a path."""

from __future__ import annotations

import numpy as np
from typing import cast
from shapely.geometry import Point
from shapely.affinity import translate as shapely_translate


class WayPoint:
    """A waypoint that can be used in a path."""

    def __init__(
        self,
        east: float,
        north: float,
        name: str | None = None,
        identifier: int | None = None,
        is_soft: bool = False
    ) -> None:
        self.east = east
        self.north = north
        self.is_soft = is_soft
        self.name = name
        self.identifier = identifier

    @classmethod
    def from_coordinates(
        cls,
        east: float,
        north: float,
        is_soft: bool = False,
        name: str | None = None,
        identifier: int | None = None,
    ) -> WayPoint:
        return cls(east, north, is_soft=is_soft, name=name, identifier=identifier)

    def distance(self, other: WayPoint) -> float:
        """Computes the Euclidean distance to another waypoint.

        Args:
            other: The other waypoint

        Returns:
            The Euclidean distance in meters
        """
        return cast(float, np.linalg.norm(self.to_numpy() - other.to_numpy()))

    def translate(self, direction: float, distance: float) -> tuple[WayPoint, np.ndarray]:
        """Translates this location.

        Args:
            direction: The direction angle in degrees (``0`` is north, clockwise)
            distance: The distance to translate bin meters

        Returns:
            The translated polygon and the translation vector ``(x_offset, y_offset)`` in meters
            that can be used to reconstruct the original polygon
        """
        x_offset = distance * np.sin(np.radians(direction))
        y_offset = distance * np.cos(np.radians(direction))

        translated_location = shapely_translate(Point(self.east, self.north), xoff=x_offset, yoff=y_offset)
        translated_waypoint = WayPoint.from_coordinates(
            translated_location.x, translated_location.y, is_soft=self.is_soft, name=self.name, identifier=self.identifier
        )
        return translated_waypoint, np.array([x_offset, y_offset])

    def __eq__(self, value):
        return (
            isinstance(value, WayPoint) and \
                self.east == value.east and \
                self.north == value.north and \
                self.is_soft == value.is_soft and \
                self.name == value.name and \
                self.identifier == value.identifier
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([self.east, self.north])

    def __str__(self) -> str:
        return f"WayPoint(({self.east}, {self.north}), {self.is_soft})"

    def __add__(self, other: np.ndarray | WayPoint | tuple | list) -> WayPoint:
        if isinstance(other, WayPoint):
            return WayPoint.from_coordinates(self.east + other.east, self.north + other.north, self.is_soft)
        else:
            return WayPoint.from_coordinates(self.east + other[0], self.north + other[1], self.is_soft)

    def __sub__(self, other: np.ndarray | WayPoint | tuple | list) -> WayPoint:
        if isinstance(other, WayPoint):
            return WayPoint.from_coordinates(self.east - other.east, self.north - other.north, self.is_soft)
        else:
            return WayPoint.from_coordinates(self.east - other[0], self.north - other[1], self.is_soft)

    def __mul__(self, other: WayPoint | np.ndarray | tuple | list | float | int) -> WayPoint:
        if isinstance(other, WayPoint):
            return WayPoint.from_coordinates(self.east * other.east, self.north * other.north, self.is_soft)
        elif isinstance(other, (float, int)):
            other = cast(float, other)
            return WayPoint.from_coordinates(self.east * other, self.north * other, self.is_soft)
        else:
            return WayPoint.from_coordinates(self.east * other[0], self.north * other[1], self.is_soft)
