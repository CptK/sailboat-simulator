"""The synthetic demo scenario.

Kept apart from `sailing`: that module is the boat's physics, this is one
particular made-up world. Mixing them meant importing the physics dragged in a
hardcoded start, goal and obstacle list.
"""

from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from graph_route_planner.geometry import polygons_of

Point = tuple[float, float]


@dataclass(frozen=True)
class Scenario:
    """A planning problem: somewhere to sail, and two points to sail between.

    Args:
        name: Human-readable label.
        water: Navigable water as a MultiPolygon; holes are land.
        start: Start position, ``(east, north)`` in metres.
        goal: Goal position, ``(east, north)`` in metres.
    """

    name: str
    water: MultiPolygon
    start: Point
    goal: Point


# Islands in a rectangular basin. The planner works on water, so the scenario
# is the basin with the islands cut out — the same shape a loaded map produces.
_BASIN = box(-10, 0, 160, 150)

_ISLANDS = [
    Polygon([(10, 75), (23, 64), (30, 80), (42, 81), (44, 93), (22, 94)]),
    Polygon([(26, 20), (74, 9), (63, 30), (40, 33)]),
    Polygon([(53, 57), (62, 45), (83, 56), (86, 51), (91, 60), (87, 71), (75, 64), (57, 70)]),
    Polygon([(47, 110), (50, 102), (69, 93), (84, 99), (92, 97), (94, 115), (79, 121), (67, 118), (57, 120)]),
    Polygon([(107, 15), (142, 19), (138, 39), (111, 35)]),
    Polygon([(106, 72), (125, 60), (142, 76), (125, 98), (110, 95)]),
]


def synthetic() -> Scenario:
    """Build the built-in demo scenario.

    Returns:
        A Scenario with a 170x150 m basin containing six islands.
    """
    water = _BASIN.difference(unary_union(_ISLANDS))
    return Scenario(
        name="synthetic basin",
        water=MultiPolygon(polygons_of(water)),
        start=(90, 20),
        goal=(70, 130),
    )
