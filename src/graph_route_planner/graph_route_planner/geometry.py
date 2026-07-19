"""Small shapely helpers shared across the package."""

import math

from shapely.geometry import Point, Polygon


def bbox_diagonal(geom) -> float:
    """The diagonal of a geometry's bounding box, in metres.

    No two points inside the geometry can be further apart than this, which
    makes it the shortest leg limit that never truncates a legal leg.

    Args:
        geom: Any shapely geometry.

    Returns:
        The diagonal length, or 0.0 for empty geometry.
    """
    if geom.is_empty:
        return 0.0
    x0, y0, x1, y1 = geom.bounds
    return math.hypot(x1 - x0, y1 - y0)


def polygons_of(geom) -> list[Polygon]:
    """Normalise a Polygon / MultiPolygon / empty geometry to a list.

    Args:
        geom: Any shapely areal geometry, possibly empty.

    Returns:
        The constituent polygons, or an empty list.
    """
    if geom.is_empty:
        return []
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def component_containing(water, point: tuple[float, float]) -> Polygon | None:
    """Find the single connected water body containing ``point``.

    Accepts a Polygon or MultiPolygon, so it works both on a whole map and on
    an eroded navigable area, which may have split into several pieces.

    Args:
        water: The navigable geometry to search.
        point: An ``(east, north)`` position in metres.

    Returns:
        The polygon containing ``point``, or None if it is on land.
    """
    p = Point(point)
    for comp in polygons_of(water):
        if comp.covers(p):
            return comp
    return None
