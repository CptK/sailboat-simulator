"""
Load sailing maps from KML files.

A map is a set of polygons in geographic (lon/lat) coordinates. Since the
planner works in a flat metric space, polygons are projected onto a local
tangent plane in metres, centred on the map itself:

    +x = east, +y = north, origin = centre of the map's bounding box

This matches the (east, north) convention used by `sailing.SailingModel`.

Polygons are classified by nesting depth: a polygon contained by an even
number of others is water, an odd number is land. So a lake is water, an
island in it is land, and a lake on that island is water again. The result
is `water` — a MultiPolygon of disjoint navigable bodies, each with its
islands punched out as holes.

There is deliberately no single "boundary". A boat sits in exactly one water
component, and the others are unreachable without a portage, so the planner
picks its component from the start point and ignores the rest.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from graph_route_planner.geometry import component_containing, polygons_of

# WGS84 ellipsoid
_A  = 6378137.0                 # semi-major axis (m)
_F  = 1 / 298.257223563         # flattening
_E2 = _F * (2 - _F)             # first eccentricity squared


@dataclass
class Feature:
    """One polygon as it was drawn, kept for labelling and debugging."""

    name: str
    polygon: Polygon
    depth: int          # 0 = outermost; even = water, odd = land

    @property
    def is_water(self) -> bool:
        return self.depth % 2 == 0


@dataclass
class SailMap:
    """A map in local metric coordinates, plus the projection back to lon/lat."""

    name: str
    water: MultiPolygon                        # navigable bodies; holes are land
    origin: tuple[float, float] = (0.0, 0.0)   # (lon, lat) of local (0, 0)
    features: list[Feature] = field(default_factory=list)

    @property
    def components(self) -> list[Polygon]:
        """The disjoint navigable bodies, largest first."""
        return sorted(self.water.geoms, key=lambda p: p.area, reverse=True)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) over every feature, in metres."""
        if not self.features:
            return (0.0, 0.0, 0.0, 0.0)
        xs0, ys0, xs1, ys1 = zip(*(f.polygon.bounds for f in self.features))
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def component_for(self, point: tuple[float, float]) -> Polygon | None:
        """
        The water body containing `point`, or None if it is on land.

        This is how the planner picks its world: everything outside the
        returned component is unreachable without carrying the boat.
        """
        return component_containing(self.water, point)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Invert the projection: local metres → (lon, lat) degrees."""
        lon0, lat0 = self.origin
        m_lon, m_lat = _metres_per_degree(lat0)
        return (lon0 + x / m_lon, lat0 + y / m_lat)


def _metres_per_degree(lat0: float) -> tuple[float, float]:
    """
    Metres per degree of longitude and latitude on the tangent plane at lat0.

    Uses the WGS84 radii of curvature, which is accurate to centimetres over
    the few hundred metres a single map spans.
    """
    phi = math.radians(lat0)
    s = math.sin(phi)
    n = _A / math.sqrt(1 - _E2 * s * s)                 # prime vertical radius
    m = _A * (1 - _E2) / (1 - _E2 * s * s) ** 1.5       # meridional radius
    return (math.radians(1.0) * n * math.cos(phi), math.radians(1.0) * m)


def _parse_coords(text: str) -> list[tuple[float, float]]:
    """Parse a KML <coordinates> blob: 'lon,lat[,alt] lon,lat[,alt] …'."""
    points = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            points.append((float(parts[0]), float(parts[1])))
    return points


def _parse_placemarks(path: Path) -> list[tuple[str, list, list]]:
    """
    Extract (name, outer_ring, inner_rings) for every polygon in the file.

    Rings are lists of (lon, lat). Namespaces are matched with a wildcard so
    this works regardless of the KML dialect (Google Earth, OGC, …).
    """
    root = ET.parse(path).getroot()
    polygons = []

    for placemark in root.iterfind(".//{*}Placemark"):
        name_el = placemark.find("{*}name")
        name = name_el.text.strip() if name_el is not None and name_el.text else "unnamed"

        for poly in placemark.iterfind(".//{*}Polygon"):
            outer_el = poly.find(".//{*}outerBoundaryIs//{*}coordinates")
            if outer_el is None or not outer_el.text:
                continue
            outer = _parse_coords(outer_el.text)
            if len(outer) < 3:
                continue

            inners = []
            for inner_el in poly.iterfind(".//{*}innerBoundaryIs//{*}coordinates"):
                if inner_el.text:
                    ring = _parse_coords(inner_el.text)
                    if len(ring) >= 3:
                        inners.append(ring)

            polygons.append((name, outer, inners))

    return polygons


def _strictly_contains(outer: Polygon, inner: Polygon) -> bool:
    """Whether ``outer`` contains ``inner`` and is not the same shape.

    Shapely's ``contains`` is true for two identical polygons, so a duplicated
    placemark — one click in Google Earth — would make each copy "contain" the
    other. Both would land on depth 1, both would be classified as land, and
    the whole lake would silently disappear. Equal shapes nest in neither
    direction, so duplicates simply collapse into one body of water.
    """
    return outer.contains(inner) and not outer.equals(inner)


def _classify(named: list[tuple[str, Polygon]]) -> list[Feature]:
    """Tag each polygon with its nesting depth.

    Args:
        named: ``(name, polygon)`` pairs in local metric coordinates.

    Returns:
        A Feature per polygon, with depth = how many polygons strictly
        contain it. Even depth is water, odd is land.
    """
    return [
        Feature(name, poly,
                sum(1 for _, other in named
                    if other is not poly and _strictly_contains(other, poly)))
        for name, poly in named
    ]


def immediate_children(features: list[Feature], f: Feature) -> list[Feature]:
    """The features nested directly inside `f`, one level down."""
    return [g for g in features
            if g.depth == f.depth + 1 and f.polygon.contains(g.polygon)]


def build_water(features: list[Feature]) -> MultiPolygon:
    """
    Dissolve classified features into disjoint navigable water bodies.

    Each water feature has only its *immediate* children subtracted. Taking
    the union of all land at once would be wrong: a lake on an island lies
    inside that island, so subtracting the island would delete the lake with
    it — the boat would find itself aground on water it is floating in.
    """
    parts = []
    for f in features:
        if not f.is_water:
            continue
        kids = [g.polygon for g in immediate_children(features, f)]
        parts.append(f.polygon.difference(unary_union(kids)) if kids else f.polygon)

    water = unary_union(parts) if parts else Polygon()
    return MultiPolygon(polygons_of(water))


def load_kml(path: str | Path, name: str | None = None) -> SailMap:
    """
    Load a KML file into a SailMap in local metric coordinates.

    Every <Polygon> in every <Placemark> is read, classified by nesting depth
    and dissolved into water bodies. Raises ValueError if the file has no
    usable polygons.
    """
    path = Path(path)
    raw = _parse_placemarks(path)
    if not raw:
        raise ValueError(f"No polygons found in {path}")

    # Project about the centre of everything we just read, so the local frame
    # does not depend on which polygon happens to come first.
    all_lons = [lon for _, outer, _ in raw for lon, _ in outer]
    all_lats = [lat for _, outer, _ in raw for _, lat in outer]
    lon0 = (min(all_lons) + max(all_lons)) / 2
    lat0 = (min(all_lats) + max(all_lats)) / 2
    m_lon, m_lat = _metres_per_degree(lat0)

    def project(ring: list) -> list:
        return [((lon - lon0) * m_lon, (lat - lat0) * m_lat) for lon, lat in ring]

    named: list[tuple[str, Polygon]] = []
    for pname, outer, inners in raw:
        poly = Polygon(project(outer), [project(r) for r in inners])
        if not poly.is_valid:
            poly = poly.buffer(0)   # repair self-intersections from hand-drawn maps
        named.append((pname, poly))

    features = _classify(named)
    return SailMap(
        name=name or path.stem,
        water=build_water(features),
        origin=(lon0, lat0),
        features=features,
    )
