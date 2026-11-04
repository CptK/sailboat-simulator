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

OpenStreetMap coastline exports (e.g. from overpass-turbo, tagged
natural=coastline) work differently: they contain only land. Small islands are
polygons; large islands and mainland are split into many coastline ways
(LineStrings), which by OSM convention have the land on their left. Those are
joined back into rings, and ways that run off the edge of the export are closed
along a crop box, which then becomes the sea. See `_coastline_features`.

Any map can be cropped (to a lon/lat box, which then becomes the local origin),
scaled (to shrink a real archipelago into something a small boat can sail
through), simplified and stripped of tiny islands. `load_map` applies those
options from the map catalog (maps/catalog.yaml), so every consumer that loads
a map by name sees exactly the same geometry.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from shapely import STRtree
from shapely.affinity import scale as scale_geometry
from shapely.geometry import LinearRing, LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union

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
    scale: float = 1.0                         # local metres per real metre

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
        return (lon0 + x / (m_lon * self.scale), lat0 + y / (m_lat * self.scale))


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


def _is_osm_coastline(root: ET.Element) -> bool:
    """Whether a KML is an OpenStreetMap coastline export (natural=coastline)."""
    for data in root.iterfind(".//{*}Data"):
        if data.get("name") == "natural" and (data.findtext("{*}value") or "").strip() == "coastline":
            return True
    return False


def _parse_linestrings(root: ET.Element) -> list[list[tuple[float, float]]]:
    """Every <LineString> in the file, as a list of (lon, lat)."""
    lines = []
    for ls in root.iterfind(".//{*}LineString"):
        coords = _parse_coords(ls.findtext("{*}coordinates") or "")
        if len(coords) >= 2:
            lines.append(coords)
    return lines


#: Distance (real metres) either side of a coastline at which points are
#: sampled to decide which side is land. Small against any real channel, large
#: against the floating-point noise of the projection.
_SIDE_PROBE_M = 0.5


def _coastline_features(polygons: list[tuple[str, Polygon]],
                        lines: list[LineString],
                        area: Polygon,
                        to_lonlat=lambda x, y: (x, y)) -> list[Feature]:
    """Turn an OSM coastline export into features: the sea and its land.

    Args:
        polygons: Named land polygons (islands mapped as closed areas).
        lines: Coastline ways, land on their left, in local metres.
        area: The box to keep; everything in it that is not land is sea.
        to_lonlat: Maps local metres back to (lon, lat), for error messages.

    Returns:
        One water Feature (the area, depth 0) and one land Feature per piece of
        land inside it (depth 1), so `build_water` can treat it like any map.

    Raises:
        ValueError: If a coastline way that does not close on itself ends
            inside the area — it is then impossible to tell where that land
            ends. Choosing a smaller crop fixes this.
    """
    merged = linemerge(lines) if lines else None
    chains = ([] if merged is None or merged.is_empty else
              list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged])

    land_parts: list[Polygon] = [p for _, p in polygons]
    lakes: list[Polygon] = []
    open_chains: list[LineString] = []
    for chain in chains:
        if chain.is_ring:
            ring = LinearRing(chain.coords)
            # Land on the left: an island's coastline runs counter-clockwise.
            # A clockwise ring is water enclosed by land (a lake or lagoon).
            (land_parts if ring.is_ccw else lakes).append(Polygon(ring).buffer(0))
        else:
            open_chains.append(chain)

    for chain in open_chains:
        for end in (chain.coords[0], chain.coords[-1]):
            if area.contains(Point(end)):
                lon, lat = to_lonlat(*end)
                raise ValueError(
                    f"A coastline runs off the edge of the export but ends at "
                    f"lon {lon:.4f}, lat {lat:.4f}, inside the map area, so it cannot "
                    f"be closed. Crop the map (the 'crop' option in maps/catalog.yaml) "
                    f"so that this point lies outside the crop box.")

    if open_chains:
        # The open chains cut the area into faces; each is all land or all sea.
        noded = unary_union([area.exterior, *open_chains])
        faces = [f for f in polygonize(noded) if area.contains(f.representative_point())]
        tree = STRtree(faces)
        votes = [0] * len(faces)            # > 0 land, < 0 sea

        def vote(x: float, y: float, weight: int) -> None:
            hits = tree.query(Point(x, y), predicate="intersects")
            if len(hits):
                votes[int(hits[0])] += weight

        for chain in open_chains:
            pts = list(chain.coords)
            step = max(1, len(pts) // 200)          # a few hundred probes per chain
            for (x0, y0), (x1, y1) in list(zip(pts, pts[1:]))[::step]:
                dx, dy = x1 - x0, y1 - y0
                length = math.hypot(dx, dy)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                if length < 1e-9 or not area.contains(Point(mx, my)):
                    continue
                nx, ny = -dy / length * _SIDE_PROBE_M, dx / length * _SIDE_PROBE_M
                vote(mx + nx, my + ny, +1)          # left of the way: land
                vote(mx - nx, my - ny, -1)          # right of the way: sea
        land_parts += [f for f, v in zip(faces, votes) if v > 0]

    land = unary_union(land_parts) if land_parts else Polygon()
    if lakes:
        land = land.difference(unary_union(lakes))
    land = land.intersection(area)

    # Name each piece of land after the largest named island inside it.
    named = sorted(((n, p) for n, p in polygons if n and n != "unnamed"),
                   key=lambda np_: -np_[1].area)
    features = [Feature("sea", area, 0)]
    for piece in polygons_of(land):
        label = next((n for n, p in named if piece.intersects(p.representative_point())),
                     "coast")
        features.append(Feature(label, piece, 1))
    return features


def _crop_features(features: list[Feature], area: Polygon) -> list[Feature]:
    """Clip hand-drawn features to a box, dropping those left empty."""
    out = []
    for f in features:
        for piece in polygons_of(f.polygon.intersection(area)):
            out.append(Feature(f.name, piece, f.depth))
    return out


def _rescale_features(features: list[Feature], factor: float, simplify: float,
                      min_area: float) -> list[Feature]:
    """Scale about the local origin, simplify, and drop land below min_area.

    Tolerances are in the scaled frame, i.e. in the metres the boat sails.
    """
    out = []
    for f in features:
        poly = f.polygon
        if factor != 1.0:
            poly = scale_geometry(poly, factor, factor, origin=(0.0, 0.0))
        if simplify > 0:
            poly = poly.simplify(simplify, preserve_topology=True)
        for piece in polygons_of(poly if poly.is_valid else poly.buffer(0)):
            if not f.is_water and piece.area < min_area:
                continue
            out.append(Feature(f.name, piece, f.depth))
    return out


def load_kml(path: str | Path, name: str | None = None, *, scale: float = 1.0,
             crop: tuple[float, float, float, float] | None = None,
             simplify: float = 0.0, min_area: float = 0.0) -> SailMap:
    """
    Load a KML file into a SailMap in local metric coordinates.

    Hand-drawn maps: every <Polygon> in every <Placemark> is read, classified by
    nesting depth and dissolved into water bodies. OpenStreetMap coastline
    exports are recognised by their natural=coastline tags and read as land
    (see `_coastline_features`).

    Args:
        path: The KML file.
        name: Name for the map; defaults to the file name without extension.
        scale: Multiply every distance by this, e.g. 0.02 to sail a real
            archipelago at 1:50. The boat itself is not scaled.
        crop: ``(lon_min, lat_min, lon_max, lat_max)`` to keep. Its centre
            becomes the local origin. Needed for coastline exports whose
            coastline runs off the edge of the file.
        simplify: Drop coastline detail smaller than this, in metres after
            scaling. 0 keeps every vertex.
        min_area: Drop land smaller than this, in m² after scaling.

    Raises:
        ValueError: If the file has no usable polygons or coastline, or the
            coastline cannot be closed inside the map area.
    """
    path = Path(path)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    root = ET.parse(path).getroot()
    raw = _parse_placemarks(path)
    osm = _is_osm_coastline(root)
    lines_ll = _parse_linestrings(root) if osm else []
    if not raw and not lines_ll:
        raise ValueError(f"No polygons found in {path}")

    # Project about the centre of the crop, or of everything we just read, so
    # the local frame does not depend on which polygon happens to come first.
    if crop is not None:
        lon_min, lat_min, lon_max, lat_max = crop
        if not (lon_min < lon_max and lat_min < lat_max):
            raise ValueError(f"crop must be [lon_min, lat_min, lon_max, lat_max], got {crop}")
        lon0, lat0 = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
    else:
        all_lons = [lon for _, outer, _ in raw for lon, _ in outer] + \
                   [lon for line in lines_ll for lon, _ in line]
        all_lats = [lat for _, outer, _ in raw for _, lat in outer] + \
                   [lat for line in lines_ll for _, lat in line]
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

    area = None
    if crop is not None:
        x0, y0 = project([(crop[0], crop[1])])[0]
        x1, y1 = project([(crop[2], crop[3])])[0]
        area = box(x0, y0, x1, y1)

    if osm:
        lines = [LineString(project(line)) for line in lines_ll]
        if area is None:
            # No crop: the sea is everything read, plus a small border so no
            # island touches the edge.
            geoms = [p for _, p in named] + lines
            gx0, gy0, gx1, gy1 = unary_union(geoms).bounds
            pad = 0.05 * max(gx1 - gx0, gy1 - gy0)
            area = box(gx0 - pad, gy0 - pad, gx1 + pad, gy1 + pad)
        features = _coastline_features(
            named, lines, area, lambda x, y: (lon0 + x / m_lon, lat0 + y / m_lat))
    else:
        features = _classify(named)
        if area is not None:
            features = _crop_features(features, area)

    features = _rescale_features(features, scale, simplify, min_area)
    return SailMap(
        name=name or path.stem,
        water=build_water(features),
        origin=(lon0, lat0),
        features=features,
        scale=scale,
    )


#: The catalog of named maps, in the maps directory.
CATALOG_FILE = "catalog.yaml"

#: Options a catalog entry may set, passed on to `load_kml`.
_MAP_OPTIONS = ("scale", "crop", "simplify", "min_area")


def load_catalog(maps_dir: str | Path | None = None) -> dict[str, dict]:
    """The map catalog: map name → its entry (file, scale, crop, …)."""
    if maps_dir is None:
        from graph_route_planner import MAPS_DIR
        maps_dir = MAPS_DIR
    path = Path(maps_dir) / CATALOG_FILE
    if not path.is_file():
        return {}
    with open(path) as f:
        catalog = yaml.safe_load(f) or {}
    return {str(k): (v if isinstance(v, dict) else {"file": v}) for k, v in catalog.items()}


def resolve_map(spec: str | Path, maps_dir: str | Path | None = None) -> tuple[Path, dict]:
    """Find the KML file and load options a map specification refers to.

    Args:
        spec: A catalog name (``"landgraben"``), a file name in the maps
            directory (``"Landgraben.kml"``, which picks up that file's catalog
            entry if it has one), or a path to any KML file.
        maps_dir: Where the maps and catalog live; defaults to ``MAPS_DIR``.

    Returns:
        ``(kml_path, options)``, options being keyword arguments for `load_kml`.

    Raises:
        FileNotFoundError: If the specification matches nothing.
    """
    if maps_dir is None:
        from graph_route_planner import MAPS_DIR
        maps_dir = MAPS_DIR
    maps_dir = Path(maps_dir)
    catalog = load_catalog(maps_dir)
    if not catalog and not Path(spec).is_file():
        raise FileNotFoundError(
            f"No map '{spec}': there is no map catalog at {maps_dir / CATALOG_FILE}. "
            f"Build the sail_maps package (colcon build) and source the workspace "
            f"again (source install/setup.bash) — a shell sourced before sail_maps "
            f"was first built does not know it — or point SAIL_MAPS_DIR at the "
            f"repository's maps/ directory.")

    def options(entry: dict) -> dict:
        return {k: (tuple(entry[k]) if k == "crop" else entry[k])
                for k in _MAP_OPTIONS if entry.get(k) is not None}

    key = str(spec)
    entry = catalog.get(key) or catalog.get(key.lower())
    if entry is not None:
        return maps_dir / entry["file"], options(entry)

    path = Path(spec)
    if not path.is_file():
        path = maps_dir / key
    if path.is_file():
        for entry in catalog.values():
            if entry.get("file") == path.name and path.parent.resolve() == maps_dir.resolve():
                return path, options(entry)
        return path, {}

    known = ", ".join(sorted(catalog)) or "none"
    raise FileNotFoundError(
        f"No map '{spec}': not a name in {maps_dir / CATALOG_FILE} (known: {known}), "
        f"nor a KML file.")


def load_map(spec: str | Path, maps_dir: str | Path | None = None) -> SailMap:
    """Load a map by catalog name, file name in the maps directory, or path.

    This is how the simulation and the planners load maps, so a name means the
    same geometry — crop, scale and all — everywhere.
    """
    path, opts = resolve_map(spec, maps_dir)
    catalog_name = str(spec) if not Path(str(spec)).suffix else None
    return load_kml(path, name=catalog_name, **opts)
