"""Tests for KML loading, projection and the nesting classification."""

import math

import pytest

from graph_route_planner import MAPS_DIR
from graph_route_planner.map_loader import _metres_per_degree, load_kml

TPL = ('<?xml version="1.0" encoding="UTF-8"?>'
       '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>{}</Document></kml>')
PM = ('<Placemark><name>{name}</name><Polygon><outerBoundaryIs><LinearRing>'
      '<coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')


def ring(cx: float, cy: float, r: float, n: int = 32) -> str:
    """A circle of roughly r metres radius, as KML lon,lat,alt text."""
    dlat = r / 111320.0
    dlon = r / (111320.0 * math.cos(math.radians(cy)))
    pts = [(cx + dlon * math.cos(2 * math.pi * i / n),
            cy + dlat * math.sin(2 * math.pi * i / n)) for i in range(n)]
    pts.append(pts[0])
    return " ".join(f"{x},{y},0" for x, y in pts)


def rect(lon0: float, lat0: float, lon1: float, lat1: float) -> str:
    """An axis-aligned rectangle, as KML lon,lat,alt text.

    Exact, unlike `ring`, whose circles are 32-gons — useful when a test needs
    two edges to coincide precisely.
    """
    pts = [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1), (lon0, lat0)]
    return " ".join(f"{x},{y},0" for x, y in pts)


def write_kml(path, specs) -> str:
    """Write a KML of named circles; specs is [(name, (lon, lat, radius_m))]."""
    return write_polys(path, [(n, ring(*geom)) for n, geom in specs])


def write_polys(path, named_coords) -> str:
    """Write a KML from raw coordinate strings; [(name, coords)]."""
    path.write_text(TPL.format("".join(
        PM.format(name=n, coords=c) for n, c in named_coords)))
    return str(path)


@pytest.fixture
def landgraben():
    return load_kml(MAPS_DIR / "Landgraben.kml")


@pytest.fixture
def nested():
    return load_kml(MAPS_DIR / "Test.kml")


def test_rejects_file_without_polygons(tmp_path):
    empty = tmp_path / "empty.kml"
    empty.write_text(TPL.format(""))
    with pytest.raises(ValueError, match="No polygons"):
        load_kml(empty)


def test_loads_features_and_one_water_body(landgraben):
    assert len(landgraben.features) == 3
    assert len(landgraben.components) == 1


def test_island_and_fountain_are_land(landgraben):
    by_name = {f.name: f for f in landgraben.features}
    assert by_name["Landgraben"].is_water
    assert not by_name["Landgraben Insel"].is_water
    assert not by_name["Landgraben Fontäne"].is_water


def test_water_excludes_the_islands(landgraben):
    """Water is the lake minus what sits in it, not the raw outline."""
    outline = next(f for f in landgraben.features if f.name == "Landgraben")
    assert landgraben.water.area < outline.polygon.area


def test_origin_maps_to_zero(landgraben):
    assert landgraben.to_lonlat(0.0, 0.0) == pytest.approx(landgraben.origin)


def test_to_lonlat_inverts_the_projection(landgraben):
    """Project a known KML vertex forward by hand, then invert it back."""
    lon0, lat0 = landgraben.origin
    m_lon, m_lat = _metres_per_degree(lat0)
    lon, lat = 8.651986631657554, 49.87749941060768   # first vertex in the file
    x, y = (lon - lon0) * m_lon, (lat - lat0) * m_lat

    assert landgraben.to_lonlat(x, y) == pytest.approx((lon, lat), abs=1e-11)


@pytest.mark.parametrize("lat, exp_lon, exp_lat", [
    (0.0, 111319.5, 110574.3),
    (45.0, 78847.0, 111132.1),
    (50.0, 71698.1, 111229.3),
])
def test_metres_per_degree_matches_published_wgs84(lat, exp_lon, exp_lat):
    """Check the scale against published WGS84 values rather than a sphere."""
    m_lon, m_lat = _metres_per_degree(lat)
    assert m_lon == pytest.approx(exp_lon, abs=2.5)
    assert m_lat == pytest.approx(exp_lat, abs=2.5)


def test_planar_distance_is_close_to_spherical(landgraben):
    """Sanity check only: the tangent plane roughly agrees with a sphere.

    The tolerance is 0.5%, not centimetres. This projection uses the WGS84
    radii of curvature while haversine assumes a mean sphere, and at this
    latitude the prime vertical radius is 0.3% above that mean — so an
    east-west leg legitimately disagrees by that much. The ellipsoid is the
    more accurate of the two; see the WGS84 test above for the real check.
    """
    coords = list(landgraben.components[0].exterior.coords)
    a, b = coords[0], coords[10]
    planar = math.dist(a, b)

    (lon1, lat1), (lon2, lat2) = landgraben.to_lonlat(*a), landgraben.to_lonlat(*b)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    hav = 2 * 6371008.8 * math.asin(math.sqrt(h))

    assert planar == pytest.approx(hav, rel=0.005)


# ── nesting: the lake-on-an-island case ───────────────────────────────────────

def test_nesting_depths(nested):
    depths = {f.name: f.depth for f in nested.features}
    assert depths["Landgraben"] == 0
    assert depths["Landgraben Insel"] == 1
    assert depths["Lake on Island"] == 2


def test_lake_on_island_is_water_again(nested):
    """Even nesting depth is water: a lake on an island is sailable."""
    lake = next(f for f in nested.features if f.name == "Lake on Island")
    assert lake.is_water
    assert len(nested.components) == 2


def test_lake_on_island_survives_dissolve(nested):
    """Subtracting all land at once would delete the inner lake entirely."""
    inner = min(nested.components, key=lambda c: c.area)
    assert inner.area > 0
    assert nested.water.covers(inner.representative_point())


def test_component_for_distinguishes_the_two_bodies(nested):
    outer, inner = nested.components[0], nested.components[1]
    assert nested.component_for(outer.representative_point().coords[0]) is not None
    assert nested.component_for(inner.representative_point().coords[0]).area == inner.area
    assert nested.component_for(outer.representative_point().coords[0]).area == outer.area


def test_point_on_land_is_not_on_any_water(nested):
    insel = next(f for f in nested.features if f.name == "Landgraben Insel")
    lake = next(f for f in nested.features if f.name == "Lake on Island")
    land = insel.polygon.difference(lake.polygon).representative_point()
    assert nested.component_for((land.x, land.y)) is None


# ── sibling water bodies ──────────────────────────────────────────────────────

def test_two_disjoint_lakes_are_two_bodies(tmp_path):
    """Neither lake contains the other; both must stay water."""
    path = write_kml(tmp_path / "two.kml", [
        ("West Lake", (8.6500, 49.877, 40)),
        ("West Island", (8.6500, 49.877, 15)),
        ("East Lake", (8.6530, 49.877, 40)),
    ])
    m = load_kml(path)
    assert len(m.components) == 2
    assert {f.name for f in m.features if not f.is_water} == {"West Island"}


def test_touching_island_still_counts_as_contained(tmp_path):
    """An island drawn against the shore is still land, not a second body.

    Shapely's contains() permits boundary contact, so a peninsula-style island
    touching the shoreline stays classified as land.
    """
    # Rectangles, so the island's north edge lies exactly on the lake's.
    path = write_polys(tmp_path / "touch.kml", [
        ("Lake", rect(8.6500, 49.8770, 8.6510, 49.8780)),
        ("Island", rect(8.6503, 49.8773, 8.6507, 49.8780)),
    ])
    m = load_kml(path)
    by_name = {f.name: f for f in m.features}
    assert by_name["Lake"].is_water
    assert not by_name["Island"].is_water
    assert len(m.components) == 1


def test_duplicate_placemark_does_not_erase_the_water(tmp_path):
    """A duplicated polygon must not turn the lake into land.

    Two identical shapes each satisfy shapely's contains(), so a naive depth
    count puts both on depth 1 and classifies both as land — silently leaving
    a map with no water at all. Duplicating a placemark is one click in Google
    Earth, so this has to collapse rather than invert.
    """
    path = write_kml(tmp_path / "dup.kml", [
        ("Lake", (8.65, 49.877, 40)),
        ("Lake copy", (8.65, 49.877, 40)),
    ])
    m = load_kml(path)
    assert all(f.depth == 0 and f.is_water for f in m.features)
    assert len(m.components) == 1
    assert m.water.area > 0
