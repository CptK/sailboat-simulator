"""Tests for OpenStreetMap coastline exports, map options and the map catalog.

The synthetic maps are built in local metres around a point in Croatia and
written as overpass-turbo style KML: every placemark tagged natural=coastline,
small islands as polygons, everything else as coastline ways with the land on
their left.
"""

import pytest
from shapely.geometry import Point

from graph_route_planner.map_loader import (
    _metres_per_degree, load_catalog, load_kml, load_map, resolve_map)

LON0, LAT0 = 15.5, 43.8
M_LON, M_LAT = _metres_per_degree(LAT0)
# A crop of +-100 m around (LON0, LAT0); local (0, 0) is its centre.
CROP = (LON0 - 100 / M_LON, LAT0 - 100 / M_LAT, LON0 + 100 / M_LON, LAT0 + 100 / M_LAT)

TAGS = ('<ExtendedData><Data name="natural"><value>coastline</value></Data>'
        '</ExtendedData>')


def ll(points):
    """Local metres → KML 'lon,lat' text."""
    return " ".join(f"{LON0 + x / M_LON:.9f},{LAT0 + y / M_LAT:.9f}" for x, y in points)


def polygon_pm(name, points):
    return (f"<Placemark><name>{name}</name>{TAGS}<Polygon><outerBoundaryIs><LinearRing>"
            f"<coordinates>{ll(points)}</coordinates></LinearRing></outerBoundaryIs>"
            f"</Polygon></Placemark>")


def way_pm(points):
    return (f"<Placemark>{TAGS}<LineString><coordinates>{ll(points)}</coordinates>"
            f"</LineString></Placemark>")


def write(path, placemarks):
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>'
                    '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                    + "".join(placemarks) + "</Document></kml>")
    return path


def square(cx, cy, half, ccw=True):
    pts = [(cx - half, cy - half), (cx + half, cy - half),
           (cx + half, cy + half), (cx - half, cy + half)]
    if not ccw:
        pts.reverse()
    return pts + [pts[0]]


# Mainland east of x = 80, its coastline running south (land on the left = east)
# and split into two ways that must be joined.
MAINLAND_WAYS = [[(80, 150), (80, 0)], [(80, 0), (80, -150)]]


@pytest.fixture
def archipelago(tmp_path):
    ring = square(-50, 50, 10)                     # island mapped as two ways
    lagoon_island = square(-40, -60, 20)           # island with a lagoon in it
    lagoon = square(-40, -60, 5, ccw=False)        # clockwise: water inside land
    return write(tmp_path / "arch.kml", [
        polygon_pm("Polygon Island", square(50, 50, 8)),
        way_pm(ring[:3]), way_pm(ring[2:]),
        way_pm(lagoon_island),
        way_pm(lagoon),
        *[way_pm(w) for w in MAINLAND_WAYS],
    ])


def on_water(sail_map, x, y):
    return sail_map.water.contains(Point(x, y))


def test_open_sea_is_water(archipelago):
    m = load_kml(archipelago, crop=CROP)
    assert on_water(m, 0, 0)
    assert on_water(m, 70, 0)       # just west of the mainland coast
    assert on_water(m, -95, 95)     # a corner of the crop


def test_island_polygon_is_land(archipelago):
    m = load_kml(archipelago, crop=CROP)
    assert not on_water(m, 50, 50)


def test_island_joined_from_two_ways_is_land(archipelago):
    m = load_kml(archipelago, crop=CROP)
    assert not on_water(m, -50, 50)


def test_mainland_is_closed_along_the_crop_edge(archipelago):
    m = load_kml(archipelago, crop=CROP)
    assert not on_water(m, 90, 0)
    assert not on_water(m, 95, -95)


def test_lagoon_inside_an_island_is_water(archipelago):
    m = load_kml(archipelago, crop=CROP)
    assert on_water(m, -40, -60)        # in the lagoon
    assert not on_water(m, -55, -75)    # on the island around it
    # The lagoon is its own water body: the open sea cannot reach it.
    assert m.component_for((-40, -60)) is not m.component_for((0, 0))


def test_way_direction_decides_which_side_is_land(tmp_path):
    reversed_ways = [list(reversed(w)) for w in reversed(MAINLAND_WAYS)]
    m = load_kml(write(tmp_path / "flipped.kml", [way_pm(w) for w in reversed_ways]),
                 crop=CROP)
    assert on_water(m, 90, 0)           # land on the left is now the west side
    assert not on_water(m, 0, 0)


def test_open_coastline_ending_inside_the_crop_is_rejected(tmp_path):
    kml = write(tmp_path / "short.kml", [way_pm([(80, 150), (80, 0)])])
    with pytest.raises(ValueError, match=r"lon \d+\.\d+, lat \d+\.\d+"):
        load_kml(kml, crop=CROP)


def test_scale_shrinks_the_map_about_the_crop_centre(archipelago):
    m = load_kml(archipelago, crop=CROP, scale=0.5)
    assert m.scale == 0.5
    assert not on_water(m, 25, 25)      # the polygon island, now at half distance
    assert on_water(m, 35, 0)           # sea west of the mainland, now at x < 40
    assert not on_water(m, 45, 0)       # mainland
    x0, y0, x1, y1 = m.bounds
    assert x1 - x0 == pytest.approx(100, abs=0.5)


def test_to_lonlat_undoes_scale(archipelago):
    m = load_kml(archipelago, crop=CROP, scale=0.02)
    lon, lat = m.to_lonlat(1.0, 0.0)    # 1 m east on the map = 50 m east in reality
    assert (lon - LON0) * M_LON == pytest.approx(50.0, rel=1e-6)
    assert lat == pytest.approx(LAT0)


def test_min_area_drops_small_islands(archipelago):
    # At 1:10 the 16 x 16 m polygon island covers 2.56 m².
    kept = load_kml(archipelago, crop=CROP, scale=0.1)
    dropped = load_kml(archipelago, crop=CROP, scale=0.1, min_area=3.0)
    assert not on_water(kept, 5, 5)
    assert on_water(dropped, 5, 5)
    assert not on_water(dropped, 9, 0)  # the mainland is far larger and stays


def test_simplify_reduces_vertices_without_moving_the_coast_much(tmp_path):
    # A wiggly coastline: 1 m zig-zag over 200 m.
    wiggle = [(80 + (i % 2), y) for i, y in enumerate(range(150, -151, -2))]
    kml = write(tmp_path / "wiggle.kml", [way_pm(wiggle)])
    full = load_kml(kml, crop=CROP)
    simple = load_kml(kml, crop=CROP, simplify=2.0)

    def verts(m):
        return sum(len(p.exterior.coords) for p in m.water.geoms)

    assert verts(simple) < verts(full) / 5
    assert simple.water.symmetric_difference(full.water).area < 200 * 2.0


def test_hand_drawn_maps_are_not_treated_as_coastline(tmp_path):
    # Without natural=coastline tags a single polygon is the water itself.
    kml = tmp_path / "pond.kml"
    kml.write_text('<?xml version="1.0" encoding="UTF-8"?>'
                   '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
                   '<name>Pond</name><Polygon><outerBoundaryIs><LinearRing><coordinates>'
                   + ll(square(0, 0, 20)) +
                   '</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>'
                   '</Document></kml>')
    m = load_kml(kml)
    assert on_water(m, 0, 0)


# --- the catalog -------------------------------------------------------------

@pytest.fixture
def maps_dir(tmp_path, archipelago):
    (tmp_path / "catalog.yaml").write_text(
        "arch:\n"
        "  file: arch.kml\n"
        f"  crop: [{CROP[0]}, {CROP[1]}, {CROP[2]}, {CROP[3]}]\n"
        "  scale: 0.5\n")
    return tmp_path


def test_catalog_name_applies_its_options(maps_dir):
    path, opts = resolve_map("arch", maps_dir)
    assert path == maps_dir / "arch.kml"
    assert opts["scale"] == 0.5 and len(opts["crop"]) == 4
    m = load_map("arch", maps_dir)
    assert m.name == "arch" and m.scale == 0.5


def test_file_name_picks_up_its_catalog_entry(maps_dir):
    _, opts = resolve_map("arch.kml", maps_dir)
    assert opts["scale"] == 0.5


def test_any_path_loads_with_defaults(maps_dir, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    copy = other / "arch.kml"
    copy.write_text((maps_dir / "arch.kml").read_text())
    _, opts = resolve_map(str(copy), maps_dir)
    assert opts == {}


def test_unknown_map_lists_the_known_ones(maps_dir):
    with pytest.raises(FileNotFoundError, match="arch"):
        resolve_map("atlantis", maps_dir)


# --- the real maps -------------------------------------------------------------

def test_catalog_lists_every_map():
    names = set(load_catalog())
    assert {"landgraben", "openwater", "test", "kornaten", "zirje_kaprije"} <= names


@pytest.mark.parametrize("name, sea, land", [
    ("kornaten", (-200, -100), (-40, 90)),        # open sea; on Kornat
    ("zirje_kaprije", (-170, -10), (-60, -100)),  # open sea; on Zirje
])
def test_real_coastline_maps_load(name, sea, land):
    m = load_map(name)
    assert m.water.contains(Point(sea))
    assert not m.water.contains(Point(land))
    assert len([f for f in m.features if not f.is_water]) > 20
