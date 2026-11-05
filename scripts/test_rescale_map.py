"""Tests for scripts/rescale_map.py.

    pytest scripts/test_rescale_map.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src" / "graph_route_planner"))

from rescale_map import parse_scale, rescale_catalog, rescale_scenario_text  # noqa: E402

SCENARIO = """description: >
  A course of about 100 x 50 m, east: 10 should not change.
map: somewhere
boat: {east: -30, north: 12.5}
wind: {direction: 315, speed: 4}
buoys:
  - {east: 10, north: -20, color: [1, 0, 0]}   # red
waypoints:
  - [10, -20]     # first mark
  - {east: 3, north: 4}
planner:
  grid_spacing: 15
  max_leg_distance: 120   # large map
  planning_delay: 15
"""

CATALOG = """# header: scale: 0.02 here is a comment
other:
  file: Other.kml
  scale: 0.02

somewhere:
  file: Somewhere.kml
  description: Islands, at 1:50
  crop: [15.1, 43.6, 15.6, 43.9]
  scale: 0.02
  simplify: 0.3
  min_area: 1.0

last:
  file: Last.kml
"""


def test_parse_scale():
    assert parse_scale("1:30") == pytest.approx(1 / 30)
    assert parse_scale("0.05") == 0.05
    with pytest.raises(ValueError):
        parse_scale("0")


def test_scenario_positions_and_planner_distances_scale():
    out = rescale_scenario_text(SCENARIO, 2.0)
    assert "boat: {east: -60, north: 25}" in out
    assert "  - {east: 20, north: -40, color: [1, 0, 0]}   # red" in out
    assert "  - [20, -40]     # first mark" in out
    assert "  - {east: 6, north: 8}" in out
    assert "grid_spacing: 30" in out
    assert "max_leg_distance: 240   # large map" in out


def test_scenario_text_that_is_not_a_position_is_untouched():
    out = rescale_scenario_text(SCENARIO, 2.0)
    assert "about 100 x 50 m, east: 10 should not change." in out    # description
    assert "wind: {direction: 315, speed: 4}" in out
    assert "planning_delay: 15" in out


def test_catalog_entry_rescales_and_keeps_the_rest():
    out, old = rescale_catalog(CATALOG, "somewhere", 1 / 30)
    assert old == 0.02
    entry = out.split("somewhere:")[1].split("last:")[0]
    assert "scale: 0.0333333" in entry
    assert "simplify: 0.5" in entry                 # x ratio
    assert "min_area: 2.77778" in entry             # x ratio squared
    assert "at 1:30" in entry
    assert "crop: [15.1, 43.6, 15.6, 43.9]" in entry
    # Other entries and comments are untouched.
    assert out.startswith("# header: scale: 0.02 here is a comment\nother:\n")
    assert "other:\n  file: Other.kml\n  scale: 0.02\n" in out
    assert out.rstrip().endswith("last:\n  file: Last.kml")


def test_unknown_map_is_refused():
    with pytest.raises(SystemExit):
        rescale_catalog(CATALOG, "nowhere", 0.1)


def test_rescaled_map_is_the_old_map_stretched():
    """Scale, simplify and min_area together give exactly the same islands."""
    from graph_route_planner.map_loader import load_kml, resolve_map

    kml, opts = resolve_map("zirje_kaprije")
    ratio = 1.5
    bigger = dict(opts, scale=opts["scale"] * ratio, simplify=opts["simplify"] * ratio,
                  min_area=opts["min_area"] * ratio ** 2)
    a, b = load_kml(kml, **opts), load_kml(kml, **bigger)
    assert len(b.features) == len(a.features)
    assert b.water.area == pytest.approx(a.water.area * ratio ** 2, rel=1e-6)


def test_dry_run_changes_nothing():
    repo = HERE.parent
    before = {p: p.read_text() for p in [repo / "maps" / "catalog.yaml",
                                         *sorted((repo / "scenarios").glob("*.yaml"))]}
    result = subprocess.run([sys.executable, str(HERE / "rescale_map.py"), "zirje_kaprije",
                             "1:25", "--dry-run"], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry run" in result.stdout
    assert {p: p.read_text() for p in before} == before
