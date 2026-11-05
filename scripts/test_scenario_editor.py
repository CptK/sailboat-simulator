"""Tests for the scenario editor, driven headlessly with synthetic events.

    pytest scripts/test_scenario_editor.py
"""

import shutil
import sys
import types
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import SCENARIOS_DIR, load_scenario  # noqa: E402
from scenario_editor import ScenarioEditor, open_scenario  # noqa: E402

# On the Landgraben pond: open water, a point on the island, and the start.
WATER = (-20.0, -35.0)
ISLAND = (0.0, 10.0)


@pytest.fixture
def editor(tmp_path):
    path = tmp_path / "landgraben.yaml"
    shutil.copy(SCENARIOS_DIR / "landgraben.yaml", path)
    scenario, sail_map, path = open_scenario(str(path), None)
    ed = ScenarioEditor(scenario, sail_map, path)
    yield ed
    plt.close(ed.fig)


def event(ed, x, y, button=1, key=None):
    return types.SimpleNamespace(inaxes=ed.ax, xdata=x, ydata=y, button=button, key=key)


def click(ed, x, y, **kw):
    ed._on_press(event(ed, x, y, **kw))
    ed._on_release(event(ed, x, y, **kw))


def drag(ed, start, end):
    ed._on_press(event(ed, *start))
    ed._on_motion(event(ed, *end))
    ed._on_release(event(ed, *end))


def test_click_appends_a_waypoint(editor):
    n = len(editor.sc.waypoints)
    click(editor, *WATER)
    assert editor.sc.waypoints[-1] == WATER and len(editor.sc.waypoints) == n + 1
    assert editor.dirty


def test_boat_mode_moves_the_start(editor):
    editor.set_mode("boat")
    click(editor, *WATER)
    assert editor.sc.boat == WATER


def test_buoy_mode_places_the_chosen_colour(editor):
    editor.set_color("blue")
    click(editor, *WATER)
    buoy = editor.sc.buoys[-1]
    assert (buoy.east, buoy.north, buoy.color) == (*WATER, "blue")


def test_buoy_can_be_placed_right_next_to_a_waypoint(editor):
    """A click near a marker of another kind places, it does not grab."""
    wx, wy = editor.sc.waypoints[0]
    editor.set_color("green")
    n = len(editor.sc.buoys)
    click(editor, wx + 0.3, wy + 0.3)
    assert len(editor.sc.buoys) == n + 1
    assert editor.sc.waypoints[0] == (wx, wy)


def test_drag_moves_and_undo_restores(editor):
    original = editor.sc.waypoints[0]
    drag(editor, original, WATER)
    assert editor.sc.waypoints[0] == WATER
    editor.undo()
    assert editor.sc.waypoints[0] == original


def test_any_marker_can_be_dragged_whatever_is_being_placed(editor):
    editor.set_mode("waypoint")
    drag(editor, editor.sc.boat, WATER)
    assert editor.sc.boat == WATER


def test_right_click_deletes(editor):
    n = len(editor.sc.waypoints)
    second = editor.sc.waypoints[1]
    click(editor, *second, button=3)
    assert len(editor.sc.waypoints) == n - 1 and second not in editor.sc.waypoints


def test_delete_key_deletes_under_the_cursor(editor):
    b = editor.sc.buoys[0]
    editor._on_key(types.SimpleNamespace(key="d", inaxes=editor.ax, xdata=b.east,
                                         ydata=b.north))
    assert b not in editor.sc.buoys


def test_the_boat_cannot_be_deleted(editor):
    boat = editor.sc.boat
    click(editor, *boat, button=3)
    assert editor.sc.boat == boat


def test_shift_click_inserts_into_the_nearest_leg(editor):
    # Between waypoint 4 (-32, 10) and waypoint 5 (0, -35).
    click(editor, -16.0, -12.0, key="shift")
    assert editor.sc.waypoints[4] == (-16.0, -12.0)
    assert len(editor.sc.waypoints) == 6


def test_a_waypoint_on_land_is_flagged_and_stops_the_route(editor):
    editor.w_route.set_active(0)
    assert editor.route and not editor.route_problem
    click(editor, *ISLAND)
    assert "On land" in editor.status.get_text()
    assert editor.route == []


def test_a_buoy_on_land_does_not_hide_the_route(editor):
    editor.w_route.set_active(0)
    editor.set_color("red")
    click(editor, *ISLAND)
    assert "On land" in editor.status.get_text()
    assert editor.route


def test_typing_a_description_does_not_trigger_shortcuts(editor):
    n = len(editor.sc.waypoints)
    editor.w_desc.capturekeystrokes = True
    wx, wy = editor.sc.waypoints[0]
    editor._on_key(types.SimpleNamespace(key="d", inaxes=editor.ax, xdata=wx, ydata=wy))
    assert len(editor.sc.waypoints) == n


def test_save_writes_a_scenario_that_loads_back(editor):
    click(editor, *WATER)
    editor.w_wind_dir.set_val(90)
    editor.w_desc.set_val("Edited in the editor.")
    assert editor.save()
    saved = load_scenario(str(editor.path))
    assert saved.waypoints[-1] == WATER
    assert saved.wind_direction == 90
    assert saved.description == "Edited in the editor."
    assert not editor.dirty


def test_new_scenario_needs_a_map_and_a_waypoint_to_save(tmp_path):
    with pytest.raises(SystemExit, match="--map"):
        open_scenario(str(tmp_path / "fresh.yaml"), None)
    scenario, sail_map, path = open_scenario(str(tmp_path / "fresh.yaml"), "landgraben")
    assert sail_map.water.contains(__import__("shapely").geometry.Point(scenario.boat))
    ed = ScenarioEditor(scenario, sail_map, path)
    try:
        assert not ed.save()                      # no waypoints yet
        click(ed, *WATER)
        assert ed.save() and path.is_file()
    finally:
        plt.close(ed.fig)


def test_existing_scenario_is_not_replaced_by_a_new_one():
    with pytest.raises(SystemExit, match="already exists"):
        open_scenario("landgraben", "kornaten")


def test_resizing_the_window_does_not_raise(editor, capsys):
    """matplotlib 3.11's TextBox raised on every resize event (no .inaxes)."""
    from matplotlib.backend_bases import ResizeEvent

    editor.w_desc.begin_typing()
    ResizeEvent("resize_event", editor.fig.canvas)._process()
    assert "Traceback" not in capsys.readouterr().err
    assert not editor.w_desc.capturekeystrokes     # resizing still stops typing

