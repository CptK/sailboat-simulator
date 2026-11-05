"""Tests for scenario files (scenarios/*.yaml) and their loader.

    pytest scripts/test_scenario.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario import (  # noqa: E402
    COLORS, DEFAULT_GRID_SPACING, DEFAULT_PLANNING_DELAY, ScenarioError,
    check_positions, list_scenarios, load_scenario)

MINIMAL = """
map: landgraben
boat: {east: 0, north: -76}
wind: {direction: 180, speed: 4}
waypoints:
  - [54, -24]
"""


def write(tmp_path, text, name="s.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# --- every shipped scenario ----------------------------------------------------

@pytest.mark.parametrize("name", list_scenarios())
def test_shipped_scenario_loads_and_is_on_water(name):
    scenario = load_scenario(name)
    errors, _ = check_positions(scenario)
    assert errors == []


def test_the_default_scenario_exists():
    from run_simulation_graph import DEFAULT_SCENARIO
    assert DEFAULT_SCENARIO in list_scenarios()


# --- the format -------------------------------------------------------------------

def test_minimal_scenario_gets_defaults(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL))
    assert s.name == "s" and s.map == "landgraben"
    assert s.boat == (0.0, -76.0)
    assert (s.wind_direction, s.wind_speed) == (180.0, 4.0)
    assert s.waypoints == [(54.0, -24.0)]
    assert s.buoys == []
    assert s.grid_spacing == DEFAULT_GRID_SPACING
    assert s.max_leg_distance == 0.0
    assert s.planning_delay == DEFAULT_PLANNING_DELAY


def test_positions_as_lists_or_mappings(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL.replace(
        "  - [54, -24]", "  - [54, -24]\n  - {east: 54, north: 16}")))
    assert s.waypoints == [(54.0, -24.0), (54.0, 16.0)]


def test_buoy_colours_by_name_or_rgb(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL + """
buoys:
  - {east: 1, north: 2, color: green}
  - {east: 3, north: 4, color: [1, 0, 0]}
  - {east: 5, north: 6, color: [0.2, 0.3, 0.4]}
  - {east: 7, north: 8}
"""))
    green, red, custom, default = s.buoys
    assert green.rgb == COLORS["green"] and green.color == "green"
    assert red.rgb == (1.0, 0.0, 0.0) and red.color == "red"    # RGB that has a name
    assert custom.rgb == (0.2, 0.3, 0.4) and custom.color is None
    assert default.color == "orange"


def test_planner_settings(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL + """
planner:
  grid_spacing: 15
  max_leg_distance: 120
  planning_delay: 20
"""))
    assert (s.grid_spacing, s.max_leg_distance, s.planning_delay) == (15.0, 120.0, 20.0)


@pytest.mark.parametrize("change, message", [
    (("waypoints:", "waypoint:"), "unknown key"),
    (("map: landgraben\n", ""), "missing required key"),
    (("speed: 4", "speed: 40"), "wind speed"),
    (("direction: 180", "direction: north"), "wind direction"),
    (("[54, -24]", "[27]"), "waypoint 1"),
    (("north: -76}", "north: -76, up: 2}"), "boat"),
])
def test_mistakes_are_reported_with_the_file(tmp_path, change, message):
    path = write(tmp_path, MINIMAL.replace(*change))
    with pytest.raises(ScenarioError, match=message) as err:
        load_scenario(path)
    assert path in str(err.value)


def test_unknown_colour_lists_the_known_ones(tmp_path):
    path = write(tmp_path, MINIMAL + "buoys:\n  - {east: 1, north: 2, color: purple}\n")
    with pytest.raises(ScenarioError, match="orange"):
        load_scenario(path)


def test_video_defaults(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL))
    assert (s.video_speedup, s.video_fps, s.video_enabled) == (4.0, 30.0, True)


def test_video_settings(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL + """
video:
  speedup: 2.5
  fps: 24
  enabled: false
"""))
    assert (s.video_speedup, s.video_fps, s.video_enabled) == (2.5, 24.0, False)


@pytest.mark.parametrize("video, message", [
    ("video:\n  speed: 4\n", "speedup"),          # unknown key lists the allowed ones
    ("video:\n  speedup: 0\n", "positive"),
    ("video:\n  fps: 500\n", "video fps"),
    ("video:\n  enabled: maybe\n", "true or false"),
])
def test_video_mistakes(tmp_path, video, message):
    with pytest.raises(ScenarioError, match=message):
        load_scenario(write(tmp_path, MINIMAL + video))


def test_video_settings_survive_saving(tmp_path):
    from scenario import save_scenario
    s = load_scenario(write(tmp_path, MINIMAL + "video:\n  speedup: 2\n  enabled: false\n"))
    again = load_scenario(str(save_scenario(s, tmp_path / "again.yaml")))
    assert (again.video_speedup, again.video_fps, again.video_enabled) == (2.0, 30.0, False)


def test_unknown_planner_setting(tmp_path):
    path = write(tmp_path, MINIMAL + "planner:\n  grid: 5\n")
    with pytest.raises(ScenarioError, match="grid_spacing"):
        load_scenario(path)


def test_unknown_scenario_lists_the_known_ones():
    with pytest.raises(ScenarioError, match="landgraben"):
        load_scenario("no_such_scenario")


# --- positions against the map ---------------------------------------------------

def test_positions_on_land_are_errors(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL.replace("[54, -24]", "[0, 10]") +
                            "buoys:\n  - {east: 5, north: 15, color: red}\n"))
    errors, _ = check_positions(s)
    assert any("waypoint 1" in e and "on land" in e for e in errors)
    assert any("buoy 1" in e for e in errors)


def test_via_point_near_shore_is_a_warning(tmp_path):
    s = load_scenario(write(tmp_path, MINIMAL.replace("[54, -24]", "[62, 16]")))
    errors, warnings = check_positions(s)
    assert errors == []
    assert any("waypoint 1" in w for w in warnings)


def test_point_exactly_on_the_coastline_is_accepted(tmp_path):
    # The corners of OpenWater's course area: exactly on its boundary.
    s = load_scenario(write(tmp_path, """
map: openwater
boat: {east: -10, north: 60}
wind: {direction: 0, speed: 4}
buoys:
  - {east: 0, north: 0}
  - {east: 50, north: 0}
waypoints:
  - [-10, -10]
"""))
    assert check_positions(s)[0] == []


# --- what the run script publishes ----------------------------------------------

def test_environment_message_carries_buoy_colours():
    from run_simulation_graph import build_env_create_message, build_mission_message

    s = load_scenario("landgraben")
    env = build_env_create_message(s)
    assert "boat_east: 0.0, boat_north: -76.0" in env
    assert "{east: 44.0, north: -44.0, color: {r: 1.0, g: 0.0, b: 0.0}}" in env
    assert "wind_direction: 180.0" in env
    mission = build_mission_message(s)
    assert mission.count("east:") == len(s.waypoints)
