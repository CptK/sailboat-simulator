"""Replan trigger 4: a fresh plan from the boat reaches the next via point much sooner.

Drives `_shortcut_reason` on a stand-in carrying what it reads, with a real
CachedPlanner on open water: a dead-upwind beat the planner lays as one tack.
"""

import math
import types

import pytest
from shapely.geometry import Point, box
from shapely.prepared import prep

from graph_route_planner.node import GraphRoutePlanner
from graph_route_planner.planner import CachedPlanner, PlannerConfig
from graph_route_planner.sailing import SailingModel

WATER = box(-200, -200, 200, 200)
START, GOAL = (0.0, -150.0), (0.0, 150.0)      # wind from the north: dead upwind


@pytest.fixture(scope="module")
def planner():
    return CachedPlanner(WATER, PlannerConfig(margin=2.0, grid_spacing=20.0,
                                              merge_threshold=1.5, segment_cost=1.0))


@pytest.fixture(scope="module")
def model():
    return SailingModel.from_bearing(0.0, no_go_deg=50.0)


@pytest.fixture(scope="module")
def beat(planner, model):
    route = planner.plan(START, GOAL, model).waypoints
    assert len(route) == 3, "expected the planner to lay this beat as one tack"
    return route


def node(planner, beat, boat, route_index=1, fraction=0.15, saving=3.0):
    config = types.SimpleNamespace(dist_threshold=5.0, segment_cost=1.0,
                                   shortcut_min_fraction=fraction, shortcut_min_saving=saving)
    return types.SimpleNamespace(
        planned_route=beat, planned_soft=[True] + [True] * (len(beat) - 2) + [False],
        boat=boat, boat_heading=0.0, route_index=route_index, config=config,
        planner=planner, safe_water_prepared=prep(WATER.buffer(-2.0)),
    )


def reason(n, model):
    return GraphRoutePlanner._shortcut_reason(n, model)


def along(a, b, fraction):
    return (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))


def test_on_the_planned_line_nothing_to_gain(planner, model, beat):
    for f in (0.1, 0.5, 0.9):
        assert reason(node(planner, beat, along(beat[0], beat[1], f)), model) is None


def test_a_little_off_the_line_is_not_worth_a_replan(planner, model, beat):
    boat = along(beat[0], beat[1], 0.5)
    for dx, dy in [(3, 0), (-3, 0), (0, 3), (0, -3)]:
        assert reason(node(planner, beat, (boat[0] + dx, boat[1] + dy)), model) is None


def test_past_the_corner_it_is_still_steering_at_replans(planner, model, beat):
    """The case this trigger is for: through the wind already, upwind of the
    tacking corner — which the route would sail back down to."""
    corner = beat[1]
    boat = (corner[0] - 30.0, corner[1] + 38.0)
    text = reason(node(planner, beat, boat), model)
    assert text is not None and text.startswith("shorter route from here")


def test_thresholds_set_what_counts(planner, model, beat):
    corner = beat[1]
    boat = (corner[0] - 30.0, corner[1] + 38.0)
    assert reason(node(planner, beat, boat, fraction=0.95), model) is None
    assert reason(node(planner, beat, boat, saving=1e6), model) is None


def test_skipped_inside_the_shore_margin(planner, model, beat):
    corner = beat[1]
    n = node(planner, beat, (corner[0] - 30.0, corner[1] + 38.0))
    n.safe_water_prepared = prep(box(-1, -1, 1, 1))     # the boat is "in the margin"
    assert reason(n, model) is None


def test_skipped_at_the_via_point(planner, model, beat):
    assert reason(node(planner, beat, (GOAL[0], GOAL[1] - 2.0), route_index=2), model) is None
