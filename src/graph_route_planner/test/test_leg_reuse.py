"""Local replans plan the boat's own leg only; the wind replans the whole route.

Drives the node's methods on stand-ins carrying what they read, with a real
CachedPlanner on open water.
"""

import types
from datetime import datetime

import pytest
from shapely.geometry import box

from graph_route_planner.node import WIND_MADE_LEG_UNSAILABLE, GraphRoutePlanner
from graph_route_planner.planner import CachedPlanner, PlannerConfig
from graph_route_planner.sailing import SailingModel

WATER = box(-300, -300, 300, 300)
VIA = [(0.0, 100.0), (150.0, 150.0), (150.0, -100.0)]
MODEL = SailingModel.from_bearing(0.0, no_go_deg=50.0)


@pytest.fixture(scope="module")
def planner():
    return CachedPlanner(WATER, PlannerConfig(margin=2.0, grid_spacing=25.0,
                                              merge_threshold=1.5, segment_cost=1.0))


def stand_in(planner, **extra):
    node = types.SimpleNamespace(planner=planner, leg_cache={}, logged=[], **extra)
    node.get_logger = lambda: types.SimpleNamespace(
        info=node.logged.append, warn=node.logged.append, error=node.logged.append)
    return node


def plan(node, boat=(0.0, -100.0)):
    return GraphRoutePlanner._plan_through(node, [boat] + VIA, MODEL)


# ── _plan_through ─────────────────────────────────────────────────────────────

def test_first_plan_plans_every_leg_and_keeps_the_later_ones(planner):
    node = stand_in(planner)
    route, soft, reused = plan(node)
    assert reused == 0
    assert set(node.leg_cache) == {(VIA[0], VIA[1]), (VIA[1], VIA[2])}   # not the boat's leg
    assert [p for p, s in zip(route, soft) if not s] == VIA               # via points are hard


def test_a_second_plan_reuses_the_later_legs(planner):
    node = stand_in(planner)
    first, _, _ = plan(node)
    second, soft, reused = plan(node, boat=(5.0, -95.0))
    assert reused == 2
    # Everything after the first via point is the same route as before.
    i, j = first.index(VIA[0]), second.index(VIA[0])
    assert second[j:] == first[i:]
    assert [p for p, s in zip(second, soft) if not s] == VIA


def test_reused_legs_come_from_the_cache_not_a_new_plan(planner):
    node = stand_in(planner)
    plan(node)
    detour = [VIA[1], (100.0, 0.0), VIA[2]]                  # a leg no planner would make
    node.leg_cache[(VIA[1], VIA[2])] = detour
    route, soft, _ = plan(node)
    assert route[-3:] == detour
    assert soft[-2:] == [True, False]        # the detour's corner is soft, the via point hard


# ── _wind_shift_reason ────────────────────────────────────────────────────────

@pytest.mark.parametrize("planned, now, fires", [
    (0.0, 9.0, False), (0.0, 10.0, True), (355.0, 3.0, False), (355.0, 6.0, True),
    (180.0, 165.0, True), (None, 50.0, False),
])
def test_wind_shift_reason(planned, now, fires):
    node = types.SimpleNamespace(planned_wind=planned, wind_direction=now,
                                 config=types.SimpleNamespace(wind_shift_replan=10.0))
    assert (GraphRoutePlanner._wind_shift_reason(node) is not None) == fires


# ── which replans are local ──────────────────────────────────────────────────

def timer_stand_in(replan_reason, wind_now=0.0, planned_wind=0.0):
    calls = []
    node = types.SimpleNamespace(
        mission=VIA, mission_complete=False, boat=(0.0, 0.0), wind_direction=wind_now,
        planned_route=[(0.0, 0.0), VIA[0]], planned_wind=planned_wind,
        last_plan_time=datetime.min,
        config=types.SimpleNamespace(min_time_between_replanning=5.0, wind_shift_replan=10.0),
    )
    node._replan_reason = lambda: replan_reason
    node._wind_shift_reason = lambda: GraphRoutePlanner._wind_shift_reason(node)
    node._plan_and_publish = lambda reason, whole_route=True: calls.append((reason, whole_route))
    GraphRoutePlanner._plan_timer_callback(node)
    return calls


@pytest.mark.parametrize("reason", [
    "off course by 11.0 m", "ran aground", "1.2 m inside the 3.0 m shore margin",
    "shorter route from here: 2.7 instead of 9.5 to the next via point",
])
def test_local_triggers_replan_the_boats_leg_only(reason):
    assert timer_stand_in(reason) == [(reason, False)]


def test_an_unsailable_leg_replans_the_whole_route():
    assert timer_stand_in(WIND_MADE_LEG_UNSAILABLE) == [(WIND_MADE_LEG_UNSAILABLE, True)]


def test_a_wind_shift_replans_the_whole_route_even_without_another_trigger():
    calls = timer_stand_in(None, wind_now=20.0, planned_wind=0.0)
    assert len(calls) == 1 and calls[0][1] is True and calls[0][0].startswith("wind shifted")


def test_nothing_to_do_nothing_planned():
    assert timer_stand_in(None) == []
