"""The replan triggers in graph_route_planner.node.

`_replan_reason` is pure logic over the boat, the route and the wind — it needs
no ROS spinning — so these drive it directly on a stand-in object carrying the
attributes it reads. That keeps the triggers under test without a live graph.
"""

import math
import types

import pytest

from graph_route_planner.node import GraphRoutePlanner
from graph_route_planner.sailing import SailingModel

WIND = 0.0          # from due north, so a bearing of B degrees is B degrees off the wind
TACK_LIMIT = 50.0


def make(route, boat, index, *, margin=5.0, off_course=10.0, dist=5.0, wind=WIND):
    """A stand-in carrying exactly what `_replan_reason` reads."""
    config = types.SimpleNamespace(
        dist_threshold=dist,
        off_course_threshold=off_course,
        min_tack_angle=TACK_LIMIT,
        no_go_replan_margin=margin,
    )
    node = types.SimpleNamespace(planned_route=route, boat=boat, route_index=index,
                                 config=config)
    node._sailing_model = lambda: SailingModel.from_bearing(wind, no_go_deg=TACK_LIMIT)
    return node


def target_at(boat, angle_deg, distance=30.0):
    """A point `distance` away from `boat`, `angle_deg` off a northerly wind."""
    r = math.radians(angle_deg)
    return (boat[0] + distance * math.sin(r), boat[1] + distance * math.cos(r))


def no_go_case(angle_deg, distance=30.0, **kwargs):
    """A scenario isolating trigger 1: the boat cannot lay its waypoint.

    The route leg runs due east, so it is always sailable and trigger 3 stays
    quiet. The boat sits `angle_deg` downwind of the waypoint, which puts it far
    from that leg — so `off_course` is disabled too. Only the live bearing from
    the boat to its target is left to fire.
    """
    target = (0.0, 0.0)
    route = [(-30.0, 0.0), target]
    r = math.radians(angle_deg)
    boat = (-distance * math.sin(r), -distance * math.cos(r))
    kwargs.setdefault("off_course", 1e6)
    return make(route, boat, 1, **kwargs)


def reason(node):
    return GraphRoutePlanner._replan_reason(node)


# ── trigger 1: the boat can no longer lay its waypoint ────────────────────────

@pytest.mark.parametrize("angle", [50.0, 49.0, 48.0, 46.0, 45.5])
def test_bearings_within_the_margin_do_not_replan(angle):
    """The planner lays upwind legs at exactly the tack limit.

    Without hysteresis the live bearing sits on the trigger for the whole beat
    and a fraction of a degree of leeway replans, several times a minute.
    """
    assert reason(no_go_case(angle)) is None


@pytest.mark.parametrize("angle", [44.9, 40.0, 31.0, 10.0, 0.0])
def test_bearings_clearly_inside_the_no_go_zone_replan(angle):
    got = reason(no_go_case(angle))
    assert got is not None and "no-go" in got


def test_the_margin_is_configurable():
    """A margin of 0 restores the old, chattering behaviour."""
    assert reason(no_go_case(49.0, margin=0.0)) is not None
    assert reason(no_go_case(49.0, margin=5.0)) is None


def test_a_waypoint_almost_reached_never_trips_the_no_go_trigger():
    """Within dist_threshold the bearing is dominated by noise, not by leeway.

    The boat sits 2 m from a waypoint that is dead upwind of it — the worst
    possible bearing — on a leg that is itself sailable.
    """
    route = [(-30.0, 2.0), (0.0, 2.0)]      # due east: sailable
    boat = (0.0, 0.0)                        # 2 m short, waypoint bears due north
    assert math.dist(boat, route[1]) < 5.0
    assert reason(make(route, boat, 1)) is None


def test_the_observed_run_is_mostly_suppressed():
    """The seven replans from the 10:44 run: six were noise, one was real."""
    observed = [47, 49, 31, 49, 50, 49, 49]
    fired = [a for a in observed if reason(no_go_case(float(a))) is not None]
    assert fired == [31]


# ── trigger 2: off course ─────────────────────────────────────────────────────

def test_drift_beyond_the_threshold_replans():
    route = [(0.0, 0.0), (100.0, 0.0)]      # due east: sailable
    got = reason(make(route, (50.0, 12.0), 1))
    assert got is not None and "off course" in got


def test_drift_within_the_threshold_does_not_replan():
    route = [(0.0, 0.0), (100.0, 0.0)]
    assert reason(make(route, (50.0, 8.0), 1)) is None


def test_drift_is_not_masked_by_legs_already_sailed():
    """A closed course returns to its start; the last leg must not hide behind the first."""
    loop = [(-10.0, 60.0), (-10.0, -10.0), (60.0, -10.0), (60.0, 60.0), (-10.0, 60.0)]
    boat = (-10.0, 45.0)          # exactly on leg 1, but 15 m off the final leg
    from shapely.geometry import LineString, Point
    assert LineString(loop).distance(Point(boat)) < 10.0, "setup: the old check would miss this"
    assert reason(make(loop, boat, 4)) is not None


# ── trigger 3: a wind shift made a remaining leg unsailable ───────────────────

def test_a_leg_ahead_that_cannot_be_sailed_replans():
    route = [(0.0, 0.0), (0.0, 30.0), (0.0, 60.0)]     # straight upwind
    got = reason(make(route, (0.6, 1.0), 0, wind=0.0))
    assert got is not None


def test_legs_already_behind_the_boat_are_ignored():
    """An unsailable leg the boat has finished is not a reason to replan."""
    route = [(0.0, 0.0), (0.0, 30.0), (30.0, 30.0), (60.0, 30.0)]
    model = SailingModel.from_bearing(WIND, no_go_deg=TACK_LIMIT)
    assert not model.heading_is_sailable(route[0], route[1]), "setup: leg 0 is dead upwind"

    # route_index 2 means the boat is sailing route[1] -> route[2]; leg 0 is done.
    assert reason(make(route, (30.0, 30.0), 2)) is None


# ── route progress ────────────────────────────────────────────────────────────

def test_route_index_follows_the_boat_along_the_route():
    route = [(60.0, -10.0), (104.0, 26.0), (60.0, 60.0)]
    node = make(route, (60.0, -10.0), 0)
    steps = []
    for position in [(60.0, -10.0), (80.0, 6.0), (104.0, 26.0), (80.0, 45.0), (60.0, 60.0)]:
        node.boat = position
        GraphRoutePlanner._advance_route_index(node)
        steps.append(node.route_index)
    assert steps == [1, 1, 2, 2, 2]


def test_route_index_never_rewinds():
    """A boat blown back past a waypoint is off course, not un-progressed."""
    route = [(0.0, 0.0), (0.0, 30.0), (0.0, 60.0)]
    node = make(route, (0.0, 30.0), 0)
    GraphRoutePlanner._advance_route_index(node)
    advanced = node.route_index
    node.boat = (0.0, -20.0)
    GraphRoutePlanner._advance_route_index(node)
    assert node.route_index == advanced


def test_a_short_route_is_never_replanned():
    assert reason(make([(0.0, 0.0)], (0.0, 0.0), 0)) is None
    assert reason(make(None, (0.0, 0.0), 0)) is None
