"""Steering: how route_follower advances along the route it was given."""

import pytest

from route_follower.node import RouteFollower
from route_follower.utils import bearing

from conftest import RouteMsgStub, step, wp


# A beat to windward: two synthetic tacking points between the boat and the mark.
START = wp(60.0, -10.0)
TACK1 = wp(90.0, 12.0, is_soft=True)
TACK2 = wp(35.0, 40.0, is_soft=True)
MARK = wp(60.0, 60.0)
BEAT = [START, TACK1, TACK2, MARK]


# ── reaching waypoints in order ───────────────────────────────────────────────

def test_advances_when_the_current_waypoint_is_reached(follower):
    node = follower(route=BEAT, target_index=1, boat=wp(90.0, 12.0))
    step(node)
    assert node.target_index == 2


def test_does_not_advance_while_short_of_the_waypoint(follower):
    node = follower(route=BEAT, target_index=1, boat=wp(80.0, 5.0))
    step(node)
    assert node.target_index == 1


def test_steers_at_the_current_waypoint(follower):
    boat = wp(60.0, -10.0)
    node = follower(route=BEAT, target_index=1, boat=boat, heading=bearing(boat, TACK1))
    assert step(node) == pytest.approx(bearing(boat, TACK1))


def test_completes_on_the_final_waypoint(follower):
    node = follower(route=BEAT, target_index=3, boat=wp(60.0, 60.0))
    step(node)
    assert node.completed is True
    assert node.published_completed == [True]


def test_stays_completed_and_does_not_advance_past_the_end(follower):
    node = follower(route=BEAT, target_index=3, boat=wp(60.0, 60.0))
    step(node)
    assert node.target_index == 3


# ── skipping soft waypoints once the mark is in reach ─────────────────────────

def test_skips_soft_waypoints_when_already_at_the_hard_one(follower):
    """The boat is on the mark but steering at a tacking point 56 m behind it."""
    node = follower(route=BEAT, target_index=1, boat=wp(60.0, 60.0))
    step(node)
    assert node.route[node.target_index] == MARK or node.completed


def test_skipping_completes_the_route_when_the_mark_is_the_last_waypoint(follower):
    node = follower(route=BEAT, target_index=1, boat=wp(60.0, 60.0))
    step(node)
    step(node)
    assert node.completed is True


def test_skips_only_up_to_the_next_hard_waypoint(follower):
    """Soft points beyond the mark belong to the next leg and must survive."""
    beyond = wp(20.0, 80.0, is_soft=True)
    route = [START, TACK1, TACK2, MARK, beyond, wp(-10.0, 60.0)]
    node = follower(route=route, target_index=1, boat=wp(60.0, 60.0))
    step(node)
    assert beyond in node.route


def test_a_hard_waypoint_in_the_way_is_never_skipped(follower):
    """Being near a later mark must not let the boat cut the one before it."""
    gate = wp(30.0, 20.0)                       # hard, must be rounded
    route = [START, TACK1, gate, TACK2, MARK]
    node = follower(route=route, target_index=1, boat=wp(60.0, 60.0))
    step(node)
    assert gate in node.route
    assert node.route[node.target_index] != MARK


def test_no_skip_while_the_mark_is_still_far(follower):
    node = follower(route=BEAT, target_index=1, boat=wp(85.0, 8.0))
    step(node)
    assert node.route[node.target_index] == TACK1


def test_a_route_of_only_hard_waypoints_is_unaffected(follower):
    route = [START, wp(20.0, 10.0), MARK]
    node = follower(route=route, target_index=1, boat=wp(60.0, 60.0))
    step(node)
    assert node.route[node.target_index] == wp(20.0, 10.0)


# ── accepting routes ──────────────────────────────────────────────────────────

def test_a_new_route_replaces_the_old_one(follower):
    node = follower(route=BEAT, target_index=2, boat=wp(60.0, -10.0))
    RouteFollower._on_target_route(node, RouteMsgStub([(0.0, 0.0), (10.0, 10.0)]))
    assert len(node.route) == 2
    assert node.target_index == 0
    assert node.completed is False


def test_is_soft_survives_the_wire(follower):
    node = follower(boat=wp(0.0, 0.0))
    RouteFollower._on_target_route(
        node, RouteMsgStub([(0.0, 0.0, False), (5.0, 5.0, True), (9.0, 9.0, False)])
    )
    assert [w.is_soft for w in node.route] == [False, True, False]


def test_a_maneuver_survives_a_replan_onto_the_same_leg(follower):
    sentinel = object()
    node = follower(route=BEAT, target_index=1, boat=wp(80.0, 5.0), maneuver=sentinel)
    RouteFollower._on_target_route(node, RouteMsgStub([(80.0, 5.0), (90.0, 12.0), (60.0, 60.0)]))
    assert node.maneuver is sentinel


def test_a_maneuver_is_cancelled_when_the_leg_changes(follower):
    sentinel = object()
    node = follower(route=BEAT, target_index=1, boat=wp(80.0, 5.0), maneuver=sentinel)
    RouteFollower._on_target_route(node, RouteMsgStub([(80.0, 5.0), (10.0, 40.0), (60.0, 60.0)]))
    assert node.maneuver is None


def test_a_maneuver_survives_the_first_step_after_a_replan(follower):
    """Regression: the route starts at the boat, so the first step used to
    "reach" waypoint 0, treat that as a fresh leg and cancel the maneuver.

    A replan every few seconds then restarted the tack before `is_stuck` could
    ever count enough steps to fall back to a jibe, and a grounded boat sat
    pinched on the shore indefinitely.
    """
    boat = wp(80.0, 5.0)
    node = follower(route=BEAT, target_index=1, boat=boat, heading=180.0)
    step(node)
    maneuver = node.maneuver
    assert maneuver is not None          # a 125° turn onto TACK1: too far to just steer

    RouteFollower._on_target_route(node, RouteMsgStub([(80.0, 5.0), (90.0, 12.0), (60.0, 60.0)]))
    step(node)
    assert node.maneuver is maneuver


# ── a heading command of 0 degrees is a heading ───────────────────────────────

def test_jibe_steering_due_north_is_not_dropped(follower):
    """In a southerly, dead downwind is 0 degrees, which is where a jibe the
    long way round steers first. A truthiness check once turned that 0 into
    "no command" and steered at the target instead — the short way, through
    the wind: a slow accidental tack."""
    from route_follower.maneuver_action import JibeManeuver

    jibe = JibeManeuver(target_heading=244.0, current_heading=128.0, heading_step_size=10.0,
                        wind_direction=180.0, command_step_size_multiplier=1.5,
                        heading_tolerance=2.0)
    far_wsw = wp(-100.0 * 0.899, -100.0 * 0.438)          # bearing about 244 degrees
    node = follower(route=[wp(0.0, 0.0), far_wsw], target_index=1, boat=wp(0.0, 0.0),
                    heading=128.0, wind_direction=180.0, maneuver=jibe)
    assert step(node) == 0.0
