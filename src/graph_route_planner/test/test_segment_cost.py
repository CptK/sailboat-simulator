"""A fixed cost per segment, so fewer legs win near-ties.

Sailing time alone prices an 8 m repositioning hop at 8 m of sailing and nothing
else, so the search will take one to gain a fraction of a percent. Every segment
boundary is a turn, and a turn is where the boat stops making progress while it
settles — a cost the model has no other way to express.
"""

import math

import pytest

from graph_route_planner import MAPS_DIR
from graph_route_planner.map_loader import load_kml
from graph_route_planner.planner import CachedPlanner, PlannerConfig, plan_route
from graph_route_planner.sailing import SailingModel

BEAT_START, BEAT_GOAL = (60.0, -10.0), (60.0, 60.0)
MISSION = [(-10.0, 60.0), (-10.0, -10.0), (60.0, -10.0), (60.0, 60.0), (-10.0, 60.0)]


@pytest.fixture(scope="module")
def water():
    return load_kml(str(MAPS_DIR / "OpenWater.kml")).water


@pytest.fixture(scope="module")
def model():
    return SailingModel.from_bearing(0.1, no_go_deg=50)


def config(segment_cost):
    return PlannerConfig(margin=2.0, grid_spacing=10.0, merge_threshold=1.5,
                         min_leg_distance=0.0, segment_cost=segment_cost)


def segments(route):
    return [math.dist(route.waypoints[i], route.waypoints[i + 1])
            for i in range(len(route.waypoints) - 1)]


def test_without_the_cost_the_search_takes_a_short_repositioning_hop(water, model):
    """The behaviour the cost exists to remove: an 8 m stub worth 0.2% of time."""
    route = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(0.0))
    assert min(segments(route)) < 10.0


def test_the_cost_removes_the_stub(water, model):
    route = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(1.0))
    assert min(segments(route)) > 30.0


def test_the_cost_prefers_fewer_legs(water, model):
    cheap = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(0.0))
    dear = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(1.0))
    assert len(dear.waypoints) < len(cheap.waypoints)


def test_it_cannot_collapse_a_beat_below_two_legs(water, model):
    """No cost can make an unsailable direct heading sailable."""
    route = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(1000.0))
    assert route.found
    assert len(route.waypoints) >= 3          # start + a tack + goal


def test_duration_reports_true_sailing_time_not_the_penalty(water, model):
    """`duration` is the boat's time. The penalty shapes the search only."""
    route = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(1.0))
    by_hand = sum(model.sailing_time(a, b)
                  for a, b in zip(route.waypoints, route.waypoints[1:]))
    assert route.duration == pytest.approx(by_hand, rel=1e-12)


def test_a_higher_cost_never_makes_the_boat_faster(water, model):
    """Shaping trades time for shape; it must not claim to do both."""
    base = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(0.0))
    for k in (0.5, 1.0, 4.0):
        route = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(k))
        assert route.duration >= base.duration - 1e-9


def test_the_penalty_reaches_the_endpoint_edges_too(water, model):
    """The endpoints are spliced in after the cached edges are costed.

    Penalising only the cached half leaves a route that starts or ends with a
    free stub — which is exactly the bug this test exists to catch.
    """
    cheap = CachedPlanner(water, config(0.0))
    dear = CachedPlanner(water, config(1.0))
    first_cheap = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=config(0.0))
    assert math.dist(*first_cheap.waypoints[:2]) < 10.0     # the free stub is first
    got = dear.plan(BEAT_START, BEAT_GOAL, model)
    assert math.dist(*got.waypoints[:2]) > 30.0


@pytest.mark.parametrize("segment_cost", [0.0, 0.5, 1.0])
def test_cached_planner_still_matches_plan_route(water, model, segment_cost):
    """Both paths must apply the cost identically, or replans would jitter."""
    planner = CachedPlanner(water, config(segment_cost))
    for start, goal in zip(MISSION, MISSION[1:]):
        want = plan_route(water, start, goal, model=model, config=config(segment_cost))
        got = planner.plan(start, goal, model)
        assert got.waypoints == want.waypoints, f"{start} -> {goal} at k={segment_cost}"
        assert got.duration == pytest.approx(want.duration, rel=1e-12)


def test_the_whole_mission_settles_at_the_minimum_leg_count(water, model):
    planner = CachedPlanner(water, config(1.0))
    counts = [len(planner.plan(s, g, model).waypoints) - 1
              for s, g in zip(MISSION, MISSION[1:])]
    assert counts == [1, 1, 2, 1]


def test_a_negative_cost_is_rejected():
    with pytest.raises(ValueError):
        PlannerConfig(segment_cost=-1.0)


def test_the_configured_default_is_the_tuned_value():
    """The library default matches the node's, so the CLI and viewer show the
    same routes the running planner produces."""
    assert PlannerConfig().segment_cost == 1.0


def test_the_default_actually_reaches_the_search(water, model):
    """A default that never reaches an edge would be a silent no-op."""
    default = PlannerConfig(margin=2.0, grid_spacing=10.0, merge_threshold=1.5,
                            min_leg_distance=0.0)
    route = plan_route(water, BEAT_START, BEAT_GOAL, model=model, config=default)
    assert min(segments(route)) > 30.0


def test_the_graph_primitives_stay_policy_free():
    """`cost_pairs`/`build_graph` default to 0: a primitive should not invent a
    shaping policy. PlannerConfig is where that decision lives, and callers
    reaching past it get pure time-optimal costing — deliberately."""
    import inspect

    from graph_route_planner.graph import build_graph, cost_pairs

    assert inspect.signature(cost_pairs).parameters["segment_cost"].default == 0.0
    assert inspect.signature(build_graph).parameters["segment_cost"].default == 0.0
