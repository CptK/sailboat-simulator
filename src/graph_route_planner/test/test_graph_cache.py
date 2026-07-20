"""Characterisation of the cached visibility graph.

`CachedPlanner` splits planning into a wind-independent geometry half that is
kept between calls and a per-wind costing half. The whole point is that it
changes performance and nothing else, so the central test here is equivalence
with `plan_route` — identical waypoints, not merely similar ones.
"""

import pytest

from graph_route_planner import MAPS_DIR
from graph_route_planner.graph import (
    cost_pairs,
    extract_grid_nodes,
    visible_from,
    visible_pairs,
)
from graph_route_planner.map_loader import load_kml
from graph_route_planner.planner import (
    CachedPlanner,
    NoWaterError,
    PlannerConfig,
    plan_route,
    resolve_max_leg,
    select_navigable,
)
from graph_route_planner.sailing import SailingModel

GOLDEN = {'max_leg': 333.75445858473796, 'n_nodes': 132, 'nodes_head': [[-117.99999532610485, 118.0000455903389], [117.99999532623252, 118.0000455903389], [117.99999532623252, -118.0000455903389], [-117.99999532610485, -118.0000455903389], [50.00000404242127, -2.0], [50.19603832308039, -1.9903694533443936]], 'n_pairs': 3357, 'pairs_head': [[0, 1], [0, 3], [0, 33], [0, 34], [0, 35], [0, 36], [0, 37], [0, 38], [0, 39], [0, 40]], 'visible_from_count': 67, 'visible_from_head': [0, 1, 3, 36, 37, 38, 39, 40, 41, 42], 'n_edges': 4680, 'edge_sample': [[[-117.99999532610485, 118.0000455903389], [-117.99999532610485, -118.0000455903389], 29.483631602250114], [[-117.99999532610485, 118.0000455903389], [-117.99999532610485, -88.0000455903389], 25.73571377881869], [[-117.99999532610485, 118.0000455903389], [-117.99999532610485, -58.0000455903389], 21.987795955387266], [[-117.99999532610485, 118.0000455903389], [-117.99999532610485, -28.0000455903389], 18.239878131955845], [[-117.99999532610485, 118.0000455903389], [-117.99999532610485, 1.9999544096611004], 14.491960308524419], [[-117.99999532610485, 118.0000455903389], [-117.99999532610485, 31.9999544096611], 10.744042485092995], [[-117.99999532610485, 118.0000455903389], [-117.99999532610485, 61.9999544096611], 6.996124661661573], [[-117.99999532610485, 118.0000455903389], [-117.99999532610485, 91.9999544096611], 3.2482068382301494]]}

START, GOAL = (-10.0, 60.0), (-10.0, -10.0)
MISSION = [(-10.0, 60.0), (-10.0, -10.0), (60.0, -10.0), (60.0, 60.0), (-10.0, 60.0)]


@pytest.fixture(scope="module")
def water():
    return load_kml(str(MAPS_DIR / "OpenWaterCourse.kml")).water


@pytest.fixture
def config():
    return PlannerConfig(margin=2.0, grid_spacing=30.0, merge_threshold=1.5,
                         min_leg_distance=0.0)


@pytest.fixture
def navigable(water, config):
    return select_navigable(water, START, GOAL, config.margin)


def test_grid_nodes_are_unchanged(navigable, config):
    nodes = extract_grid_nodes(navigable, config.grid_spacing)
    assert len(nodes) == GOLDEN["n_nodes"]
    for got, want in zip(nodes[:6], GOLDEN["nodes_head"]):
        assert got == pytest.approx(want, rel=1e-12)


def test_grid_nodes_exclude_start_and_goal(navigable, config):
    """The cached node set must not depend on the query, or it cannot be cached."""
    nodes = extract_grid_nodes(navigable, config.grid_spacing)
    assert START not in nodes and GOAL not in nodes


def test_visible_pairs_are_unchanged(navigable, config):
    nodes = extract_grid_nodes(navigable, config.grid_spacing)
    max_leg = resolve_max_leg(config, navigable)
    pairs = visible_pairs(nodes, navigable, max_leg, config.min_leg_distance)
    assert len(pairs) == GOLDEN["n_pairs"]
    assert [list(p) for p in pairs[:10]] == GOLDEN["pairs_head"]
    assert all(i < j for i, j in pairs), "pairs must be upper-triangular"


def test_visible_from_is_unchanged(navigable, config):
    nodes = extract_grid_nodes(navigable, config.grid_spacing)
    max_leg = resolve_max_leg(config, navigable)
    seen = visible_from(START, nodes, navigable, max_leg, config.min_leg_distance)
    assert len(seen) == GOLDEN["visible_from_count"]
    assert seen[:10] == GOLDEN["visible_from_head"]


def test_visible_from_agrees_with_visible_pairs(navigable, config):
    """Attaching a node must see exactly what the n^2 pass would have found."""
    nodes = extract_grid_nodes(navigable, config.grid_spacing)
    max_leg = resolve_max_leg(config, navigable)
    pairs = set(visible_pairs(nodes, navigable, max_leg, config.min_leg_distance))
    probe = nodes[7]
    from_pairs = {j for i, j in pairs if i == 7} | {i for i, j in pairs if j == 7}
    from_scan = set(visible_from(probe, nodes, navigable, max_leg, config.min_leg_distance))
    assert from_scan - {7} == from_pairs


def test_cost_pairs_are_unchanged(navigable, config):
    nodes = extract_grid_nodes(navigable, config.grid_spacing)
    max_leg = resolve_max_leg(config, navigable)
    pairs = visible_pairs(nodes, navigable, max_leg, config.min_leg_distance)
    # segment_cost=0 on purpose: these goldens pin the sailing-cost maths, and
    # must not move when the route-shaping default is retuned.
    graph = cost_pairs(nodes, pairs, SailingModel.from_bearing(0.1, no_go_deg=50),
                       segment_cost=0.0)
    assert sum(len(v) for v in graph.values()) == GOLDEN["n_edges"]
    sample = sorted(
        (list(u), list(v), c) for u, es in list(graph.items())[:3] for v, c in es.items()
    )[:8]
    assert len(sample) == len(GOLDEN["edge_sample"])
    for (u, v, cost), (wu, wv, wcost) in zip(sample, GOLDEN["edge_sample"]):
        assert u == pytest.approx(wu, rel=1e-12)
        assert v == pytest.approx(wv, rel=1e-12)
        assert cost == pytest.approx(wcost, rel=1e-12)


def test_cost_pairs_covers_every_node(navigable, config):
    nodes = extract_grid_nodes(navigable, config.grid_spacing)
    max_leg = resolve_max_leg(config, navigable)
    pairs = visible_pairs(nodes, navigable, max_leg, config.min_leg_distance)
    graph = cost_pairs(nodes, pairs, SailingModel.from_bearing(0.1, no_go_deg=50))
    assert set(graph) == set(nodes)


# ── the equivalence that justifies the whole thing ────────────────────────────

@pytest.mark.parametrize("bearing", [0.1, 90.0, 200.0])
def test_cached_planner_matches_plan_route(water, config, bearing):
    model = SailingModel.from_bearing(bearing, no_go_deg=50)
    planner = CachedPlanner(water, config)
    for start, goal in zip(MISSION, MISSION[1:]):
        want = plan_route(water, start, goal, model=model, config=config)
        got = planner.plan(start, goal, model)
        assert got.waypoints == want.waypoints, f"{start} -> {goal} at {bearing} deg"
        assert got.duration == pytest.approx(want.duration, rel=1e-12)


def test_geometry_is_built_once_across_a_mission(water, config):
    model = SailingModel.from_bearing(0.1, no_go_deg=50)
    planner = CachedPlanner(water, config)
    for start, goal in zip(MISSION, MISSION[1:]):
        planner.plan(start, goal, model)
    assert planner.builds == 1


def test_a_wind_change_does_not_rebuild_the_geometry(water, config):
    planner = CachedPlanner(water, config)
    planner.plan(START, GOAL, SailingModel.from_bearing(0.1, no_go_deg=50))
    planner.plan(START, GOAL, SailingModel.from_bearing(180.0, no_go_deg=50))
    assert planner.builds == 1


def test_a_new_boat_position_does_not_rebuild_the_geometry(water, config):
    model = SailingModel.from_bearing(0.1, no_go_deg=50)
    planner = CachedPlanner(water, config)
    planner.plan(START, GOAL, model)
    for boat in [(-20.0, 40.0), (-30.0, 10.0), (-8.0, -5.0)]:
        planner.plan(boat, GOAL, model)
    assert planner.builds == 1


def test_the_direct_leg_is_offered(water, config):
    """Neither endpoint is in the cached node set, so this edge is easy to lose."""
    model = SailingModel.from_bearing(0.1, no_go_deg=50)
    planner = CachedPlanner(water, config)
    # Straight downwind across open water: the answer is a single leg.
    route = planner.plan((-60.0, 60.0), (-60.0, -60.0), model)
    assert route.found
    assert route.waypoints[0] == (-60.0, 60.0)
    assert route.waypoints[-1] == (-60.0, -60.0)


def test_a_start_off_the_water_is_reported(water, config):
    planner = CachedPlanner(water, config)
    with pytest.raises(NoWaterError):
        planner.plan((25.0, 25.0), GOAL, SailingModel())   # inside the course island


def test_cached_planner_reuses_across_repeated_identical_plans(water, config):
    model = SailingModel.from_bearing(0.1, no_go_deg=50)
    planner = CachedPlanner(water, config)
    first = planner.plan(START, GOAL, model)
    second = planner.plan(START, GOAL, model)
    assert first.waypoints == second.waypoints
    assert planner.builds == 1
