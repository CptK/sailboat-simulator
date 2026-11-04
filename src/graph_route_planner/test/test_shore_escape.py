"""A start inside the shore margin, and mission points snapped out of it.

A boat can drift closer to shore than the margin. The planner used to refuse
to plan from there, so the boat never got a route out. Now such a start is
attached through the real water, and the time it spends in the strip is
charged extra so the route leaves the shore first.
"""

import math

import pytest
from shapely.geometry import LineString, Point, box

from graph_route_planner.graph import navigable_area
from graph_route_planner.planner import (
    SNAP_INSET,
    CachedPlanner,
    NoWaterError,
    PlannerConfig,
    plan_route,
    snap_to_navigable,
    true_duration,
)
from graph_route_planner.sailing import SailingModel

MARGIN = 2.0
WATER = box(0, 0, 100, 60)
SAFE = navigable_area(WATER, MARGIN)
CONFIG = PlannerConfig(margin=MARGIN, grid_spacing=6.0, merge_threshold=1.5)

#: 1 m off the south shore, inside the 2 m margin.
IN_STRIP = (50.0, 1.0)
NORTHERLY = SailingModel.from_bearing(0.0, no_go_deg=50.0)    # onshore here
SOUTHERLY = SailingModel.from_bearing(180.0, no_go_deg=50.0)  # offshore here


def strip_length(waypoints) -> float:
    """How much of a route lies inside the shore margin, in metres."""
    line = LineString(waypoints)
    return line.length - line.intersection(SAFE).length


def plan_both(start, goal, model, config=CONFIG):
    return [CachedPlanner(WATER, config).plan(start, goal, model),
            plan_route(WATER, start, goal, model, config)]


@pytest.mark.parametrize("model", [NORTHERLY, SOUTHERLY])
def test_a_start_inside_the_margin_gets_a_route(model):
    for route in plan_both(IN_STRIP, (90.0, 50.0), model):
        assert route.found
        assert route.waypoints[0] == IN_STRIP
        # Only the first leg may touch the strip; everything after is safe water.
        assert all(SAFE.covers(Point(p)) for p in route.waypoints[1:])
        if len(route.waypoints) > 2:
            assert SAFE.covers(LineString(route.waypoints[1:]))


def test_an_onshore_wind_escapes_on_a_sailable_heading():
    """Straight off this shore is dead upwind; the way out is a close reach."""
    for route in plan_both(IN_STRIP, (90.0, 50.0), NORTHERLY):
        start, first = route.waypoints[:2]
        assert NORTHERLY.heading_is_sailable(start, first)
        assert all(NORTHERLY.heading_is_sailable(u, v)
                   for u, v in zip(route.waypoints, route.waypoints[1:]))


def test_the_penalty_gets_off_the_shore_before_heading_for_the_goal():
    """A goal along the shore tempts a free strip into a long run inside it."""
    goal = (95.0, 2.5)
    free = plan_both(IN_STRIP, goal, SOUTHERLY, PlannerConfig(
        margin=MARGIN, grid_spacing=6.0, merge_threshold=1.5, shore_penalty=0.0))
    weighted = plan_both(IN_STRIP, goal, SOUTHERLY)

    for lax, strict in zip(free, weighted):
        assert strip_length(lax.waypoints) > 10.0
        # Straight off a 1 m-deep incursion with an offshore wind: about 1 m.
        assert strip_length(strict.waypoints) == pytest.approx(1.0, abs=0.1)


def test_the_penalty_is_not_part_of_the_duration():
    for route in plan_both(IN_STRIP, (90.0, 50.0), NORTHERLY):
        assert route.duration == pytest.approx(true_duration(route.waypoints, NORTHERLY))


def test_planning_from_the_strip_reuses_the_cache():
    planner = CachedPlanner(WATER, CONFIG)
    planner.plan((50.0, 30.0), (90.0, 50.0), NORTHERLY)
    for start in [IN_STRIP, (10.0, 0.5), (99.5, 30.0), (50.0, 30.0)]:
        assert planner.plan(start, (90.0, 50.0), NORTHERLY).found
    assert planner.builds == 1


def test_a_first_plan_from_the_strip_builds_once():
    planner = CachedPlanner(WATER, CONFIG)
    for _ in range(3):
        assert planner.plan(IN_STRIP, (90.0, 50.0), NORTHERLY).found
    assert planner.builds == 1


def test_no_edge_leads_back_into_the_strip():
    route = CachedPlanner(WATER, CONFIG).plan(IN_STRIP, (90.0, 50.0), NORTHERLY)
    assert not any(IN_STRIP in edges for node, edges in route.graph.items()
                   if node != IN_STRIP)


# ── the margin still means something ──────────────────────────────────────────

#: Two lakes joined by a channel 3 m wide, which a 2 m margin closes.
CHANNEL = box(0, 0, 40, 40).union(box(40, 18.5, 60, 21.5)).union(box(60, 0, 100, 40))


def test_the_strip_is_not_a_way_through_a_closed_channel():
    """A boat in the strip escapes to its nearest safe water, not any of it."""
    in_left_strip = (39.0, 20.0)
    with pytest.raises(NoWaterError, match="unreachable"):
        CachedPlanner(CHANNEL, CONFIG).plan(in_left_strip, (80.0, 20.0), NORTHERLY)
    with pytest.raises(NoWaterError, match="unreachable"):
        plan_route(CHANNEL, in_left_strip, (80.0, 20.0), NORTHERLY, CONFIG)


def test_the_strip_beside_the_goals_piece_is_fine():
    assert CachedPlanner(CHANNEL, CONFIG).plan((61.0, 20.0), (80.0, 20.0), NORTHERLY).found


def test_a_goal_inside_the_margin_is_still_rejected():
    with pytest.raises(NoWaterError, match="within"):
        CachedPlanner(WATER, CONFIG).plan((50.0, 30.0), (90.0, 1.0), NORTHERLY)


def test_a_start_on_land_is_still_rejected():
    with pytest.raises(NoWaterError, match="not on water"):
        CachedPlanner(WATER, CONFIG).plan((50.0, -1.0), (90.0, 50.0), NORTHERLY)


def test_config_rejects_a_negative_shore_penalty():
    with pytest.raises(ValueError, match="shore_penalty"):
        PlannerConfig(shore_penalty=-1.0)


# ── snapping mission points ───────────────────────────────────────────────────

def test_a_point_clear_of_the_shore_is_not_moved():
    assert snap_to_navigable(WATER, (50.0, 30.0), MARGIN) == (50.0, 30.0)


@pytest.mark.parametrize("point", [(50.0, 1.0), (0.5, 30.0), (99.0, 59.0), (100.0, 0.0)])
def test_a_point_in_the_strip_moves_just_inside_the_margin(point):
    snapped = snap_to_navigable(WATER, point, MARGIN)
    assert SAFE.covers(Point(snapped))
    assert math.dist(point, snapped) <= Point(point).distance(SAFE) + SNAP_INSET + 1e-9
    # And it is a planner-acceptable goal.
    assert CachedPlanner(WATER, CONFIG).plan((50.0, 30.0), snapped, NORTHERLY).found


def test_snapping_stays_in_the_points_own_piece():
    assert snap_to_navigable(CHANNEL, (39.0, 20.0), MARGIN)[0] < 40.0


def test_snapping_a_point_on_land_raises():
    with pytest.raises(NoWaterError):
        snap_to_navigable(WATER, (50.0, -1.0), MARGIN)


# ── the node snaps a mission once, and says so ────────────────────────────────

class _Log:
    def __init__(self):
        self.warnings, self.errors = [], []

    def warn(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        pass


def _mission_node():
    """A stand-in carrying what `_on_mission_change` reads and writes."""
    import types

    from graph_route_planner.geometry import component_containing

    log = _Log()
    return types.SimpleNamespace(
        config=types.SimpleNamespace(margin=MARGIN),
        map=types.SimpleNamespace(water=WATER,
                                  component_for=lambda p: component_containing(WATER, p)),
        get_logger=lambda: log,
        log=log,
    )


def _mission(*points):
    import types
    return types.SimpleNamespace(
        waypoints=[types.SimpleNamespace(east=e, north=n) for e, n in points])


def test_the_node_snaps_mission_points_in_the_margin_and_logs_it():
    from graph_route_planner.node import GraphRoutePlanner

    node = _mission_node()
    GraphRoutePlanner._on_mission_change(node, _mission((50.0, 30.0), (90.0, 0.5)))

    assert node.mission[0] == (50.0, 30.0)
    assert SAFE.covers(Point(node.mission[1]))
    assert len(node.log.warnings) == 1 and "via point 1" in node.log.warnings[0]
    assert not node.log.errors


def test_the_node_leaves_a_clear_mission_alone():
    from graph_route_planner.node import GraphRoutePlanner

    node = _mission_node()
    GraphRoutePlanner._on_mission_change(node, _mission((50.0, 30.0), (90.0, 50.0)))
    assert node.mission == [(50.0, 30.0), (90.0, 50.0)]
    assert not node.log.warnings


# ── a grounded boat ───────────────────────────────────────────────────────────

def test_a_start_on_water_is_not_moved():
    from graph_route_planner.planner import snap_to_water
    assert snap_to_water(WATER, IN_STRIP) == IN_STRIP


def test_a_start_just_on_land_moves_onto_the_water_and_plans():
    from graph_route_planner.planner import snap_to_water
    start = snap_to_water(WATER, (50.0, -0.3))
    assert WATER.covers(Point(start))
    assert math.dist(start, (50.0, -0.3)) <= 0.3 + SNAP_INSET + 1e-9
    assert CachedPlanner(WATER, CONFIG).plan(start, (90.0, 50.0), NORTHERLY).found


def test_the_landgraben_grounding_is_planned_out():
    """Regression: the boat ran aground here and no route was ever planned.

    Wind from the south, close-hauled on a leg along the margin's edge; leeway
    put the boat 0.3 m off the shore at (-17.3, -2.8).
    """
    from graph_route_planner import MAPS_DIR
    from graph_route_planner.map_loader import load_kml

    water = load_kml(str(MAPS_DIR / "Landgraben.kml")).water
    model = SailingModel.from_bearing(180.0, no_go_deg=50.0)
    planner = CachedPlanner(water, CONFIG)
    planner.plan((-32.0, 10.0), (0.0, -35.0), model)

    route = planner.plan((-17.3, -2.8), (0.0, -35.0), model)
    assert route.found
    assert all(model.heading_is_sailable(u, v)
               for u, v in zip(route.waypoints, route.waypoints[1:]))
    assert planner.builds == 1


# ── a boat with no way on cannot maneuver its way out ─────────────────────────

def _turn(route, heading):
    (ax, ay), (bx, by) = route.waypoints[:2]
    leg = math.degrees(math.atan2(bx - ax, by - ay)) % 360.0
    return leg, (leg - heading + 180.0) % 360.0 - 180.0


def test_the_pinned_landgraben_boat_sails_out_straight_ahead():
    """Regression: land to port, wind from starboard pinning the boat on it.

    Without its heading the planner sent the boat out to the south-west — a
    111° tack it had no speed for, while bearing away put the bow into the land.
    Close-hauled straight ahead is open water.
    """
    from graph_route_planner import MAPS_DIR
    from graph_route_planner.map_loader import load_kml

    water = load_kml(str(MAPS_DIR / "Landgraben.kml")).water
    model = SailingModel.from_bearing(180.0, no_go_deg=50.0)
    heading = 129.1
    for route in [CachedPlanner(water, CONFIG).plan((-16.82, -3.17), (0.0, -35.0), model, heading),
                  plan_route(water, (-16.82, -3.17), (0.0, -35.0), model, CONFIG, heading)]:
        assert route.found
        leg, turn = _turn(route, heading)
        assert abs(turn) <= CONFIG.escape_max_turn
        # Still on starboard tack: the wind is on the same side as before.
        assert ((180.0 - leg) % 360.0 < 180.0) == ((180.0 - heading) % 360.0 < 180.0)


def test_without_a_heading_any_sailable_way_out_is_allowed():
    route = CachedPlanner(WATER, CONFIG).plan(IN_STRIP, (90.0, 50.0), SOUTHERLY)
    assert route.found


def test_no_way_out_on_this_tack_falls_back_to_any_sailable_leg():
    """A route that needs a tack beats no route: a boat with way on can make it."""
    from dataclasses import replace
    config = replace(CONFIG, escape_max_turn=0.0)
    route = CachedPlanner(WATER, config).plan(IN_STRIP, (90.0, 50.0), NORTHERLY, heading=37.0)
    assert route.found


def test_the_heading_is_ignored_outside_the_margin():
    planner = CachedPlanner(WATER, CONFIG)
    free = planner.plan((50.0, 30.0), (90.0, 50.0), NORTHERLY)
    assert planner.plan((50.0, 30.0), (90.0, 50.0), NORTHERLY, heading=200.0).waypoints == free.waypoints


@pytest.mark.parametrize("turn", [-1.0, 181.0])
def test_config_rejects_an_escape_turn_outside_a_half_circle(turn):
    with pytest.raises(ValueError, match="escape_max_turn"):
        PlannerConfig(escape_max_turn=turn)
