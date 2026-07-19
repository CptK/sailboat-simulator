"""Tests for route planning, including the water-component rules."""

import math
from dataclasses import replace

import pytest
from shapely.geometry import Point

from graph_route_planner import MAPS_DIR
from graph_route_planner.geometry import polygons_of
from graph_route_planner.map_loader import load_kml
from graph_route_planner.planner import (
    NoWaterError,
    PlannerConfig,
    plan_route,
    resolve_max_leg,
    select_navigable,
)
from graph_route_planner.sailing import SailingModel
from graph_route_planner.scenario import synthetic


@pytest.fixture
def nested():
    return load_kml(MAPS_DIR / "Test.kml")


@pytest.fixture
def bodies(nested):
    """The outer lake and the lake sitting on its island."""
    outer, inner = nested.components[0], nested.components[1]
    return outer, inner


#: Tuning that suits the pond-sized Test.kml; derive variants with replace().
POND = PlannerConfig(margin=2.0, grid_spacing=6.0, merge_threshold=1.5)


def point_in(poly) -> tuple[float, float]:
    p = poly.representative_point()
    return (p.x, p.y)


def as_position(coord) -> tuple[float, float]:
    """A shapely coord is tuple[float, ...]; the planner wants exactly two."""
    return (coord[0], coord[1])


# ── the synthetic scenario ────────────────────────────────────────────────────

def test_synthetic_scenario_plans():
    scene = synthetic()
    route = plan_route(scene.water, scene.start, scene.goal,
                       config=PlannerConfig(margin=5.0, grid_spacing=12.0,
                                            merge_threshold=3.0))
    assert route.found
    assert len(route.waypoints) >= 2
    assert route.waypoints[0] == scene.start
    assert route.waypoints[-1] == scene.goal


def test_every_leg_of_a_route_is_sailable():
    """The planner must never emit a leg inside the no-go zone."""
    scene = synthetic()
    model = SailingModel()
    route = plan_route(scene.water, scene.start, scene.goal, model=model,
                       config=PlannerConfig(margin=5.0, grid_spacing=12.0))
    assert route.found
    for u, v in zip(route.waypoints, route.waypoints[1:]):
        assert model.heading_is_sailable(u, v)


def test_route_stays_on_water():
    scene = synthetic()
    route = plan_route(scene.water, scene.start, scene.goal,
                       config=PlannerConfig(margin=5.0, grid_spacing=12.0))
    for wp in route.waypoints:
        assert scene.water.covers(Point(wp))


def test_wind_direction_changes_the_route():
    """Flipping the wind must change the plan, or wind is being ignored."""
    scene = synthetic()
    config = PlannerConfig(margin=5.0, grid_spacing=12.0)
    north = plan_route(scene.water, scene.start, scene.goal,
                       model=SailingModel.from_bearing(0.0), config=config)
    south = plan_route(scene.water, scene.start, scene.goal,
                       model=SailingModel.from_bearing(180.0), config=config)
    assert north.found and south.found
    assert north.duration != pytest.approx(south.duration)


# ── water components ──────────────────────────────────────────────────────────

def test_start_on_land_is_rejected(nested):
    insel = next(f for f in nested.features if f.name == "Landgraben Insel")
    lake = next(f for f in nested.features if f.name == "Lake on Island")
    land = point_in(insel.polygon.difference(lake.polygon))

    with pytest.raises(NoWaterError, match="not on water"):
        plan_route(nested.water, land, point_in(nested.components[0]))


def test_cannot_sail_between_separate_bodies(bodies, nested):
    """No portage: the lake on the island is a different world."""
    outer, inner = bodies
    with pytest.raises(NoWaterError, match="different body of water"):
        plan_route(nested.water, point_in(outer), point_in(inner),
                   config=PlannerConfig(margin=1.0, grid_spacing=6.0))


def test_can_sail_within_the_lake_on_the_island(bodies):
    """The inner lake is sailable in its own right."""
    _outer, inner = bodies
    usable = inner.buffer(-1.5)
    pts = list(usable.exterior.coords)
    start, goal = as_position(pts[0]), as_position(pts[len(pts) // 2])

    route = plan_route(inner, start, goal,
                       config=PlannerConfig(margin=1.0, grid_spacing=2.5,
                                            merge_threshold=0.8))
    assert route.found
    for wp in route.waypoints:
        assert inner.covers(Point(wp))


def test_margin_that_swallows_the_water_is_reported(bodies):
    _outer, inner = bodies
    with pytest.raises(NoWaterError):
        plan_route(inner, point_in(inner), point_in(inner),
                   config=PlannerConfig(margin=50.0))


def test_select_navigable_erodes_the_body(bodies):
    outer, _inner = bodies
    start = goal = point_in(outer)
    navigable = select_navigable(outer, start, goal, margin=2.0)
    assert navigable.area < outer.area


# ── leg length bounds ─────────────────────────────────────────────────────────

def test_config_rejects_negative_min_leg():
    with pytest.raises(ValueError, match="min_leg_distance must be >= 0"):
        PlannerConfig(min_leg_distance=-1.0)


def test_config_rejects_empty_leg_range():
    """min above max leaves no legal leg; fail loudly, not with an empty graph."""
    with pytest.raises(ValueError, match="must be below"):
        PlannerConfig(min_leg_distance=80.0, max_leg_distance=60.0)


def test_min_leg_defaults_to_disabled():
    assert PlannerConfig().min_leg_distance == 0.0


def test_max_leg_defaults_to_the_map_and_does_not_truncate(nested):
    """The default must sail the 63 m crossing straight, with no tuning.

    A fixed 60 m limit used to filter this leg out before the visibility check
    ever ran, forcing a detour through a waypoint the boat does not need.
    """
    model = SailingModel.from_bearing(90.0)
    start, goal = (-28.5, -40.7), (-31.5, 22.5)
    auto = plan_route(nested.water, start, goal, model=model,
                      config=POND)
    assert len(auto.waypoints) == 2


def test_explicit_max_leg_overrides_the_derived_default(nested):
    """An explicit limit always wins, even when it truncates."""
    model = SailingModel.from_bearing(90.0)
    start, goal = (-28.5, -40.7), (-31.5, 22.5)
    auto = plan_route(nested.water, start, goal, model=model,
                      config=POND)
    clipped = plan_route(nested.water, start, goal, model=model,
                         config=replace(POND, max_leg_distance=60.0))

    assert len(clipped.waypoints) == 3
    assert auto.duration <= clipped.duration


def test_resolve_max_leg_prefers_an_explicit_value():
    from shapely.geometry import box
    area = box(0, 0, 100, 100)                      # diagonal ~141
    assert resolve_max_leg(PlannerConfig(max_leg_distance=25.0), area) == 25.0
    assert resolve_max_leg(PlannerConfig(), area) == pytest.approx(141.42, abs=0.01)


def test_derived_max_leg_never_truncates_a_legal_leg():
    """The bbox diagonal bounds any leg two points inside the area can form."""
    from shapely.geometry import box
    area = box(0, 0, 80, 60)
    derived = resolve_max_leg(PlannerConfig(), area)
    corners = [(0, 0), (80, 0), (0, 60), (80, 60)]
    for a in corners:
        for b in corners:
            assert math.dist(a, b) <= derived + 1e-9


def test_derived_max_leg_below_min_leg_is_rejected():
    from shapely.geometry import box
    tiny = box(0, 0, 5, 5)                          # diagonal ~7.1
    with pytest.raises(ValueError, match="exceeds the map's own span"):
        resolve_max_leg(PlannerConfig(min_leg_distance=20.0), tiny)


def test_min_leg_suppresses_short_legs(nested):
    """No leg in the route may be shorter than min_leg_distance."""
    model = SailingModel.from_bearing(90.0)
    start, goal = (-28.5, -40.7), (-31.5, 22.5)
    route = plan_route(nested.water, start, goal, model=model,
                       config=PlannerConfig(margin=2.0, grid_spacing=6.0,
                                            merge_threshold=1.5,
                                            min_leg_distance=12.0,
                                            max_leg_distance=100.0))
    assert route.found
    for u, v in zip(route.waypoints, route.waypoints[1:]):
        assert math.dist(u, v) >= 12.0 - 1e-9


def test_min_leg_can_disconnect_the_graph(bodies):
    """Raising min leg past the map's scale legitimately kills every route.

    This is the cost of the knob: it is an edge filter, so it can leave the
    start with no neighbours at all. It must report no route, not crash.
    """
    _outer, inner = bodies
    pts = list(inner.buffer(-1.5).exterior.coords)
    start, goal = as_position(pts[0]), as_position(pts[len(pts) // 2])

    route = plan_route(inner, start, goal,
                       config=PlannerConfig(margin=1.0, grid_spacing=2.5,
                                            merge_threshold=0.8,
                                            min_leg_distance=150.0,
                                            max_leg_distance=200.0))
    assert not route.found


def test_min_leg_zero_matches_no_filter(nested):
    """The default must not change any existing behaviour."""
    model = SailingModel.from_bearing(90.0)
    start, goal = (-28.5, -40.7), (-31.5, 22.5)
    off = plan_route(nested.water, start, goal, model=model,
                     config=POND)
    zero = plan_route(nested.water, start, goal, model=model,
                      config=replace(POND, min_leg_distance=0.0))
    assert off.waypoints == zero.waypoints
    assert off.duration == pytest.approx(zero.duration)


def test_bigger_margin_gives_a_slower_route(bodies):
    """Clearance costs time: it pushes the route away from the direct line."""
    outer, _inner = bodies
    # Eroding can split a body, so take the largest surviving piece.
    usable = max(polygons_of(outer.buffer(-6.0)), key=lambda p: p.area)
    pts = list(usable.exterior.coords)
    start, goal = as_position(pts[0]), as_position(pts[len(pts) // 2])

    tight = plan_route(outer, start, goal,
                       config=PlannerConfig(margin=1.0, grid_spacing=6.0))
    loose = plan_route(outer, start, goal,
                       config=PlannerConfig(margin=4.0, grid_spacing=6.0))
    assert tight.found and loose.found
    assert loose.duration >= tight.duration
