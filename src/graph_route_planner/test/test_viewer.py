"""Tests for the interactive viewer's state machine.

Driven headlessly with synthetic events; no GUI backend is involved.
"""

import types

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt        # noqa: E402

from graph_route_planner import MAPS_DIR                       # noqa: E402
from graph_route_planner.geometry import bbox_diagonal         # noqa: E402
from graph_route_planner.map_loader import load_kml            # noqa: E402
from graph_route_planner.planner import PlannerConfig          # noqa: E402
from graph_route_planner.sailing import SailingModel           # noqa: E402
from graph_route_planner.viewer import RoutePlannerViewer      # noqa: E402


@pytest.fixture
def sail_map():
    return load_kml(MAPS_DIR / "Test.kml")


@pytest.fixture
def viewer(sail_map):
    v = RoutePlannerViewer(sail_map)
    yield v
    plt.close(v.fig)


def click(viewer, x: float, y: float) -> None:
    """Synthesise a click on the map axes."""
    viewer._on_click(types.SimpleNamespace(inaxes=viewer.ax, xdata=x, ydata=y))


def water_point(viewer) -> tuple[float, float]:
    p = viewer.map.components[0].representative_point()
    return (p.x, p.y)


def land_point(viewer) -> tuple[float, float]:
    insel = next(f for f in viewer.map.features if f.name == "Landgraben Insel")
    lake = next(f for f in viewer.map.features if f.name == "Lake on Island")
    p = insel.polygon.difference(lake.polygon).representative_point()
    return (p.x, p.y)


def test_starts_with_no_points(viewer):
    assert viewer.start is None
    assert viewer.goal is None


def test_clicking_land_is_refused(viewer):
    click(viewer, *land_point(viewer))
    assert viewer.start is None
    assert "land" in viewer.status.lower()


def test_first_click_sets_start_and_does_not_plan(viewer):
    click(viewer, *water_point(viewer))
    assert viewer.start is not None
    assert viewer.goal is None
    assert not viewer.path


def test_second_click_plans(viewer):
    click(viewer, -30.0, -1.1)
    click(viewer, 22.0, 35.0)
    assert viewer.goal is not None
    assert viewer.path is not None
    assert "Route" in viewer.status


def test_third_click_starts_a_fresh_pair(viewer):
    click(viewer, -30.0, -1.1)
    click(viewer, 22.0, 35.0)
    click(viewer, -25.0, -20.0)
    assert viewer.goal is None
    assert not viewer.path


def test_clear_resets_everything(viewer):
    click(viewer, -30.0, -1.1)
    click(viewer, 22.0, 35.0)
    viewer._on_reset(None)
    assert viewer.start is None
    assert viewer.goal is None
    assert not viewer.path


def test_goal_on_another_body_is_reported(viewer):
    inner = viewer.map.components[1].representative_point()
    click(viewer, -30.0, -1.1)
    click(viewer, inner.x, inner.y)
    assert not viewer.path
    assert "different body of water" in viewer.status


def test_slider_marks_dirty_but_does_not_plan_until_release(viewer):
    click(viewer, -30.0, -1.1)
    click(viewer, 22.0, 35.0)
    before = viewer.status

    viewer.s_margin.set_val(4.0)
    assert viewer._dirty is True
    assert viewer.status == before          # not replanned yet

    viewer._on_release(None)
    assert viewer._dirty is False


def test_wind_slider_starts_at_the_models_bearing(sail_map):
    v = RoutePlannerViewer(sail_map, model=SailingModel.from_bearing(120.0))
    assert v.s_wind.val == pytest.approx(120.0)
    plt.close(v.fig)


def test_wind_slider_rebuilds_the_model(viewer):
    viewer.s_wind.set_val(90.0)
    assert viewer.model.wind_bearing_deg == pytest.approx(90.0)
    assert viewer._dirty is True


def test_wind_slider_keeps_the_no_go_zone(sail_map):
    v = RoutePlannerViewer(sail_map, model=SailingModel(no_go_deg=30.0))
    v.s_wind.set_val(200.0)
    assert v.model.no_go_deg == pytest.approx(30.0)
    plt.close(v.fig)


def test_turning_the_wind_changes_the_route(viewer):
    """The wind slider must reach the planner, not just the label."""
    click(viewer, -30.0, -1.1)
    click(viewer, 22.0, 35.0)
    assert viewer.path is not None
    upwind = viewer.status

    # Turn the wind around so the same trip runs downwind instead.
    viewer.s_wind.set_val(180.0)
    viewer._on_release(None)

    assert viewer.path is not None
    assert viewer.status != upwind


def test_wind_can_be_turned_before_any_points_are_set(viewer):
    """Turning the wind with no route must redraw, not crash."""
    viewer.s_wind.set_val(270.0)
    viewer._on_release(None)
    assert viewer.model.wind_bearing_deg == pytest.approx(270.0)
    assert not viewer.path


@pytest.mark.parametrize("bearing", [0.0, 360.0])
def test_wind_slider_endpoints_are_valid(viewer, bearing):
    viewer.s_wind.set_val(bearing)
    assert viewer.model.wind_bearing_deg == pytest.approx(0.0, abs=1e-6)


def test_leg_sliders_start_from_the_config(sail_map):
    v = RoutePlannerViewer(sail_map, config=PlannerConfig(min_leg_distance=3.0,
                                                         max_leg_distance=90.0))
    assert v.s_min_leg.val == pytest.approx(3.0)
    assert v.s_max_leg.val == pytest.approx(90.0)
    plt.close(v.fig)


def test_max_leg_slider_defaults_to_the_maps_span(viewer):
    """It opens at the top: no leg the boat could sail is truncated."""
    assert viewer.s_max_leg.val == viewer.s_max_leg.valmax
    assert viewer.s_max_leg.val >= bbox_diagonal(viewer.map.water)


def test_leg_sliders_reach_the_planner(viewer):
    viewer.s_max_leg.set_val(60.0)
    cfg = viewer._config_from_sliders()
    assert cfg.max_leg_distance == pytest.approx(60.0)
    assert viewer._dirty is True


def test_default_does_not_force_a_via_point(viewer):
    """The 63 m crossing is sailed straight, with no tuning at all."""
    viewer.s_wind.set_val(90.0)
    click(viewer, -28.5, -40.7)
    click(viewer, -31.5, 22.5)
    assert len(viewer.path) == 2


def test_lowering_max_leg_reintroduces_the_detour(viewer):
    """Dragging the limit below the crossing is an explicit accuracy trade."""
    viewer.s_wind.set_val(90.0)
    click(viewer, -28.5, -40.7)
    click(viewer, -31.5, 22.5)
    assert len(viewer.path) == 2

    viewer.s_max_leg.set_val(60.0)      # below the 63 m crossing
    viewer._on_release(None)
    assert len(viewer.path) == 3


def test_dragging_min_leg_above_max_does_not_crash(viewer):
    """The sliders are independent; PlannerConfig rejects an empty range."""
    viewer.s_max_leg.set_val(10.0)
    viewer.s_min_leg.set_val(20.0)
    cfg = viewer._config_from_sliders()
    assert cfg.min_leg_distance < cfg.max_leg_distance

    click(viewer, -30.0, -1.1)
    click(viewer, 22.0, 35.0)             # must not raise


def test_graph_toggle(viewer):
    click(viewer, -30.0, -1.1)
    click(viewer, 22.0, 35.0)
    assert viewer.show_graph is False
    viewer.check.set_active(0)
    assert viewer.show_graph is True


def test_replan_is_not_reentrant(viewer, monkeypatch):
    """Events dispatched mid-plan must not start a second plan.

    A GUI callback that re-enters the event loop can deliver a queued click or
    slider release while planning is still running. Without a guard that
    recurses without bound and the window locks up for good.
    """
    import graph_route_planner.viewer as viewer_mod

    depth = {"now": 0, "max": 0}
    real_plan = viewer_mod.plan_route

    def reentrant_plan(*args, **kwargs):
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        try:
            # Stand in for flush_events dispatching queued input.
            viewer._dirty = True
            viewer._on_release(None)
            click(viewer, -30.0, -1.1)
            return real_plan(*args, **kwargs)
        finally:
            depth["now"] -= 1

    monkeypatch.setattr(viewer_mod, "plan_route", reentrant_plan)

    viewer.start, viewer.goal = (-30.0, -1.1), (22.0, 35.0)
    viewer.replan()

    assert depth["max"] == 1
