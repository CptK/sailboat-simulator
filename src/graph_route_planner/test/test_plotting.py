"""Tests for the drawing layer.

These are smoke tests plus a check that the picture agrees with the model.
They do not assert on pixels; they assert the code runs and that labels are
not lying about the wind.
"""

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt                 # noqa: E402

from graph_route_planner import MAPS_DIR                   # noqa: E402
from graph_route_planner.map_loader import load_kml        # noqa: E402
from graph_route_planner.planner import PlannerConfig, plan_route  # noqa: E402
from graph_route_planner.plotting import (                 # noqa: E402
    draw_wind,
    plot_environment,
    plot_graph,
    plot_map,
    plot_polar,
)
from graph_route_planner.sailing import SailingModel       # noqa: E402
from graph_route_planner.scenario import synthetic         # noqa: E402


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture
def sail_map():
    return load_kml(MAPS_DIR / "Test.kml")


def texts_of(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def test_plot_map_runs(ax, sail_map):
    """plot_map is public API; it had a stale import and raised on every call."""
    plot_map(ax, sail_map)
    assert ax.patches


def test_plot_map_labels_every_feature(ax, sail_map):
    plot_map(ax, sail_map)
    labels = texts_of(ax)
    for feature in sail_map.features:
        assert feature.name in labels


def test_plot_polar_runs(ax):
    plot_polar(ax, SailingModel())
    assert ax.lines


@pytest.mark.parametrize("bearing", [0.0, 90.0, 180.0, 270.0])
def test_wind_label_reports_the_actual_wind(ax, bearing):
    """The label must follow the model, not assume a northerly."""
    model = SailingModel.from_bearing(bearing)
    draw_wind(ax, (0.0, 0.0, 100.0, 100.0), 100.0, model)
    assert any(f"{bearing:.0f}°" in t for t in texts_of(ax))


def test_wind_arrows_point_downwind(ax):
    """An easterly must blow towards the west, not southwards."""
    model = SailingModel.from_bearing(90.0)          # wind from the east
    draw_wind(ax, (0.0, 0.0, 100.0, 100.0), 100.0, model)

    # ax.texts holds both the plain label and the arrow Annotations.
    arrows = [a for a in ax.texts if getattr(a, "arrow_patch", None) is not None]
    assert arrows
    for a in arrows:
        tail_x, head_x = a.xyann[0], a.xy[0]
        assert head_x < tail_x                       # travelling west


def test_polar_label_reports_the_actual_wind(ax):
    plot_polar(ax, SailingModel.from_bearing(225.0))
    assert any("225" in t for t in texts_of(ax))


def test_plot_environment_runs(ax):
    scene = synthetic()
    model = SailingModel()
    route = plan_route(scene.water, scene.start, scene.goal, model=model,
                       config=PlannerConfig(margin=5.0, grid_spacing=12.0))
    plot_environment(ax, scene.water, route.navigable, route.waypoints,
                     scene.start, scene.goal, model)
    assert ax.patches


def test_plot_graph_uses_one_collection(ax):
    """Thousands of ax.plot() artists made every redraw take seconds."""
    scene = synthetic()
    route = plan_route(scene.water, scene.start, scene.goal,
                       config=PlannerConfig(margin=5.0, grid_spacing=12.0))
    plot_graph(ax, route.graph)

    assert len(ax.collections) == 1
    assert not ax.lines


def test_plot_graph_deduplicates_edges(ax):
    """The graph is directed; u->v and v->u must not both be drawn."""
    graph = {(0.0, 0.0): {(1.0, 1.0): 1.0}, (1.0, 1.0): {(0.0, 0.0): 1.0}}
    plot_graph(ax, graph)
    assert len(ax.collections[0].get_segments()) == 1
