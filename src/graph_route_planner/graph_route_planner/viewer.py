"""
Interactive route planner.

    python -m graph_route_planner.viewer [map.kml]

Click the water to drop a start point, click again for a goal, and the route
is planned between them. The header bar turns the wind (0-360°), retunes the
search (margin / grid / merge) and the leg-length bounds, and toggles the
search graph overlay.

Built on matplotlib's own widgets so there is no extra dependency, and so the
drawing code is shared with the static plots in `plotting.py`.
"""

import argparse
import logging
import math
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider

from graph_route_planner import DEFAULT_MAP
from shapely.geometry.base import BaseGeometry

from graph_route_planner.geometry import bbox_diagonal
from graph_route_planner.map_loader import load_kml
from graph_route_planner.planner import (
    Graph,
    NoWaterError,
    PlannerConfig,
    Position,
    plan_route,
)
from graph_route_planner.plotting import (
    LAND_COLOR,
    draw_navigable,
    draw_path,
    draw_water,
    draw_wind,
    plot_graph,
)
from graph_route_planner.sailing import SailingModel

log = logging.getLogger(__name__)

HEADER_BG = "#eceff1"

#: Tuning suited to a pond-sized map, rather than the planner's own defaults
#: which assume the larger synthetic basin.
VIEWER_DEFAULTS = PlannerConfig(margin=2.0, grid_spacing=6.0, merge_threshold=1.5)


class RoutePlannerViewer:
    """A click-to-plan window over one loaded map.

    Args:
        sail_map: The map to show.
        model: The boat's sailing model. Defaults to a northerly.
        config: Initial slider positions. Defaults to VIEWER_DEFAULTS.
    """

    # Map axes geometry, as a fraction of the figure.
    AX_X, AX_Y, AX_W, AX_H = 0.07, 0.05, 0.88, 0.79
    PAD = 0.04          # blank margin around the map, in spans
    HEADROOM = 0.18     # extra space at the top for the wind arrows

    def __init__(self, sail_map, model=None, config: PlannerConfig | None = None):
        self.map = sail_map
        self.model = model or SailingModel()
        self.config = config or VIEWER_DEFAULTS
        self.start: Position | None = None
        self.goal: Position | None = None

        # An empty path means "nothing to draw", mirroring Route: it keeps the
        # drawing code free of None checks.
        self.path: list[Position] = []
        self.graph: Graph = {}
        self.navigable: BaseGeometry | None = None
        self.status = "Click on water to set the start point."
        self.stats = ""

        # Set by slider callbacks, consumed on mouse release: sliders fire
        # continuously while dragging, and replanning per pixel would stall.
        self._dirty = False
        self._planning = False

        x0, y0, x1, y1 = sail_map.bounds
        self.span = max(x1 - x0, y1 - y0)

        # Match the window to the map's own aspect, or an equal-aspect plot
        # leaves half the width empty. Floored so the header row stays legible.
        self.view_w = (x1 - x0) + 2 * self.PAD * self.span
        self.view_h = (y1 - y0) + (self.PAD + self.HEADROOM) * self.span
        fig_h = 9.0
        fig_w = fig_h * (self.AX_H / self.AX_W) * (self.view_w / self.view_h)
        fig_w = min(max(fig_w, 9.0), 15.0)

        self.fig = plt.figure(figsize=(fig_w, fig_h))
        manager = self.fig.canvas.manager
        if manager is not None:          # None under headless backends
            manager.set_window_title(
                f"Route planner — {sail_map.name} "
                f"({len(sail_map.features)} features, "
                f"{len(sail_map.components)} water body(ies), "
                f"{sail_map.water.area:.0f} m² navigable)"
            )

        self._build_header()

        # The map gets everything the header does not.
        self.ax = self.fig.add_axes((self.AX_X, self.AX_Y, self.AX_W, self.AX_H))

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self.draw()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_header(self):
        """The toolbar across the top: six sliders, a toggle, a button."""
        bar = self.fig.add_axes((0.0, 0.87, 1.0, 0.13))
        bar.set_facecolor(HEADER_BG)
        bar.set_xticks([])
        bar.set_yticks([])
        for spine in bar.spines.values():
            spine.set_visible(False)
        bar.set_navigate(False)
        self.header = bar

        # Six sliders do not fit on one line at the window's minimum width, so
        # the bar carries two rows. Everything still lives in the header.
        top_y, bot_y, row_h, w = 0.947, 0.902, 0.020, 0.105
        self.s_wind = Slider(self.fig.add_axes((0.055, top_y, w, row_h)),
                             "wind", 0.0, 360.0,
                             valinit=self.model.wind_bearing_deg, valstep=5.0,
                             valfmt="%.0f°")
        self.s_margin = Slider(self.fig.add_axes((0.275, top_y, w, row_h)),
                               "margin", 0.5, 10.0, valinit=self.config.margin, valstep=0.5)
        self.s_grid = Slider(self.fig.add_axes((0.495, top_y, w, row_h)),
                             "grid", 2.0, 20.0, valinit=self.config.grid_spacing, valstep=0.5)

        self.s_merge = Slider(self.fig.add_axes((0.055, bot_y, w, row_h)),
                              "merge", 0.0, 5.0, valinit=self.config.merge_threshold, valstep=0.25)
        self.s_min_leg = Slider(self.fig.add_axes((0.275, bot_y, w, row_h)),
                                "min leg", 0.0, 20.0,
                                valinit=self.config.min_leg_distance, valstep=0.5)

        # The slider tops out at the map's own span, because a longer leg than
        # that cannot exist — so it opens at "no truncation", and dragging it
        # down is an explicit trade of accuracy for speed. Rounded up onto the
        # step grid, or valstep would snap below the span and clip legs again.
        step = 5.0
        span = math.ceil(bbox_diagonal(self.map.water) / step) * step
        init = self.config.max_leg_distance
        init = span if init is None else min(init, span)
        self.s_max_leg = Slider(self.fig.add_axes((0.495, bot_y, w, row_h)),
                                "max leg", step, span, valinit=init, valstep=step)

        # Wind rebuilds the model; the rest only retune the search.
        self.s_wind.on_changed(self._on_wind)
        self.sliders = (self.s_wind, self.s_margin, self.s_grid,
                        self.s_merge, self.s_min_leg, self.s_max_leg)
        for s in self.sliders[1:]:
            s.on_changed(self._mark_dirty)
        for s in self.sliders:
            s.label.set_fontsize(9)
            s.valtext.set_fontsize(9)

        check_ax = self.fig.add_axes((0.685, 0.900, 0.115, 0.065))
        check_ax.set_facecolor(HEADER_BG)
        for spine in check_ax.spines.values():
            spine.set_visible(False)
        self.check = CheckButtons(check_ax, ["show graph"], [False])
        self.check.on_clicked(lambda _label: self.draw())

        self.btn_reset = Button(self.fig.add_axes((0.845, 0.915, 0.07, 0.035)), "Clear")
        self.btn_reset.on_clicked(self._on_reset)

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def show_graph(self) -> bool:
        return self.check.get_status()[0]

    def _mark_dirty(self, _val):
        self._dirty = True

    def _config_from_sliders(self) -> PlannerConfig:
        """Read the header sliders into a PlannerConfig.

        min leg is clamped below max leg: PlannerConfig rejects an empty range,
        and the sliders are independent, so a user can otherwise drag them past
        each other and crash the window.
        """
        max_leg = self.s_max_leg.val
        min_leg = min(self.s_min_leg.val, max_leg - 0.5)
        return PlannerConfig(
            margin=self.s_margin.val,
            grid_spacing=self.s_grid.val,
            merge_threshold=self.s_merge.val,
            max_leg_distance=max_leg,
            min_leg_distance=max(0.0, min_leg),
        )

    def _on_wind(self, bearing: float):
        """Rebuild the sailing model when the wind slider moves.

        Wind is a model parameter rather than a global, so turning the wind is
        just a new SailingModel; the no-go zone carries over.
        """
        self.model = SailingModel.from_bearing(bearing, no_go_deg=self.model.no_go_deg)
        self._dirty = True

    def _clear_route(self):
        """Drop the last plan, keeping start/goal alone."""
        self.path = []
        self.graph = {}
        self.navigable = None
        self.stats = ""

    def _on_reset(self, _event):
        self.start = None
        self.goal = None
        self._clear_route()
        self.status = "Click on water to set the start point."
        self.draw()

    def _navigating(self) -> bool:
        """True while the toolbar's pan/zoom is active, so clicks aren't points."""
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def _on_click(self, event):
        if self._planning or event.inaxes is not self.ax or self._navigating():
            return
        if event.xdata is None or event.ydata is None:
            return

        point = (float(event.xdata), float(event.ydata))
        if self.map.component_for(point) is None:
            self.status = "That is land — click on water."
            self.draw()
            return

        if self.start is None or self.goal is not None:
            # Starting a fresh pair.
            self.start, self.goal = point, None
            self._clear_route()
            self.status = "Now click the goal."
            self.draw()
        else:
            self.goal = point
            self.replan()

    def _on_release(self, _event):
        if self._dirty and not self._planning:
            self._dirty = False
            self.replan()

    # ── planning ──────────────────────────────────────────────────────────────

    def replan(self):
        start, goal = self.start, self.goal
        if start is None or goal is None:
            self.draw()
            return
        # Re-entrancy guard. Planning runs on the UI thread, so any event
        # dispatched mid-plan must not start a second one on top of it.
        if self._planning:
            return

        self._planning = True
        try:
            self._plan_now(start, goal)
        finally:
            self._planning = False

    def _plan_now(self, start: Position, goal: Position):
        config = self._config_from_sliders()

        t0 = time.time()
        try:
            route = plan_route(self.map.water, start, goal,
                               model=self.model, config=config)
        except NoWaterError as e:
            self._clear_route()
            self.status = f"No route: {e}"
            self.draw()
            return

        self.path = route.waypoints
        self.navigable = route.navigable
        self.graph = route.graph

        elapsed = time.time() - t0
        if route.found:
            edges = sum(len(v) for v in route.graph.values()) // 2
            self.status = f"Route: {len(route.waypoints)} waypoints, {route.duration:.2f} time units"
            self.stats = f"{len(route.graph)} nodes · {edges} edges · planned in {elapsed:.2f}s"
        else:
            self.status = "No route found — try a smaller margin."
            self.stats = f"{len(route.graph)} nodes · planned in {elapsed:.2f}s"
        self.draw()

    # ── drawing ───────────────────────────────────────────────────────────────

    def draw(self):
        # Preserve the user's zoom across redraws.
        had_view = self.ax.has_data()
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()

        draw_water(self.ax, self.map.water)

        if self.show_graph and self.graph:
            plot_graph(self.ax, self.graph)
        if self.navigable is not None:
            draw_navigable(self.ax, self.navigable)

        draw_wind(self.ax, self.map.bounds, self.span, self.model)
        draw_path(self.ax, self.path, self.span)

        if self.start:
            self.ax.plot(*self.start, "go", markersize=10, zorder=8, label="Start")
        if self.goal:
            self.ax.plot(*self.goal, "ro", markersize=10, zorder=8, label="Goal")

        self.ax.set_aspect("equal")
        self.ax.set_facecolor(LAND_COLOR)
        title = f"{self.status}          {self.stats}" if self.stats else self.status
        self.ax.set_title(title, fontsize=10)
        if self.start or self.goal:
            self.ax.legend(loc="lower left", fontsize=8)

        if had_view:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            x0, y0, x1, y1 = self.map.bounds
            pad = self.PAD * self.span
            self.ax.set_xlim(x0 - pad, x1 + pad)
            self.ax.set_ylim(y0 - pad, y1 + self.HEADROOM * self.span)

        self.fig.canvas.draw_idle()


def main():
    """Entry point: ``python -m graph_route_planner.viewer [map.kml]``."""
    ap = argparse.ArgumentParser(description="Interactive sailing route planner")
    ap.add_argument("kml", nargs="?", default=str(DEFAULT_MAP), help="KML map to load")
    ap.add_argument("--wind", type=float, default=0.0,
                    help="bearing the wind blows FROM, degrees (0 = north)")
    ap.add_argument("--margin", type=float, default=VIEWER_DEFAULTS.margin)
    ap.add_argument("--grid", type=float, default=VIEWER_DEFAULTS.grid_spacing)
    ap.add_argument("--merge", type=float, default=VIEWER_DEFAULTS.merge_threshold)
    ap.add_argument("--min-leg", type=float, default=VIEWER_DEFAULTS.min_leg_distance,
                    help="shortest leg to consider, metres (0 disables)")
    ap.add_argument("--max-leg", type=float, default=None,
                    help="longest leg to consider, metres (default: the map's span)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log planner internals")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(message)s")

    sail_map = load_kml(args.kml)
    model = SailingModel.from_bearing(args.wind)
    print(f"Loaded '{sail_map.name}': {len(sail_map.features)} features, "
          f"{len(sail_map.components)} navigable body(ies)")
    print("Click water to set start, click again for goal.")

    config = PlannerConfig(margin=args.margin, grid_spacing=args.grid,
                           merge_threshold=args.merge,
                           min_leg_distance=args.min_leg,
                           max_leg_distance=args.max_leg)
    RoutePlannerViewer(sail_map, model=model, config=config)
    plt.show()


if __name__ == "__main__":
    main()
