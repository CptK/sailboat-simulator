#!/usr/bin/env python3
"""Interactive editor for scenario files: place things on the map with the mouse.

    python scripts/scenario_editor.py landgraben                  # edit scenarios/landgraben.yaml
    python scripts/scenario_editor.py my_course --map kornaten    # new scenario on a map
    python scripts/scenario_editor.py path/to/course.yaml         # any scenario file

Mouse
    click            place a boat / waypoint / buoy (pick which on the right)
    shift+click      insert a waypoint into the nearest leg of the course
    drag             move anything
    right-click      delete what is under the cursor (on a Mac: ctrl+click)

Keys
    b / w / m        place the boat / waypoints / marks (buoys)
    d, delete        delete what is under the cursor
    u, ctrl+z        undo
    ctrl+s           save
    o / p            zoom / pan (matplotlib toolbar), again to leave

Anything on land is drawn as a red cross and listed in the status line. Tick
"show route" to see the route the planner would sail; it updates after every
edit (on a large map the first route takes a few seconds while the planner
builds its graph). Save writes scenarios/<name>.yaml, which
scripts/run_simulation_graph.py --scenario <name> runs.
"""
import argparse
import copy
import math
import sys
from pathlib import Path

import matplotlib
from matplotlib.widgets import TextBox   # widgets only; does not pick a backend

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src" / "graph_route_planner"))

from scenario import (  # noqa: E402
    COLORS, DEFAULT_PLANNING_DELAY, LAND_TOLERANCE, PLANNER_MARGIN, SCENARIOS_DIR, Buoy,
    Scenario,
    ScenarioError, check_positions, load_scenario, load_scenario_map, resolve_scenario,
    save_scenario)

#: How close (in screen pixels) a click must be to grab a marker.
PICK_PX = 12
#: Planning delay for new scenarios on large maps, whose graph takes a while.
LARGE_MAP_PLANNING_DELAY = 20.0

MODES = ("boat", "waypoint", "buoy")
MODE_KEYS = {"b": "boat", "w": "waypoint", "m": "buoy"}


def _free_keys():
    """Take the keys this editor uses away from matplotlib's defaults."""
    import matplotlib as mpl

    mpl.rcParams["keymap.save"] = []
    mpl.rcParams["keymap.back"] = [k for k in mpl.rcParams["keymap.back"]
                                   if k not in ("backspace",)]
    for name in ("keymap.grid", "keymap.grid_minor"):
        mpl.rcParams[name] = []


class ScenarioEditor:
    """The editor window. Usable headless: the handlers take plain event objects."""

    def __init__(self, scenario: Scenario, sail_map, path: Path):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        from render_maps import render_map

        _free_keys()
        self.sc = scenario
        self.map = sail_map
        self.path = Path(path)
        self.mode = "waypoint"
        self.color = "orange"
        self.undo_stack: list = []
        self.dirty = False
        self.drag = None            # (kind, index) while dragging
        self._drag_moved = False
        self._press_at = None       # where a press that grabbed a marker happened
        self._planner = None
        self.route: list = []
        self.route_problem = ""
        self.message = ""
        self.dynamic: list = []

        self.fig = plt.figure(figsize=(13, 9.5))
        self.ax = self.fig.add_axes((0.06, 0.19, 0.65, 0.76))
        x0, y0, x1, y1 = sail_map.bounds
        render_map(self.ax, sail_map, PLANNER_MARGIN, "")
        self.span = max(x1 - x0, y1 - y0)

        # Right-hand panel.
        px, pw = 0.75, 0.22
        self.fig.text(px, 0.925, "Place", fontsize=11, weight="bold")
        self.w_mode = RadioButtons(self.fig.add_axes((px, 0.80, pw, 0.11)),
                                   ("boat (b)", "waypoint (w)", "buoy (m)"), active=1)
        self.w_mode.on_clicked(lambda label: self.set_mode(label.split()[0]))
        self.fig.text(px, 0.775, "Buoy colour", fontsize=11, weight="bold")
        self.w_color = RadioButtons(self.fig.add_axes((px, 0.56, pw, 0.20)), tuple(COLORS),
                                    active=0)
        for label, rgb in zip(self.w_color.labels, COLORS.values()):
            label.set_color(rgb if rgb != (1.0, 1.0, 1.0) else (0.6, 0.6, 0.6))
        self.w_color.on_clicked(self.set_color)

        self.fig.text(px, 0.525, "Wind", fontsize=11, weight="bold")
        self.w_wind_dir = Slider(self.fig.add_axes((px + 0.05, 0.485, pw - 0.09, 0.025)),
                                 "from °", 0, 359, valinit=scenario.wind_direction % 360,
                                 valstep=1)
        self.w_wind_speed = Slider(self.fig.add_axes((px + 0.05, 0.445, pw - 0.09, 0.025)),
                                   "m/s", 0, 15, valinit=scenario.wind_speed, valstep=0.5)
        self.w_wind_dir.on_changed(self._on_wind)
        self.w_wind_speed.on_changed(self._on_wind)

        self.w_route = CheckButtons(self.fig.add_axes((px, 0.37, pw, 0.05)), ["show route"],
                                    [False])
        self.w_route.on_clicked(lambda _label: self.refresh(replan=True))

        self.w_undo = Button(self.fig.add_axes((px, 0.30, pw / 2 - 0.005, 0.045)), "Undo (u)")
        self.w_undo.on_clicked(lambda _e: self.undo())
        self.w_save = Button(self.fig.add_axes((px + pw / 2 + 0.005, 0.30, pw / 2 - 0.005,
                                                0.045)), "Save (ctrl+s)")
        self.w_save.on_clicked(lambda _e: self.save())

        self.help = self.fig.text(
            px, 0.07,
            "click         place\n"
            "shift+click   insert waypoint\n"
            "drag          move\n"
            "right-click   delete (or d)\n"
            "u / ctrl+z    undo\n"
            "ctrl+s        save\n"
            "o / p         zoom / pan",
            fontsize=9, family="monospace", va="bottom", color="#444", linespacing=1.4)

        self.w_desc = _text_box(self.fig.add_axes((0.12, 0.02, 0.59, 0.035)), "description ",
                              initial=scenario.description)
        self.w_desc.on_submit(self._on_description)
        self.w_desc.text_disp.set_clip_on(True)      # long descriptions stay in the box
        self.status = self.fig.text(0.02, 0.07, "", fontsize=9, va="bottom")

        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("key_press_event", self._on_key)
        canvas.mpl_connect("close_event", self._on_close)
        self.refresh()

    # ── state ──────────────────────────────────────────────────────────────────

    def _snapshot(self):
        sc = self.sc
        return copy.deepcopy((sc.boat, sc.waypoints, sc.buoys, sc.description))

    def _push_undo(self):
        self.undo_stack.append(self._snapshot())
        del self.undo_stack[:-100]
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            self.message = "Nothing to undo."
        else:
            sc = self.sc
            sc.boat, sc.waypoints, sc.buoys, sc.description = self.undo_stack.pop()
            self.w_desc.set_val(sc.description)
            self.message = "Undone."
        self.refresh(replan=True)

    def set_mode(self, mode: str):
        self.mode = mode
        if self._mode_index() != MODES.index(mode):
            self.w_mode.set_active(MODES.index(mode))   # calls back here; harmless
        self.message = f"Click to place: {mode}."
        self.refresh()

    def _mode_index(self) -> int:
        return [label.get_text().split()[0] for label in self.w_mode.labels].index(
            self.w_mode.value_selected.split()[0])

    def set_color(self, color: str):
        self.color = color
        self.mode = "buoy"
        if self._mode_index() != MODES.index("buoy"):
            self.w_mode.set_active(MODES.index("buoy"))
        self.message = f"Click to place a {color} buoy."
        self.refresh()

    # ── editing ────────────────────────────────────────────────────────────────

    def item_at(self, x: float, y: float):
        """The marker nearest (x, y) within PICK_PX pixels: (kind, index) or None.

        On a tie the kind being placed wins, so a buoy marking a via point can
        still be grabbed in buoy mode, and the waypoint in waypoint mode.
        """
        to_px = self.ax.transData.transform
        cx, cy = to_px((x, y))
        candidates = [("boat", None, self.sc.boat)]
        candidates += [("waypoint", i, p) for i, p in enumerate(self.sc.waypoints)]
        candidates += [("buoy", i, (b.east, b.north)) for i, b in enumerate(self.sc.buoys)]
        best = None
        for kind, index, pos in candidates:
            px, py = to_px(pos)
            d = math.hypot(px - cx, py - cy)
            if d <= PICK_PX:
                rank = (round(d / 3), 0 if kind == self.mode else 1)
                if best is None or rank < best[0]:
                    best = (rank, (kind, index))
        return best[1] if best else None

    def place(self, x: float, y: float):
        self._push_undo()
        if self.mode == "boat":
            self.sc.boat = (x, y)
            self.message = f"Boat start at ({x:.1f}, {y:.1f})."
        elif self.mode == "waypoint":
            self.sc.waypoints.append((x, y))
            self.message = f"Waypoint {len(self.sc.waypoints)} at ({x:.1f}, {y:.1f})."
        else:
            self.sc.buoys.append(Buoy(x, y, COLORS[self.color], self.color))
            self.message = f"{self.color.capitalize()} buoy at ({x:.1f}, {y:.1f})."
        self.refresh(replan=True)

    def insert_waypoint(self, x: float, y: float):
        """Insert a waypoint into the leg of the course nearest to (x, y)."""
        pts = [self.sc.boat, *self.sc.waypoints]
        if len(pts) < 2:
            return self.place_waypoint_at_end(x, y)
        best, at = None, len(self.sc.waypoints)
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            d = _distance_to_segment((x, y), a, b)
            if best is None or d < best:
                best, at = d, i
        self._push_undo()
        self.sc.waypoints.insert(at, (x, y))
        self.message = f"Inserted waypoint {at + 1} at ({x:.1f}, {y:.1f})."
        self.refresh(replan=True)

    def place_waypoint_at_end(self, x, y):
        mode, self.mode = self.mode, "waypoint"
        self.place(x, y)
        self.mode = mode

    def move(self, item, x: float, y: float):
        kind, index = item
        if kind == "boat":
            self.sc.boat = (x, y)
        elif kind == "waypoint":
            self.sc.waypoints[index] = (x, y)
        else:
            b = self.sc.buoys[index]
            self.sc.buoys[index] = Buoy(x, y, b.rgb, b.color)

    def delete(self, item):
        kind, index = item
        if kind == "boat":
            self.message = "The boat cannot be deleted; drag it or click to place it."
            return self.refresh()
        self._push_undo()
        if kind == "waypoint":
            del self.sc.waypoints[index]
            self.message = f"Deleted waypoint {index + 1}."
        else:
            b = self.sc.buoys.pop(index)
            self.message = f"Deleted {b.color or ''} buoy.".replace("  ", " ")
        self.refresh(replan=True)

    def save(self) -> bool:
        self._on_description(self.w_desc.text)
        try:
            save_scenario(self.sc, self.path)
        except (ScenarioError, OSError) as err:
            self.message = f"Not saved: {err}"
            self.refresh()
            return False
        self.dirty = False
        errors, _ = check_positions(self.sc, self.map)
        self.message = (f"Saved {_short(self.path)}" +
                        (" — but it cannot be run until the errors are fixed." if errors else
                         f". Run: python scripts/run_simulation_graph.py "
                         f"--scenario {self.path.stem}"))
        self.refresh()
        return True

    # ── events ─────────────────────────────────────────────────────────────────

    def _navigating(self) -> bool:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self._navigating():
            return
        x, y = event.xdata, event.ydata
        item = self.item_at(x, y)
        button = getattr(event, "button", 1)
        if button == 3 or (button == 1 and getattr(event, "key", None) == "control"):
            if item:
                self.delete(item)
            return
        if button != 1:
            return
        if getattr(event, "key", None) == "shift":
            return self.insert_waypoint(x, y)
        if item:
            # Grab it; whether this was a drag or a click is decided on release.
            self.drag, self._drag_moved, self._press_at = item, False, (x, y)
        else:
            self.place(x, y)

    def _on_motion(self, event):
        if self.drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        if not self._drag_moved:
            self._push_undo()
            self._drag_moved = True
        self.move(self.drag, event.xdata, event.ydata)
        self.refresh(replan=False)

    def _on_release(self, _event):
        if self.drag is not None:
            kind, index = self.drag
            self.drag = None
            if self._drag_moved:
                self.message = f"Moved {kind}{'' if index is None else f' {index + 1}'}."
                self.refresh(replan=True)
            elif kind != self.mode:
                # A plain click next to a marker of another kind: place, e.g. a
                # buoy beside a waypoint. Grabbing needs a drag then.
                self.place(*self._press_at)

    def _on_key(self, event):
        if self.w_desc.capturekeystrokes:      # typing into the description
            return
        key = event.key or ""
        if key in MODE_KEYS:
            self.set_mode(MODE_KEYS[key])
        elif key in ("u", "ctrl+z", "cmd+z"):
            self.undo()
        elif key in ("ctrl+s", "cmd+s"):
            self.save()
        elif key in ("d", "delete", "backspace"):
            if event.inaxes is self.ax and event.xdata is not None:
                item = self.item_at(event.xdata, event.ydata)
                if item:
                    self.delete(item)

    def _on_wind(self, _value):
        self.sc.wind_direction = float(self.w_wind_dir.val)
        self.sc.wind_speed = float(self.w_wind_speed.val)
        self.dirty = True
        self.refresh(replan=True)

    def _on_description(self, text: str):
        text = " ".join(text.split())
        if text != self.sc.description:
            self._push_undo()
            self.sc.description = text

    def _on_close(self, _event):
        if self.dirty:
            print(f"Closed with unsaved changes; {self.path} was not updated.")

    # ── planning ───────────────────────────────────────────────────────────────

    @property
    def show_route(self) -> bool:
        return bool(self.w_route.get_status()[0])

    def plan(self):
        """The planner's route through the course, or a reason it has none."""
        from graph_route_planner.planner import CachedPlanner, NoWaterError, PlannerConfig
        from graph_route_planner.sailing import SailingModel

        self.route, self.route_problem = [], ""
        if not self.sc.waypoints:
            return
        config = PlannerConfig(margin=PLANNER_MARGIN, grid_spacing=self.sc.grid_spacing,
                               merge_threshold=1.5, segment_cost=1.0,
                               max_leg_distance=self.sc.max_leg_distance or None)
        if self._planner is None or self._planner.config != config:
            self.status.set_text("Building the planner's graph for this map "
                                 "(once; a few seconds on large maps)…")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._planner = CachedPlanner(self.map.water, config)
        model = SailingModel.from_bearing(self.sc.wind_direction, no_go_deg=50)
        pts = [self.sc.boat, *self.sc.waypoints]
        route = [pts[0]]
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            try:
                leg = self._planner.plan(a, b, model)
            except NoWaterError as err:
                self.route_problem = f"leg {i + 1}: {err}"
                break
            if not leg.found:
                self.route_problem = f"no route for leg {i + 1} (to waypoint {i + 1})"
                break
            route += list(leg.waypoints)[1:]
        self.route = route

    # ── drawing ────────────────────────────────────────────────────────────────

    def on_land(self, x: float, y: float) -> bool:
        """Whether a point is on land, with the same tolerance as check_positions."""
        from shapely.geometry import Point

        p = Point(x, y)
        return not self.map.water.contains(p) and self.map.water.distance(p) > LAND_TOLERANCE

    def refresh(self, replan: bool = False):
        errors, warnings = check_positions(self.sc, self.map)
        # Buoys do not affect the route; only a course point on land blocks it.
        course_on_land = any(self.on_land(*p) for p in [self.sc.boat, *self.sc.waypoints])
        if not self.show_route or course_on_land:
            self.route, self.route_problem = [], ""
        elif replan or not self.route:
            self.plan()
        self._draw(errors, warnings, course_on_land)

    def _draw(self, errors, warnings, course_on_land=False):
        for artist in self.dynamic:
            artist.remove()
        self.dynamic = []
        ax, add = self.ax, self.dynamic.append
        sc = self.sc

        if self.route:
            rx, ry = zip(*self.route)
            add(ax.plot(rx, ry, "-", color="#c0392b", lw=1.6, zorder=6)[0])
        course = [sc.boat, *sc.waypoints]
        if len(course) > 1:
            cx, cy = zip(*course)
            add(ax.plot(cx, cy, "k--", lw=0.8, alpha=0.6, zorder=5)[0])
        for b in sc.buoys:
            add(ax.plot(b.east, b.north, "o", color=b.rgb, mec="black", ms=9, zorder=7)[0])
        for i, (x, y) in enumerate(sc.waypoints, 1):
            add(ax.plot(x, y, "o", mfc="none", mec="black", ms=13, mew=1.0, zorder=8)[0])
            add(ax.annotate(str(i), (x, y), xytext=(10, 10), textcoords="offset points",
                            fontsize=8, ha="center", va="center", zorder=9,
                            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black",
                                      lw=0.6)))
        add(ax.plot(*sc.boat, "^", color="black", ms=12, zorder=9)[0])
        # Anything on land: a red cross on top.
        bad = [p for p in [sc.boat, *sc.waypoints, *((b.east, b.north) for b in sc.buoys)]
               if self.on_land(*p)]
        if bad:
            bx, by = zip(*bad)
            add(ax.plot(bx, by, "x", color="red", ms=14, mew=3, zorder=10)[0])
        add(self._wind_arrow())

        title = (f"{sc.name}{' *' if self.dirty else ''}  —  map {sc.map}  —  "
                 f"{len(sc.waypoints)} waypoints, {len(sc.buoys)} buoys")
        ax.set_title(title, fontsize=10)

        lines = []
        if self.message:
            lines.append(self.message)
        if errors:
            lines.append("On land: " + "; ".join(errors))
        if warnings:
            lines.append("Warning: " + "; ".join(warnings))
        if self.route_problem:
            lines.append(f"Route: {self.route_problem}")
        elif self.route:
            length = sum(math.dist(p, q) for p, q in zip(self.route, self.route[1:]))
            lines.append(f"Planned route: {length:.0f} m, {len(self.route)} points.")
        elif self.show_route and course_on_land:
            lines.append("Route: move the boat / waypoints off the land first.")
        import textwrap
        wrapped = [w for line in lines for w in textwrap.wrap(line, 150)]
        self.status.set_text("\n".join(wrapped[-4:]))
        self.status.set_color("red" if errors or self.route_problem else "black")
        self.fig.canvas.draw_idle()

    def _wind_arrow(self):
        x0, y0, x1, y1 = self.map.bounds
        span = self.span
        frm = math.radians(self.sc.wind_direction)
        tail = (x0 + 0.08 * span, y1 - 0.08 * span)
        head = (tail[0] - 0.06 * span * math.sin(frm), tail[1] - 0.06 * span * math.cos(frm))
        return self.ax.annotate(
            f"wind {self.sc.wind_speed:g} m/s", xy=head, xytext=tail, zorder=9,
            fontsize=8, ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0))


class ResizeSafeTextBox(TextBox):
    """A matplotlib TextBox that survives window resizes.

    matplotlib 3.11's TextBox handles resize events through a wrapper that reads
    event.inaxes, which resize events do not have, so every resize of the
    window prints an AttributeError traceback. The handler only stops typing;
    this does the same without the wrapper.
    """

    def _resize(self, event):
        self.stop_typing()


def _text_box(ax, label, initial=""):
    return ResizeSafeTextBox(ax, label, initial=initial)


def _short(path: Path) -> str:
    """A path relative to the repository when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _distance_to_segment(p, a, b) -> float:
    (px, py), (ax_, ay), (bx, by) = p, a, b
    dx, dy = bx - ax_, by - ay
    if dx == dy == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((px - ax_) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, (ax_ + t * dx, ay + t * dy))


def new_scenario(name: str, map_name: str, sail_map) -> Scenario:
    """An empty scenario: boat in the middle of the largest water, no course yet."""
    from graph_route_planner.viewer import search_defaults

    water = sail_map.components[0]
    start = water.representative_point()
    grid, max_leg = search_defaults(sail_map, None, None)
    return Scenario(
        name=name, map=map_name, boat=(round(start.x, 1), round(start.y, 1)),
        wind_direction=0.0, wind_speed=4.0, waypoints=[], buoys=[],
        grid_spacing=float(grid), max_leg_distance=float(max_leg or 0.0),
        planning_delay=LARGE_MAP_PLANNING_DELAY if max_leg else DEFAULT_PLANNING_DELAY)


def open_scenario(spec: str, map_name: str | None):
    """Load an existing scenario, or start a new one; returns (scenario, map, path)."""
    try:
        path = resolve_scenario(spec)
        exists = True
    except ScenarioError:
        path = Path(spec) if spec.endswith(".yaml") else SCENARIOS_DIR / f"{spec}.yaml"
        exists = False

    if exists:
        if map_name:
            raise SystemExit(f"Scenario '{spec}' already exists ({path}); open it without "
                             f"--map to edit it, or choose another name.")
        scenario = load_scenario(str(path))
        print(f"Editing {path}")
    else:
        if not map_name:
            raise SystemExit(f"No scenario '{spec}'. To create it, name its map: "
                             f"scenario_editor.py {spec} --map <map>")
        scenario = Scenario(name=path.stem, map=map_name, boat=(0.0, 0.0),
                            wind_direction=0.0, wind_speed=4.0, waypoints=[])
        print(f"New scenario {path} on map {map_name}")
    sail_map = load_scenario_map(scenario)
    if not exists:
        scenario = new_scenario(path.stem, map_name, sail_map)
    return scenario, sail_map, path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", help="scenario name (scenarios/<name>.yaml) or path")
    parser.add_argument("--map", help="create a new scenario on this map (catalog name)")
    args = parser.parse_args()

    scenario, sail_map, path = open_scenario(args.scenario, args.map)
    import matplotlib.pyplot as plt

    editor = ScenarioEditor(scenario, sail_map, path)
    if not scenario.waypoints:
        editor.message = "Click on the water to add waypoints; pick 'boat' to move the start."
        editor.refresh()
    plt.show()


if __name__ == "__main__":
    if "--help" not in sys.argv and "-h" not in sys.argv:
        matplotlib.rcParams["toolbar"] = "toolbar2"
    main()
