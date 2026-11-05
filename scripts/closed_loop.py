#!/usr/bin/env python3
"""Deterministic closed-loop run of the graph-planner stack, without ROS spinning.

    python3 scripts/closed_loop.py my_course
    python3 scripts/closed_loop.py my_course --plot run.png --trace run.csv
    python3 scripts/closed_loop.py my_course --start 149 -128 --heading 20 --skip-via 3

The live system (run_simulation_graph.py) steps physics by wall-clock time and
passes messages between processes, so two runs of one scenario differ, and how
much depends on CPU load. That makes it useless for checking a steering change:
a crash may or may not happen again.

This runs the *same node classes* — RouteFollower, GraphRoutePlanner and
ControllerNode, with their shipped parameter files — inside one process on a
simulated clock:

  * the physics advances one MuJoCo timestep at a time;
  * every node timer fires on its own period of simulated time;
  * every message a node publishes is delivered, in order, to the callbacks
    subscribed to that topic — no DDS, no threads;
  * ``datetime.now()`` inside the nodes reads the simulated clock, so their
    rate limits (min_time_between_replanning, ...) run in simulated time too.

The wiring is read from the nodes themselves (their publishers, subscriptions
and timers), so it follows the real code as that changes. Only the simulation
node is replaced, by a few lines that do what it does each tick: step physics,
publish /sense/boat_info, /sense/wind and /sense/collision, and apply
/control/rudder and /control/sail.

Same scenario and options in, same run out. What it does not model: message
latency and planning time (a route is delivered the instant it is planned,
while the live planner blocks for ~0.1-0.6 s per leg as the boat sails on).

Requires the ROS environment: ``source install/setup.bash``. The nodes are
imported from src/, so uncommitted changes to them are what runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime as _datetime, timedelta
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
# The working tree, not the colcon install, so local edits are what runs.
for _pkg in ("simulation", "controller", "route_follower", "graph_route_planner"):
    sys.path.insert(0, str(REPO / "src" / _pkg))
sys.path.insert(0, str(REPO / "scripts"))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from shapely.geometry import Point  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from boat_msgs.msg import BoatInfo, EnvironmentCreation, Location, Route, Wind  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402

import controller.node as controller_module  # noqa: E402
import graph_route_planner.node as planner_module  # noqa: E402
import route_follower.node as follower_module  # noqa: E402
from graph_route_planner.map_loader import load_map  # noqa: E402
from scenario import Scenario, load_scenario  # noqa: E402
from simulation.build_env import land_from_water, world_half_size_for  # noqa: E402
from simulation.sailboat_simulation import SailboatSimulation  # noqa: E402

#: The live simulation node's timer rate: physics catches up, then it publishes.
SIM_PUBLISH_RATE = 60.0
#: How often the trace records the boat's state.
TRACE_RATE = 10.0
#: Simulated time all node clocks start at. Arbitrary, but fixed.
EPOCH = _datetime(2000, 1, 1)


# ── simulated time ────────────────────────────────────────────────────────────

class SimClock:
    """Seconds of simulated time since the run started."""

    def __init__(self):
        self.t = 0.0


def _clocked_datetime(clock: SimClock) -> type:
    """A datetime class whose now() is the simulated clock.

    Installed as the ``datetime`` name inside the node modules, so their rate
    limits and log throttles count simulated, not wall, seconds.
    """

    class ClockedDatetime(_datetime):
        @classmethod
        def now(cls, tz=None):
            return EPOCH + timedelta(seconds=clock.t)

    return ClockedDatetime


# ── message passing ───────────────────────────────────────────────────────────

class Bus:
    """In-process topic delivery: published messages queue, then go out in order.

    Queuing (rather than calling subscribers inside publish) keeps a node's
    callback from re-entering another node mid-update, as it cannot on ROS.
    """

    def __init__(self):
        self.subscribers: dict[str, list] = {}
        self.queue: deque = deque()
        self.taps: dict[str, list] = {}

    def subscribe(self, topic: str, callback) -> None:
        self.subscribers.setdefault(topic, []).append(callback)

    def tap(self, topic: str, callback) -> None:
        """Observe a topic from the harness (not a node)."""
        self.taps.setdefault(topic, []).append(callback)

    def publish(self, topic: str, msg) -> None:
        self.queue.append((topic, msg))

    def drain(self) -> None:
        while self.queue:
            topic, msg = self.queue.popleft()
            for callback in self.taps.get(topic, []):
                callback(msg)
            for callback in self.subscribers.get(topic, []):
                callback(msg)


@dataclass
class _Timer:
    period: float
    callback: object
    next_fire: float
    order: int


# ── the run ───────────────────────────────────────────────────────────────────

@dataclass
class LogLine:
    t: float
    node: str
    level: str
    text: str


@dataclass
class TraceRow:
    t: float
    east: float
    north: float
    heading: float
    speed: float
    desired_heading: float | None
    colliding: bool
    land_distance: float


@dataclass
class Result:
    """What happened in a run."""

    scenario: str
    completed: bool
    sim_time: float
    time_aground: float
    groundings: list[tuple[float, float, float]]   # (t, east, north) of each grounding
    min_land_distance: float
    replans: list[tuple[float, str]]               # (t, reason)
    trace: list[TraceRow]
    routes: list[tuple[float, list[tuple[float, float]]]]
    log: list[LogLine]
    final_state: tuple[float, float, float]        # east, north, heading

    def summary(self) -> str:
        lines = [
            f"Scenario {self.scenario}: "
            + (f"completed in {self.sim_time:.1f} s" if self.completed
               else f"NOT completed after {self.sim_time:.1f} s"),
            f"  groundings: {len(self.groundings)}, {self.time_aground:.1f} s aground in total",
        ]
        for t, e, n in self.groundings:
            lines.append(f"    t={t:7.1f} s at ({e:.1f}, {n:.1f})")
        lines.append(f"  closest approach to land: {self.min_land_distance:.2f} m "
                     f"(boat reference point; 0 means on the shoreline)")
        lines.append(f"  routes published: {len(self.replans)}")
        for t, reason in self.replans:
            lines.append(f"    t={t:7.1f} s  {reason}")
        e, n, h = self.final_state
        lines.append(f"  final position ({e:.2f}, {n:.2f}), heading {h:.1f}°")
        return "\n".join(lines)


def _parameters(package: str, node_name: str, overrides: dict | None = None) -> list[Parameter]:
    """A node's parameters from its package's parameters.yaml in src/."""
    path = REPO / "src" / package / "resource" / "parameters.yaml"
    values = yaml.safe_load(path.read_text())[node_name]["ros__parameters"]
    values.update(overrides or {})
    return [Parameter(name, value=value) for name, value in values.items()]


class ClosedLoop:
    """One scenario, the planner/follower/controller nodes and the physics, in lockstep.

    Args:
        scenario: The scenario to sail.
        start: Override the boat's start (east, north), e.g. to begin just
            before a known trouble spot.
        heading: The boat's initial compass heading in degrees. None keeps the
            simulation's default.
        skip_via: Drop this many of the scenario's via points from the mission,
            for a start part way round the course.
        echo: Print the nodes' log lines as they happen.
        follower_overrides: route_follower parameters to change, e.g.
            {"map_path": ""} to sail without the land check.
        planner_overrides: graph_route_planner parameters to change.
    """

    def __init__(self, scenario: Scenario, start: tuple[float, float] | None = None,
                 heading: float | None = None, skip_via: int = 0, echo: bool = False,
                 follower_overrides: dict | None = None,
                 planner_overrides: dict | None = None):
        self.scenario = scenario
        self.clock = SimClock()
        self.bus = Bus()
        self.echo = echo
        self.log: list[LogLine] = []

        # The map, built exactly as the simulation node builds its world.
        self.map = load_map(scenario.map)
        half = world_half_size_for(self.map.water)
        land = land_from_water(self.map.water, half)
        self._shore = self.map.water.boundary

        boat = start or scenario.boat
        buoys = [(b.east, b.north, *b.rgb) for b in scenario.buoys]
        self.sim = SailboatSimulation(
            buoys=buoys, include_course_line=True,
            wind_direction_deg=scenario.wind_direction, wind_speed=scenario.wind_speed,
            obstacles=land, world_half_size=half,
        )
        self.sim.set_boat_position(*boat)
        if heading is not None:
            self.sim.data.qpos[self.sim.qadr_yaw] = math.radians(90.0 - heading)

        # Node modules read the simulated clock.
        clocked = _clocked_datetime(self.clock)
        self._patched = []
        for module in (planner_module, follower_module):
            self._patched.append((module, module.datetime))
            module.datetime = clocked

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        # Constructors log too (config, map): send that to the harness's log as well.
        original_get_logger = Node.get_logger
        Node.get_logger = lambda node: _Logger(self, node.get_name())
        try:
            self.nodes: list[Node] = [
                planner_module.GraphRoutePlanner(parameter_overrides=_parameters(
                    "graph_route_planner", "graph_route_planner_node",
                    {"map_path": scenario.map,
                     "grid_spacing": float(scenario.grid_spacing),
                     "max_leg_distance": float(scenario.max_leg_distance),
                     **(planner_overrides or {})})),
                follower_module.RouteFollower(parameter_overrides=_parameters(
                    "route_follower", "route_follower_node",
                    {"map_path": scenario.map, **(follower_overrides or {})})),
                controller_module.ControllerNode(parameter_overrides=_parameters(
                    "controller", "controller_node")),
            ]
        finally:
            Node.get_logger = original_get_logger
        self.timers: list[_Timer] = []
        for node in self.nodes:
            self._wire(node)

        # The simulation node's side of the bus.
        self.bus.subscribe("/control/rudder", lambda m: self.sim.set_rudder(m.angle))
        self.bus.subscribe("/control/sail", lambda m: self.sim.set_sail_angle(m.sail_angle))
        self.completed_at: float | None = None
        self.bus.tap("/planning/route_completed", self._on_completed)
        self.desired_heading: float | None = None
        self.bus.tap("/planning/desired_heading", self._on_desired)
        self.routes: list[tuple[float, list[tuple[float, float]]]] = []
        self.bus.tap("/planning/target_route", self._on_route)

        # What run_simulation_graph.py sends: the wind, then the mission.
        self.bus.publish("/env/wind", Wind(angle=float(scenario.wind_direction),
                                           speed=float(scenario.wind_speed)))
        mission = Route(waypoints=[Location(east=float(e), north=float(n))
                                   for e, n in scenario.waypoints[skip_via:]])
        self.bus.publish("/planning/mission", mission)
        self._publish_sensors()
        self.bus.drain()

    # ── wiring ────────────────────────────────────────────────────────────────

    def _wire(self, node: Node) -> None:
        """Connect a node's publishers, subscriptions, timers and logger to the harness."""
        for publisher in node.publishers:
            topic = publisher.topic_name
            publisher.publish = (lambda m, _t=topic: self.bus.publish(_t, m))
        for subscription in node.subscriptions:
            self.bus.subscribe(subscription.topic_name, subscription.callback)
        for timer in node.timers:
            period = timer.timer_period_ns / 1e9
            self.timers.append(_Timer(period, timer.callback, period, len(self.timers)))
            timer.cancel()
        name = node.get_name()
        logger = _Logger(self, name)
        node.get_logger = lambda: logger

    def _record(self, node: str, level: str, text: str) -> None:
        line = LogLine(self.clock.t, node, level, str(text))
        self.log.append(line)
        if self.echo:
            print(f"[{line.t:8.2f}] [{level}] [{node}] {line.text}")

    def _on_completed(self, msg) -> None:
        if msg.data and self.completed_at is None:
            self.completed_at = self.clock.t

    def _on_desired(self, msg) -> None:
        self.desired_heading = float(msg.heading)

    def _on_route(self, msg) -> None:
        self.routes.append((self.clock.t, [(w.east, w.north) for w in msg.waypoints]))

    # ── the simulation node's part ────────────────────────────────────────────

    def _publish_sensors(self) -> None:
        status = self.sim.get_status()
        self.bus.publish("/sense/wind", Wind(angle=status["wind_dir"], speed=status["wind_speed"]))
        self.bus.publish("/sense/boat_info", BoatInfo(
            x=status["x"], y=status["y"],
            velocity_x=status["vel_x"], velocity_y=status["vel_y"],
            velocity_magnitude=status["speed"], yaw_rate=status["yaw_rate"],
            heading=status["heading"], rudder_angle=status["rudder"],
            sail_angle=status["sail_angle"],
        ))
        self.bus.publish("/sense/collision", Bool(data=bool(status["collision"])))

    def land_distance(self, east: float, north: float) -> float:
        """Distance from a point to the shoreline; 0 on or over it."""
        p = Point(east, north)
        if not self.map.water.contains(p):
            return 0.0
        return self._shore.distance(p)

    # ── stepping ──────────────────────────────────────────────────────────────

    def run(self, max_time: float = 900.0, stop_on_completion: bool = True) -> Result:
        """Sail until the route is completed or max_time simulated seconds pass."""
        dt = self.sim.model.opt.timestep
        sensor_period = 1.0 / SIM_PUBLISH_RATE
        next_sensor = sensor_period
        trace_period = 1.0 / TRACE_RATE
        next_trace = 0.0

        trace: list[TraceRow] = []
        groundings: list[tuple[float, float, float]] = []
        time_aground = 0.0
        was_colliding = False
        min_land = math.inf
        steps = 0

        try:
            while self.clock.t < max_time:
                self.sim.step()
                steps += 1
                self.clock.t = steps * dt

                colliding = self.sim.in_collision()
                if colliding:
                    time_aground += dt
                    if not was_colliding:
                        s = self.sim.get_status()
                        groundings.append((self.clock.t, s["x"], s["y"]))
                was_colliding = colliding

                if self.clock.t + 1e-9 >= next_sensor:
                    next_sensor += sensor_period
                    self._publish_sensors()
                    self.bus.drain()

                for timer in sorted(self.timers, key=lambda tm: (tm.next_fire, tm.order)):
                    if self.clock.t + 1e-9 >= timer.next_fire:
                        timer.next_fire += timer.period
                        timer.callback()
                        self.bus.drain()

                if self.clock.t + 1e-9 >= next_trace:
                    next_trace += trace_period
                    s = self.sim.get_status()
                    land = self.land_distance(s["x"], s["y"])
                    min_land = min(min_land, land)
                    trace.append(TraceRow(self.clock.t, s["x"], s["y"], s["heading"], s["speed"],
                                          self.desired_heading, colliding, land))

                if stop_on_completion and self.completed_at is not None:
                    break
        finally:
            self.close()

        replans = []
        for line in self.log:
            if line.node == "graph_route_planner" and line.text.startswith("Published route ("):
                replans.append((line.t, line.text[len("Published route ("):].split("):")[0]))
        s = self.sim.get_status()
        return Result(
            scenario=self.scenario.name, completed=self.completed_at is not None,
            sim_time=self.completed_at if self.completed_at is not None else self.clock.t,
            time_aground=time_aground, groundings=groundings, min_land_distance=min_land,
            replans=replans, trace=trace, routes=self.routes, log=self.log,
            final_state=(s["x"], s["y"], s["heading"]),
        )

    def close(self) -> None:
        """Release the nodes and restore the node modules' datetime."""
        for module, original in self._patched:
            module.datetime = original
        self._patched = []
        for node in self.nodes:
            Node.destroy_node(node)   # not ControllerNode's: it saves a plot
        self.nodes = []
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()


class _Logger:
    """Stands in for a node's rclpy logger, recording with simulated timestamps."""

    def __init__(self, harness: ClosedLoop, node: str):
        self._harness, self._node = harness, node

    def _log(self, level):
        return lambda msg, *a, **k: self._harness._record(self._node, level, msg)

    def __getattr__(self, name):
        if name in ("debug", "info", "warn", "warning", "error", "fatal"):
            return self._log(name.upper())
        raise AttributeError(name)


# ── output ────────────────────────────────────────────────────────────────────

def write_trace(result: Result, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "east", "north", "heading", "speed", "desired_heading",
                         "colliding", "land_distance"])
        for r in result.trace:
            writer.writerow([f"{r.t:.2f}", f"{r.east:.3f}", f"{r.north:.3f}", f"{r.heading:.2f}",
                             f"{r.speed:.3f}",
                             "" if r.desired_heading is None else f"{r.desired_heading:.2f}",
                             int(r.colliding), f"{r.land_distance:.3f}"])


def plot(result: Result, sail_map, path: str, scenario: Scenario,
         zoom: tuple[float, float, float] | None = None) -> None:
    """Map, planned routes, the sailed track and where the boat ran aground."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor((0.37, 0.51, 0.27))
    for poly in getattr(sail_map.water, "geoms", [sail_map.water]):
        rings = [poly.exterior, *poly.interiors]
        verts, codes = [], []
        for ring in rings:
            xs = list(ring.coords)
            verts += xs
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(xs) - 2) + [MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=(0.16, 0.35, 0.59), lw=0))

    for i, (_, route) in enumerate(result.routes):
        xs, ys = zip(*route)
        last = i == len(result.routes) - 1
        ax.plot(xs, ys, "-", color="gold", lw=1.2 if last else 0.6, alpha=1.0 if last else 0.35)
    xs = [r.east for r in result.trace]
    ys = [r.north for r in result.trace]
    sc = ax.scatter(xs, ys, c=[r.t for r in result.trace], s=2, cmap="plasma", zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="simulated time (s)")
    for t, e, n in result.groundings:
        ax.plot(e, n, "x", color="red", ms=10, mew=2, zorder=4)
        ax.annotate(f"{t:.0f}s", (e, n), color="white", fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    vx, vy = zip(*scenario.waypoints)
    ax.plot(vx, vy, "o", color="orange", mec="black", zorder=4)

    if zoom:
        e, n, r = zoom
        ax.set_xlim(e - r, e + r)
        ax.set_ylim(n - r, n + r)
    else:
        pad = 30
        ax.set_xlim(min(xs + list(vx)) - pad, max(xs + list(vx)) + pad)
        ax.set_ylim(min(ys + list(vy)) - pad, max(ys + list(vy)) + pad)
    ax.set_aspect("equal")
    status = "completed" if result.completed else "not completed"
    ax.set_title(f"{result.scenario}: {status} in {result.sim_time:.0f} s, "
                 f"{len(result.groundings)} grounding(s), {result.time_aground:.0f} s aground")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", help="scenario name (scenarios/*.yaml) or path")
    parser.add_argument("--max-time", type=float, default=900.0,
                        help="simulated seconds before giving up (default 900)")
    parser.add_argument("--start", type=float, nargs=2, metavar=("EAST", "NORTH"),
                        help="start the boat here instead of the scenario's start")
    parser.add_argument("--heading", type=float, help="initial compass heading, degrees")
    parser.add_argument("--skip-via", type=int, default=0,
                        help="drop this many via points from the start of the mission")
    parser.add_argument("--plot", help="save a map of the run to this PNG")
    parser.add_argument("--zoom", type=float, nargs=3, metavar=("EAST", "NORTH", "RADIUS"),
                        help="limit the plot to this square")
    parser.add_argument("--trace", help="save the boat's state at 10 Hz to this CSV")
    parser.add_argument("--log", help="save the nodes' log to this file")
    parser.add_argument("--echo", action="store_true", help="print node logs as they happen")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a route_follower parameter (YAML value), repeatable; "
                             "e.g. --set map_path= to disable the land check")
    parser.add_argument("--set-planner", action="append", default=[], metavar="NAME=VALUE",
                        help="override a graph_route_planner parameter (YAML value), repeatable")
    args = parser.parse_args()

    def parse(items):
        out = {}
        for item in items:
            name, _, value = item.partition("=")
            out[name] = yaml.safe_load(value) if value else ""
        return out

    overrides = parse(args.set)
    planner_overrides = parse(args.set_planner)


    scenario = load_scenario(args.scenario)
    loop = ClosedLoop(scenario, start=tuple(args.start) if args.start else None,
                      heading=args.heading, skip_via=args.skip_via, echo=args.echo,
                      follower_overrides=overrides, planner_overrides=planner_overrides)
    sail_map = loop.map
    result = loop.run(max_time=args.max_time)
    print(result.summary())

    if args.trace:
        write_trace(result, args.trace)
    if args.log:
        Path(args.log).write_text("".join(
            f"[{l.t:8.2f}] [{l.level}] [{l.node}] {l.text}\n" for l in result.log))
    if args.plot:
        plot(result, sail_map, args.plot, scenario, tuple(args.zoom) if args.zoom else None)
    return 0 if result.completed and not result.groundings else 1


if __name__ == "__main__":
    sys.exit(main())
