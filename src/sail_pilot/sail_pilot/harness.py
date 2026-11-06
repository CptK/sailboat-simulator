"""The run: physics, the controller node and a pilot, on simulated time.

The controller node runs exactly as in the live system, on an in-process bus
(scripts/closed_loop.py's Bus, SimClock and timers). Tacks are carried out by
the autopilot here, not by the pilot.
"""

from __future__ import annotations

import concurrent.futures
import math
import statistics
import time
from dataclasses import dataclass, field

import mujoco
import rclpy
from rclpy.node import Node

import controller.node as controller_module
from boat_msgs.msg import BoatInfo, Heading, Wind
from closed_loop import Bus, SimClock, _parameters, _Timer
from graph_route_planner.map_loader import load_map
from scenario import Scenario
from simulation.build_env import land_from_water, world_half_size_for
from simulation.recording import Recording, Scene
from simulation.sailboat_simulation import ONBOARD_CAMERA, SailboatSimulation
from std_msgs.msg import Bool

from .observation import Decision, Observation
from .render import (CHART_SIZE, CHART_VIEW, draw_bearing_scale, draw_course_line,
                     draw_distance_marks, draw_north, draw_scale_and_position, draw_wind,
                     jpeg_bytes, water_distance_rows)
from .sailing import crosses_wind, guard_heading, wrap180
from .task import Task

#: The live simulation node's sensor rate, as in closed_loop.py.
SENSOR_RATE = 60.0
TRACE_RATE = 10.0
#: Recorded frames per simulated second, and the video settings, as the
#: simulation node's defaults (src/simulation/simulation/node.py).
RECORD_RATE = 30.0
VIDEO_SETTINGS = {"fps": 30, "width": 1280, "height": 720, "onboard_inset_scale": 0.22,
                  "minimap_inset_scale": 0.22, "minimap_view_size": 300.0}

#: Tacks, carried out by the autopilot with the route follower's maneuvers
#: (src/route_follower: TackManeuver, JibeManeuver) and its measured numbers: a
#: tack is carried through the wind by momentum, so below MIN_TACK_SPEED the
#: boat first bears away to BUILD_SPEED_ANGLE off the wind to pick up speed
#: (but at most BUILD_SPEED_TIMEOUT s); a tack that has not come through the
#: wind after TACK_STUCK_S becomes a jibe. Maneuvers are stepped every
#: MANEUVER_STEP s; one counts as done within MANEUVER_TOLERANCE of its heading,
#: and is given up after MANEUVER_TIMEOUT s.
MIN_TACK_SPEED = 0.8
BUILD_SPEED_ANGLE = 65.0
BUILD_SPEED_TIMEOUT = 20.0
TACK_STUCK_S = 8.0
MANEUVER_STEP = 0.2
MANEUVER_TOLERANCE = 10.0
MANEUVER_TIMEOUT = 40.0
#: After turning onto the new heading, the autopilot holds it until the boat is
#: moving again (MIN_TACK_SPEED), at most SETTLE_TIMEOUT s, before the pilot is
#: asked: a tack bleeds off speed, and asked straight away the pilot saw a slow
#: boat close to the wind and tacked right back.
SETTLE_TIMEOUT = 15.0
#: ... "moving again" meaning this fast: at MIN_TACK_SPEED the boat is still
#: accelerating and pointing high, and the pilot read that as being stuck.
SETTLED_SPEED = 1.0

#: How many past decisions the pilot is shown.
HISTORY = 6
#: Stop after this many failed requests in a row.
MAX_CONSECUTIVE_ERRORS = 5
#: Contact with land flickers while the boat grinds along a bank; a new
#: grounding is counted only after this many seconds clear of land.
GROUNDING_GAP = 1.0

#: --stop-on-grounding: a run ends once the boat has been stuck on land (in
#: contact, slower than this) for that many seconds. Brief touches do not count.
STUCK_SPEED = 0.3


@dataclass
class Maneuver:
    """A tack the autopilot is carrying out (see MIN_TACK_SPEED and below)."""

    target: float                 # the heading to end up on
    started: float                # simulated time it began
    decision: "Decision"          # whose note to update with how it went
    phase: str = "tacking"        # "building speed", "tacking", "jibing" or "settling"
    action: object = None         # the route follower's TackManeuver / JibeManeuver
    how: str = ""                 # "tacked" or "could not tack, so jibed", once turned
    phase_started: float = 0.0
    record: dict = field(default_factory=dict)


@dataclass
class Result:
    success: bool
    end_reason: str
    sim_time: float
    wall_time: float
    decisions: list[Decision]
    min_distance: float | None
    final_distance: float | None
    time_aground: float
    groundings: int

    def summary(self) -> dict:
        lat = [d.latency for d in self.decisions if not d.error]
        tokens_in = sum(d.input_tokens for d in self.decisions)
        tokens_out = sum(d.output_tokens for d in self.decisions)
        return {
            "success": self.success, "end_reason": self.end_reason,
            "sim_time": round(self.sim_time, 1), "wall_time": round(self.wall_time, 1),
            "decisions": len(self.decisions),
            "errors": sum(1 for d in self.decisions if d.error),
            "claimed_done": any(d.done for d in self.decisions),
            "min_distance": None if self.min_distance is None else round(self.min_distance, 1),
            "final_distance": (None if self.final_distance is None
                               else round(self.final_distance, 1)),
            "time_aground": round(self.time_aground, 1), "groundings": self.groundings,
            "latency_mean": round(statistics.mean(lat), 2) if lat else None,
            "latency_p90": (round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2) if lat else None),
            "latency_max": round(max(lat), 2) if lat else None,
            "input_tokens": tokens_in, "output_tokens": tokens_out,
        }


class PilotRun:
    """One scenario, the controller node and the physics, steered by a pilot.

    Args:
        scenario: Map, boat start, wind and buoys (its waypoints are ignored).
        task: What the pilot is asked to do.
        start: Override the boat's start (east, north).
        heading: Initial compass heading in degrees.
        image_size: (width, height) of the onboard image sent to the pilot.
    """

    def __init__(self, scenario: Scenario, task: Task, start=None, heading: float | None = None,
                 image_size: tuple[int, int] = (640, 480), guard: bool = True,
                 chart: bool = False, chart_size: int = CHART_SIZE,
                 chart_view: float = CHART_VIEW):
        self.scenario = scenario
        self.task = task
        self.guard = guard
        self.image_size = image_size
        self.clock = SimClock()
        self.bus = Bus()
        self.log: list[str] = []

        self.map = load_map(scenario.map)
        half = world_half_size_for(self.map.water)
        land = land_from_water(self.map.water, half)
        buoys = [(b.east, b.north, *b.rgb) for b in scenario.buoys]
        self.sim = SailboatSimulation(
            buoys=buoys, include_course_line=True,
            wind_direction_deg=scenario.wind_direction, wind_speed=scenario.wind_speed,
            obstacles=land, world_half_size=half)
        self.start = tuple(start or scenario.boat)
        self.sim.set_boat_position(*self.start)
        if heading is not None:
            self.sim.data.qpos[self.sim.qadr_yaw] = math.radians(90.0 - heading)
        # Body and camera poses follow qpos only after a forward pass; without
        # it the first image would be taken from the default pose at the origin.
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.fovy = float(self.sim.model.cam(ONBOARD_CAMERA).fovy[0])

        self.chart = None
        if chart:
            from simulation.minimap import Minimap
            self.chart = Minimap(chart_size, chart_size, water=self.map.water, boat=self.start,
                                 buoys=buoys, view_size=chart_view)

        self.recording = Recording(map_path=scenario.map, rate=RECORD_RATE,
                                   render=dict(VIDEO_SETTINGS))
        self.recording.start_segment(Scene(buoys=buoys, boat_east=self.start[0],
                                           boat_north=self.start[1],
                                           wind_direction=scenario.wind_direction,
                                           wind_speed=scenario.wind_speed))

        # The controller node, exactly as the live system runs it, on the bus.
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        original_get_logger = Node.get_logger
        Node.get_logger = lambda node: _Logger(self, node.get_name())
        try:
            self.controller = controller_module.ControllerNode(
                parameter_overrides=_parameters("controller", "controller_node"))
        finally:
            Node.get_logger = original_get_logger
        self.timers: list[_Timer] = []
        node = self.controller
        for publisher in node.publishers:
            publisher.publish = (lambda m, _t=publisher.topic_name: self.bus.publish(_t, m))
        for subscription in node.subscriptions:
            self.bus.subscribe(subscription.topic_name, subscription.callback)
        for timer in node.timers:
            period = timer.timer_period_ns / 1e9
            self.timers.append(_Timer(period, timer.callback, period, len(self.timers)))
            timer.cancel()
        logger = _Logger(self, node.get_name())
        node.get_logger = lambda: logger
        self.bus.subscribe("/control/rudder", lambda m: self.sim.set_rudder(m.angle))
        self.bus.subscribe("/control/sail", lambda m: self.sim.set_sail_angle(m.sail_angle))

        self.maneuver: Maneuver | None = None
        self.maneuvers: list[dict] = []     # every tack: when, how it went
        self.desired_heading = self.sim.get_status()["heading"]
        self._set_heading(self.desired_heading)
        self._publish_sensors()
        self.bus.drain()

    @property
    def half_fov(self) -> float:
        """Degrees the onboard image covers either side of the bow."""
        w, h = self.image_size
        return math.degrees(math.atan(w / h * math.tan(math.radians(self.fovy) / 2)))

    def land_distance(self, east: float, north: float, heading: float,
                      reach: float = 300.0) -> float | None:
        """Metres from (east, north) to the shore along a compass heading, or None
        if there is open water for `reach` metres. From the map: logged to
        judge the pilot's land_ahead estimates, never shown to the pilot."""
        from shapely.geometry import LineString, Point

        a = math.radians(heading)
        ray = LineString([(east, north),
                          (east + reach * math.sin(a), north + reach * math.cos(a))])
        outside = ray.difference(self.map.water)
        if outside.is_empty:
            return None
        here = Point(east, north)
        return min(here.distance(g) for g in getattr(outside, "geoms", [outside]))

    def _record(self, node: str, level: str, text: str) -> None:
        self.log.append(f"[{self.clock.t:8.2f}] [{level}] [{node}] {text}")

    def _publish_sensors(self) -> None:
        s = self.sim.get_status()
        self.bus.publish("/sense/wind", Wind(angle=s["wind_dir"], speed=s["wind_speed"]))
        self.bus.publish("/sense/boat_info", BoatInfo(
            x=s["x"], y=s["y"], velocity_x=s["vel_x"], velocity_y=s["vel_y"],
            velocity_magnitude=s["speed"], yaw_rate=s["yaw_rate"], heading=s["heading"],
            rudder_angle=s["rudder"], sail_angle=s["sail_angle"]))
        self.bus.publish("/sense/collision", Bool(data=bool(s["collision"])))

    def _set_heading(self, compass_deg: float) -> None:
        self.desired_heading = compass_deg % 360.0
        self.bus.publish("/planning/desired_heading", Heading(heading=float(self.desired_heading)))
        self.bus.drain()

    def steer(self, target: float, decision: Decision | None = None) -> None:
        """Head for `target`: a plain turn, or a tack if it means turning through the wind."""
        s = self.sim.get_status()
        if not crosses_wind(s["heading"], target, s["wind_dir"]):
            self._set_heading(target)
            return
        self.maneuver = Maneuver(target=target % 360.0, started=self.clock.t,
                                 decision=decision or Decision(), phase_started=self.clock.t)
        self.maneuver.record = {"start": round(self.clock.t, 1),
                                "from": round(s["heading"], 1), "to": round(target, 1),
                                "speed": round(s["speed"], 2)}
        if s["speed"] < MIN_TACK_SPEED:
            # Bear away on the current side to pick up speed first.
            side = math.copysign(1.0, wrap180(s["heading"] - s["wind_dir"]) or 1.0)
            self.maneuver.phase = "building speed"
            self._set_heading(s["wind_dir"] + side * BUILD_SPEED_ANGLE)
        else:
            self._begin_tack()
        self.maneuver.decision.maneuver = (
            f"the autopilot is tacking onto {self.maneuver.target:.0f}°"
            + (" after picking up speed" if self.maneuver.phase == "building speed" else ""))

    def _begin_tack(self) -> None:
        from route_follower.maneuver_action import TackManeuver

        m = self.maneuver
        s = self.sim.get_status()
        m.phase, m.phase_started = "tacking", self.clock.t
        m.action = TackManeuver(m.target, s["heading"], 10.0, s["wind_dir"], 1.5,
                                MANEUVER_TOLERANCE,
                                max_steps_to_cross_wind=round(TACK_STUCK_S / MANEUVER_STEP))
        self._set_heading(m.target)

    def step_maneuver(self) -> None:
        """Advance the tack in progress; called every MANEUVER_STEP s."""
        from route_follower.maneuver_action import JibeManeuver

        m = self.maneuver
        if m is None:
            return
        s = self.sim.get_status()
        heading = s["heading"]
        elapsed = self.clock.t - m.started
        if elapsed > MANEUVER_TIMEOUT:
            return self._end_maneuver("gave up after {:.0f} s".format(elapsed), heading)
        if m.phase == "building speed":
            if (s["speed"] >= MIN_TACK_SPEED
                    or self.clock.t - m.phase_started > BUILD_SPEED_TIMEOUT):
                self._begin_tack()
            return
        if m.phase == "settling":
            if s["speed"] >= SETTLED_SPEED or self.clock.t - m.phase_started > SETTLE_TIMEOUT:
                return self._end_maneuver(f"{m.how} onto {m.target:.0f}° in {elapsed:.0f} s",
                                          heading)
            return
        command = m.action.step(heading)
        if command is None:
            # On the new heading: hold it until the boat has picked up speed.
            m.how = "tacked" if m.phase == "tacking" else "could not tack, so jibed"
            m.phase, m.phase_started = "settling", self.clock.t
            self._set_heading(m.target)
            return
        if m.phase == "tacking" and m.action.is_stuck(heading):
            m.phase, m.phase_started = "jibing", self.clock.t
            m.action = JibeManeuver(m.target, heading, 10.0, s["wind_dir"], 1.5,
                                    MANEUVER_TOLERANCE)
            command = m.action.current_command
        if abs(wrap180(command - self.desired_heading)) > 0.5:
            self._set_heading(command)

    def _end_maneuver(self, outcome: str, heading: float) -> None:
        m = self.maneuver
        self.maneuver = None
        self._set_heading(m.target)
        m.decision.maneuver = f"the autopilot {outcome}"
        m.record.update({"end": round(self.clock.t, 1), "outcome": outcome,
                         "final_heading": round(heading, 1),
                         "final_speed": round(self.sim.get_boat_speed(), 2)})
        self.maneuvers.append(m.record)
        self._record("autopilot", "INFO", f"maneuver: {outcome}")

    def observe(self, index: int, history: list[Decision]) -> Observation:
        s = self.sim.get_status()
        w, h = self.image_size
        image = draw_bearing_scale(self.sim.render_onboard(w, h), self.fovy)
        image = draw_distance_marks(image, water_distance_rows(self.sim, w, h))
        chart = chart_jpeg = None
        chart_width = arrow_m = 0.0
        if self.chart is not None:
            x, y = float(s["x"]), float(s["y"])
            yaw = float(self.sim.data.qpos[self.sim.qadr_yaw])
            chart = self.chart.render(x, y, yaw)
            chart = draw_wind(chart, self.chart, x, y, float(s["wind_dir"]))
            chart = draw_course_line(chart, self.chart, x, y, float(s["heading"]))
            chart = draw_north(draw_scale_and_position(chart, self.chart, x, y))
            chart_jpeg = jpeg_bytes(chart)
            chart_width = self.chart.width / self.chart._scale
            # The minimap's dart reaches 1.5 x max(6, width // 22) px, back to tip.
            arrow_m = 1.5 * max(6, self.chart.width // 22) / self.chart._scale
        return Observation(
            chart=chart, chart_jpeg=chart_jpeg, chart_width_m=chart_width,
            arrow_length_m=arrow_m,
            true_land_ahead=self.land_distance(float(s["x"]), float(s["y"]),
                                               float(s["heading"])),
            t=self.clock.t, index=index, image=image, jpeg=jpeg_bytes(image),
            heading=float(s["heading"]), speed=float(s["speed"]),
            wind_from_rel=wrap180(s["wind_dir"] - s["heading"]),
            wind_from=float(s["wind_dir"]) % 360.0,
            wind_speed=float(s["wind_speed"]), colliding=bool(s["collision"]),
            east=float(s["x"]), north=float(s["y"]), instruction=self.task.instruction,
            history=list(history[-HISTORY:]))

    def run(self, pilot, mode: str = "paused", period: float = 2.0, max_time: float = 300.0,
            stop_on_done: bool = True, on_decision=None, trace=None,
            stop_on_grounding: float = 0.0) -> Result:
        """Sail until success, "done", max_time or too many errors.

        Args:
            pilot: Has decide(Observation) -> Decision.
            mode: "paused" or "realtime" (see the module docstring).
            period: paused: simulated seconds between decisions; realtime: the
                least wall seconds between two requests.
            on_decision: Called with (Observation, Decision) after each decision.
            trace: A list that gets (t, east, north, heading, desired, speed,
                colliding) rows at TRACE_RATE.
            stop_on_grounding: End the run once the boat has been stuck on land
                (touching it within GROUNDING_GAP s, slower than STUCK_SPEED)
                for this many seconds; 0 sails on.
        """
        if mode not in ("paused", "realtime"):
            raise ValueError(f"mode must be 'paused' or 'realtime', got {mode!r}")
        dt = self.sim.model.opt.timestep
        steps = 0
        next_sensor = next_trace = next_record = 0.0
        next_decision = 0.0                 # paused: simulated time of the next decision
        next_maneuver = 0.0
        decisions: list[Decision] = []
        pending = None                      # realtime: (future, observation, wall time sent)
        last_sent = -math.inf
        errors_in_row = 0
        time_aground, groundings, last_contact = 0.0, 0, -math.inf
        stuck_for = 0.0
        min_dist = None
        end_reason = "time limit"
        success = False
        wall_start = time.monotonic()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def apply(obs: Observation, decision: Decision) -> bool:
            """Act on a decision; True if the run should end."""
            nonlocal errors_in_row
            decisions.append(decision)
            if decision.error:
                errors_in_row += 1           # keep the course
            else:
                errors_in_row = 0
                if self.guard:
                    status = self.sim.get_status()
                    decision.steered_heading = guard_heading(
                        decision.new_heading, status["wind_dir"], status["heading"])
                self.steer(decision.applied_heading, decision)
            if on_decision:
                on_decision(obs, decision)
            return errors_in_row >= MAX_CONSECUTIVE_ERRORS or (stop_on_done and decision.done)

        try:
            while True:
                s = self.sim.get_status()
                dist = self.task.distance(s["x"], s["y"])
                if dist is not None:
                    min_dist = dist if min_dist is None else min(min_dist, dist)
                if self.task.succeeded(s["x"], s["y"]):
                    success, end_reason = True, "reached the target"
                    break
                if self.clock.t >= max_time:
                    break
                if stop_on_grounding and stuck_for >= stop_on_grounding:
                    end_reason = f"stuck on land for {stop_on_grounding:g} s"
                    break

                # Decisions — none while the autopilot is tacking: the pilot is
                # asked again once the tack is complete.
                if self.maneuver is not None:
                    pass
                elif mode == "paused" and self.clock.t + 1e-9 >= next_decision:
                    next_decision += period
                    if next_decision <= self.clock.t:
                        # Fell behind while a tack ran: resume the cadence from
                        # now, rather than catching up with a burst of requests.
                        next_decision = self.clock.t + period
                    obs = self.observe(len(decisions) + 1, decisions)
                    if apply(obs, pilot.decide(obs)):
                        end_reason = self._end_reason(decisions)
                        break
                elif mode == "realtime":
                    if pending is not None and pending[0].done():
                        future, obs, _ = pending
                        pending = None
                        if apply(obs, future.result()):
                            end_reason = self._end_reason(decisions)
                            break
                    now = time.monotonic()
                    if pending is None and self.maneuver is None and now - last_sent >= period:
                        obs = self.observe(len(decisions) + 1, decisions)
                        pending = (executor.submit(pilot.decide, obs), obs, now)
                        last_sent = now

                # Physics, one timestep, and everything that runs on simulated time.
                self.sim.step()
                steps += 1
                self.clock.t = steps * dt
                colliding = self.sim.in_collision()
                if colliding:
                    time_aground += dt
                    if self.clock.t - last_contact > GROUNDING_GAP:
                        groundings += 1
                    last_contact = self.clock.t
                if (self.clock.t - last_contact <= GROUNDING_GAP
                        and self.sim.get_boat_speed() < STUCK_SPEED):
                    stuck_for += dt
                else:
                    stuck_for = 0.0
                if self.maneuver is not None and self.clock.t + 1e-9 >= next_maneuver:
                    next_maneuver = self.clock.t + MANEUVER_STEP
                    self.step_maneuver()
                if self.clock.t + 1e-9 >= next_sensor:
                    next_sensor += 1.0 / SENSOR_RATE
                    self._publish_sensors()
                    self.bus.drain()
                for timer in sorted(self.timers, key=lambda tm: (tm.next_fire, tm.order)):
                    if self.clock.t + 1e-9 >= timer.next_fire:
                        timer.next_fire += timer.period
                        timer.callback()
                        self.bus.drain()
                if self.clock.t + 1e-9 >= next_record:
                    next_record += 1.0 / RECORD_RATE
                    self.sim.update_course_line(math.radians(90.0 - self.desired_heading))
                    self.recording.add_frame(self.clock.t, self.sim.data, colliding)
                if self.clock.t + 1e-9 >= next_trace:
                    # Every 1/TRACE_RATE s: the chart's track, and the trace.
                    next_trace += 1.0 / TRACE_RATE
                    s = self.sim.get_status()
                    if self.chart is not None:
                        self.chart.update_boat(s["x"], s["y"])
                    if trace is not None:
                        trace.append((round(self.clock.t, 2), round(s["x"], 2),
                                      round(s["y"], 2), round(s["heading"], 1),
                                      round(self.desired_heading, 1), round(s["speed"], 2),
                                      colliding))
                if mode == "realtime":
                    ahead = self.clock.t - (time.monotonic() - wall_start)
                    if ahead > 0.005:
                        time.sleep(ahead)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        s = self.sim.get_status()
        return Result(success=success, end_reason=end_reason, sim_time=self.clock.t,
                      wall_time=time.monotonic() - wall_start, decisions=decisions,
                      min_distance=min_dist, final_distance=self.task.distance(s["x"], s["y"]),
                      time_aground=time_aground, groundings=groundings)

    @staticmethod
    def _end_reason(decisions: list[Decision]) -> str:
        if decisions and decisions[-1].done:
            return "the pilot said it was done"
        return f"{MAX_CONSECUTIVE_ERRORS} failed requests in a row ({decisions[-1].error})"

    def close(self) -> None:
        Node.destroy_node(self.controller)   # not ControllerNode's: it saves a plot
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        self.sim.close()


class _Logger:
    """Stands in for the controller's rclpy logger."""

    def __init__(self, harness: PilotRun, node: str):
        self._harness, self._node = harness, node

    def __getattr__(self, name):
        if name in ("debug", "info", "warn", "warning", "error", "fatal"):
            return lambda msg, *a, **k: self._harness._record(self._node, name.upper(), msg)
        raise AttributeError(name)
