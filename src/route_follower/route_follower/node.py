"""The node that steers along a fixed route, without re-planning it.

This is the *steering* half of navigation: it takes a route as given — the
sequence of waypoints someone else decided — and produces the headings that
sail it, executing tack and jibe maneuvers where a leg crosses the wind. It
never changes the route.

    /planning/target_route ─┐
    /sense/boat_info        ─┼─> route_follower ──> /planning/desired_heading
    /sense/wind             ─┘                 ──> /planning/current_route
                                               ──> /planning/route_completed

It is deliberately distinct from `route_planner`, which both plans *and*
steers. Pairing route_follower with `graph_route_planner` keeps the two jobs in
two nodes: the planner decides the path, the follower only sails it. The
maneuver execution is reused from `route_planner` rather than reimplemented, so
the two agree on how a tack is physically carried out.
"""

import math
import signal
from datetime import datetime

import rclpy
from rclpy.executors import ExternalShutdownException
from pydantic import BaseModel, ConfigDict, Field
from rclpy.node import Node
from std_msgs.msg import Bool

# Boat messages
from boat_msgs.msg import BoatInfo
from boat_msgs.msg import Heading
from boat_msgs.msg import Location as LocationMsg
from boat_msgs.msg import Route as RouteMsg
from boat_msgs.msg import Wind

# Maneuver execution: the tack/jibe state machines and geometry helpers. These
# modules were copied from route_planner rather than imported, so this package
# is self-contained and does not depend on it (see maneuver_action.py etc.).
from route_follower.clearance import (
    Clearance, HeadingChoice, TurnModel, choose_heading, path_clearance, turn_angle,
)
from route_follower.maneuver_action import JibeManeuver, ManeuverAction, TackManeuver
from route_follower.utils import angular_distance, bearing
from route_follower.waypoint import WayPoint


class Config(BaseModel):
    """Configuration model for the route follower node, with validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Cadence
    step_interval: float = Field(..., gt=0, description="Step interval must be larger than 0")

    # Following
    dist_threshold: float = Field(
        ..., ge=0, description="Radius in metres within which a waypoint counts as reached"
    )
    maneuver_threshold: float = Field(
        70.0, gt=0, le=180,
        description="Heading error in degrees that triggers a tack/jibe rather than a turn",
    )

    # Maneuver execution — passed straight to the reused Tack/Jibe maneuvers.
    safe_jibe_threshold: float = Field(..., ge=0, description="Wind speed above which jibing is unsafe")
    maneuver_heading_step_size: float = Field(..., ge=0, description="Heading step per maneuver tick")
    command_step_size_multiplier: float = Field(..., ge=0, description="Command step size multiplier")
    heading_tolerance: float = Field(..., ge=0, description="Allowed heading error, degrees")
    max_steps_to_cross_wind_when_tacking: int = Field(
        ..., ge=0, description="Max steps to cross the wind before a tack is deemed stuck"
    )

    # Keeping clear of land — only with a map; see `_avoid_land`.
    map_path: str = Field("", description="Map whose land to keep clear of; empty disables it")
    hull_clearance: float = Field(
        1.5, ge=0, description="Closest the boat's reference point may come to land, metres"
    )
    min_lookahead: float = Field(
        10.0, ge=0, description="Shortest distance ahead that must be clear of land, metres"
    )
    lookahead_time: float = Field(
        5.0, ge=0, description="Seconds of sailing ahead that must be clear of land"
    )
    no_go_angle: float = Field(
        45.0, gt=0, lt=90, description="Closest the boat can sail to the wind, degrees"
    )
    heading_search_step: float = Field(
        5.0, gt=0, le=45, description="Spacing of the headings tried when avoiding land, degrees"
    )

    # Room to turn — see clearance.TurnModel for what each one means.
    turn_delay: float = Field(1.5, ge=0, description="Seconds the boat holds course after a turn is ordered")
    turn_rate: float = Field(60.0, gt=0, description="Turn rate once turning, degrees per second")
    turn_settle: float = Field(2.0, ge=0, description="Seconds on the new heading checked after a turn")
    turn_buffer: float = Field(
        1.5, ge=0, description="Extra distance from land a turn's predicted path keeps, metres"
    )

    # Speed to tack — see `_build_speed_for_tack`.
    min_tack_speed: float = Field(
        0.8, ge=0, description="Slowest boat speed a tack is started at, m/s"
    )
    build_speed_angle: float = Field(
        65.0, gt=0, lt=180, description="Angle off the wind sailed to build speed for a tack, degrees"
    )

    # Logging
    time_between_missing_values_logs: float = Field(
        5.0, ge=0, description="Seconds between repeated warnings about missing inputs"
    )

    def __str__(self) -> str:
        return self.model_dump_json(indent=4)


class RouteFollower(Node):
    """Steers the boat along the route it is given, waypoint by waypoint.

    The route is followed exactly as received: no waypoint is added, moved or
    dropped. Where the heading to the next waypoint is too far from the current
    heading to simply turn — i.e. the leg crosses the wind — a tack or jibe
    maneuver is executed, reusing route_planner's maneuver logic.
    """

    def __init__(self, **kwargs):
        super().__init__(
            "route_follower", automatically_declare_parameters_from_overrides=True, **kwargs
        )

        # STATE - INIT
        try:
            self.config: Config = self._load_config()
            self.get_logger().info(f"Config loaded successfully:\n{self.config}")
        except ValueError as err:
            self.get_logger().error(f"Config validation failed: {err}")
            return

        self.route: list[WayPoint] = []
        self.target_index: int = 0
        self.maneuver: ManeuverAction | None = None
        self.completed: bool = False

        self.location: WayPoint | None = None
        self.heading: float | None = None
        self.velocity: tuple[float, float] = (0.0, 0.0)    # east, north; m/s
        self.wind_direction: float | None = None
        self.wind_speed: float | None = None

        # Land to keep clear of, if a map is configured. A turn is checked
        # against the stricter clearance first, falling back to the hull's.
        self.clearance: Clearance | None = None
        self._turn_checks: list[Clearance] = []
        self._load_clearance()
        self._land_override: HeadingChoice | None = None
        self.turn_model = TurnModel(
            delay=self.config.turn_delay, rate=self.config.turn_rate,
            settle=self.config.turn_settle, buffer=self.config.turn_buffer,
        )
        self._waiting_for_room: bool = False
        self._building_speed: bool = False

        # LOGGING - TIMER
        self._last_missing_route_log_time: datetime = datetime.now()
        self._last_missing_location_log_time: datetime = datetime.now()
        self._last_missing_heading_log_time: datetime = datetime.now()
        self._last_missing_wind_log_time: datetime = datetime.now()

        # SUBSCRIBER - INIT
        self.create_subscription(RouteMsg, "/planning/target_route", self._on_target_route, 1)
        self.create_subscription(BoatInfo, "/sense/boat_info", self._on_boat_info, 1)
        self.create_subscription(Wind, "/sense/wind", self._on_wind_change, 1)

        # PUBLISHER - INIT
        self._publisher_desired_heading = self.create_publisher(Heading, "/planning/desired_heading", 1)
        self._publisher_current_route = self.create_publisher(RouteMsg, "/planning/current_route", 1)
        self._publisher_route_completed = self.create_publisher(Bool, "/planning/route_completed", 1)

        # TIMER for steering
        self._timer = self.create_timer(self.config.step_interval, self._step_timer_callback)

    def _load_config(self) -> Config:
        def optional(name: str) -> dict:
            # Parameters added later may be missing from an older parameter file.
            return ({name: self.get_parameter(name).value}
                    if self.has_parameter(name) else {})

        return Config(
            **optional("map_path"),
            **optional("hull_clearance"),
            **optional("min_lookahead"),
            **optional("lookahead_time"),
            **optional("no_go_angle"),
            **optional("heading_search_step"),
            **optional("turn_delay"),
            **optional("turn_rate"),
            **optional("turn_settle"),
            **optional("turn_buffer"),
            **optional("min_tack_speed"),
            **optional("build_speed_angle"),
            step_interval=self.get_parameter("step_interval").value,
            dist_threshold=self.get_parameter("dist_threshold").value,
            maneuver_threshold=self.get_parameter("maneuver_threshold").value,
            safe_jibe_threshold=self.get_parameter("safe_jibe_threshold").value,
            maneuver_heading_step_size=self.get_parameter("maneuver_heading_step_size").value,
            command_step_size_multiplier=self.get_parameter("command_step_size_multiplier").value,
            heading_tolerance=self.get_parameter("heading_tolerance").value,
            max_steps_to_cross_wind_when_tacking=self.get_parameter(
                "max_steps_to_cross_wind_when_tacking"
            ).value,
            time_between_missing_values_logs=self.get_parameter(
                "time_between_missing_values_logs"
            ).value,
        )

    def _load_clearance(self) -> None:
        """Load the map's clearance models, or leave them unset to ignore land.

        Loaded exactly as the planner and simulation load the map (by catalog
        name, file in maps/, or path), so all three see the same coastline.
        """
        if not self.config.map_path:
            self.get_logger().info("No map_path: steering without checking for land")
            return None
        try:
            from graph_route_planner.map_loader import load_map

            sail_map = load_map(self.config.map_path)
        except Exception as err:   # a missing map costs the check, not the node
            self.get_logger().error(
                f"Could not load map '{self.config.map_path}': {err}. "
                f"Steering without checking for land."
            )
            return None
        self.get_logger().info(
            f"Keeping {self.config.hull_clearance} m clear of land on map '{sail_map.name}', "
            f"{self.config.hull_clearance + self.config.turn_buffer} m when turning"
        )
        self.clearance = Clearance(sail_map.water, self.config.hull_clearance)
        self._turn_checks = [
            Clearance(sail_map.water, self.config.hull_clearance + self.config.turn_buffer),
            self.clearance,
        ]

    def _throttled_warn(self, message: str, last_logged: datetime) -> datetime:
        """Log a warning at most every `time_between_missing_values_logs` seconds."""
        if (datetime.now() - last_logged).total_seconds() > self.config.time_between_missing_values_logs:
            self.get_logger().warn(message)
            return datetime.now()
        return last_logged

    # ── subscriptions ─────────────────────────────────────────────────────────

    def _prospective_index(self, route: list[WayPoint]) -> int:
        """Index of the waypoint `_step` will steer at once it has skipped reached ones.

        Every published route starts at the boat, so the first waypoint is
        always already reached; this reproduces the skip without waiting a tick.
        """
        if not route or self.location is None:
            return 0
        for index, waypoint in enumerate(route):
            if self.location.distance(waypoint) >= self.config.dist_threshold:
                return index
        return len(route) - 1

    def _prospective_target(self, route: list[WayPoint]) -> WayPoint | None:
        """The waypoint `_step` will steer at once it has skipped reached ones."""
        return route[self._prospective_index(route)] if route else None

    def _on_target_route(self, msg: RouteMsg) -> None:
        """Accept a new route to follow, replacing any current one.

        A maneuver in progress survives if the new route still steers at the
        same place. The planner republishes on every replan, and a tack that
        restarts each time never accumulates the steps `is_stuck` counts — so it
        would pinch head to wind forever instead of falling back to a jibe.
        """
        route = [WayPoint(float(wp.east), float(wp.north), is_soft=bool(wp.is_soft))
                 for wp in msg.waypoints]

        old_target = (self.route[self.target_index]
                      if self.route and self.target_index < len(self.route) else None)
        new_target = self._prospective_target(route)
        same_leg = (old_target is not None and new_target is not None
                    and old_target.distance(new_target) < self.config.dist_threshold)

        self.route = route
        if same_leg:
            # Start at the target already, not at the boat's own position.
            # `_step` treats reaching a waypoint as starting a fresh leg and
            # cancels the maneuver — so starting at index 0 would undo the
            # survival this branch exists for, one tick later.
            self.target_index = self._prospective_index(route)
        else:
            self.target_index = 0
            self.maneuver = None
        self.completed = False
        self.get_logger().info(
            f"Following new route: {len(self.route)} waypoints"
            + (" (continuing the maneuver in progress)" if same_leg and self.maneuver else "")
        )

        # Echo it on current_route, which the simulation renders.
        self._publisher_current_route.publish(msg)

    def _on_boat_info(self, msg: BoatInfo) -> None:
        self.location = WayPoint(float(msg.x), float(msg.y))
        self.heading = float(msg.heading)
        self.velocity = (float(msg.velocity_x), float(msg.velocity_y))

    def _on_wind_change(self, msg: Wind) -> None:
        if not (0 <= msg.angle <= 360):
            self.get_logger().error(f"Invalid wind angle: {msg.angle}. Must be between 0 and 360 degrees.")
            return
        self.wind_direction = msg.angle
        self.wind_speed = msg.speed

    # ── steering ──────────────────────────────────────────────────────────────

    def _step_timer_callback(self) -> None:
        if not self.route:
            self._last_missing_route_log_time = self._throttled_warn(
                "No route to follow yet", self._last_missing_route_log_time
            )
            return
        if self.location is None:
            self._last_missing_location_log_time = self._throttled_warn(
                "No boat position published yet", self._last_missing_location_log_time
            )
            return
        if self.heading is None:
            self._last_missing_heading_log_time = self._throttled_warn(
                "Boat heading is unknown", self._last_missing_heading_log_time
            )
            return
        # `is None`, not falsiness: wind from due north is 0 degrees.
        if self.wind_direction is None or self.wind_speed is None:
            self._last_missing_wind_log_time = self._throttled_warn(
                "Wind data is not valid yet", self._last_missing_wind_log_time
            )
            return

        if self.completed:
            return

        desired_heading = self._step(self.location, self.heading, self.wind_direction, self.wind_speed)
        if not self.completed:
            desired_heading = self._avoid_land(desired_heading)
        self._publisher_desired_heading.publish(Heading(heading=desired_heading))

    def _avoid_land(self, desired_heading: float) -> float:
        """Replace the desired heading if sailing it would run onto land.

        Looks ahead along the path the boat will actually make good — the
        heading plus its current leeway — for `lookahead_time` seconds of
        sailing (at least `min_lookahead` metres), but never past the waypoint
        being steered at: the boat turns there, and the next leg gets its own
        check. If land is in the way, steers the nearest sailable heading that
        is clear instead (see `clearance.choose_heading`).

        This is the fast safety loop. The planner replans, but every few
        seconds at best; this runs every steering step.
        """
        if self.clearance is None or self.location is None or not self.route:
            return desired_heading

        speed = math.hypot(*self.velocity)
        drift = 0.0
        if speed > 0.5:     # below that, the velocity's direction means little
            course = math.degrees(math.atan2(self.velocity[0], self.velocity[1]))
            drift = max(-30.0, min(30.0, (course - self.heading + 180.0) % 360.0 - 180.0))

        target = self.route[min(self.target_index, len(self.route) - 1)]
        lookahead = max(self.config.min_lookahead, speed * self.config.lookahead_time)
        lookahead = min(lookahead, self.location.distance(target))
        if lookahead <= 0:
            return desired_heading

        choice = choose_heading(
            self.clearance, self.location.east, self.location.north, desired_heading,
            wind_direction=self.wind_direction, no_go_angle=self.config.no_go_angle,
            lookahead=lookahead, current_heading=self.heading, drift=drift,
            search_step=self.config.heading_search_step,
        )

        previous = self._land_override
        if choice.overridden:
            if previous is None or previous.kind != choice.kind:
                self.get_logger().info(f"Avoiding land: {choice.reason}")
            self._land_override = choice
            return choice.heading
        if previous is not None:
            self.get_logger().info(f"Clear of land again, steering {desired_heading:.0f}°")
        self._land_override = None
        return desired_heading

    def _step(self, location: WayPoint, heading: float,
              wind_direction: float, wind_speed: float) -> float:
        """Produce one steering command toward the current waypoint.

        Args:
            location: The boat's position.
            heading: The boat's current heading, degrees.
            wind_direction: Wind bearing, degrees.
            wind_speed: Wind speed, m/s.

        Returns:
            The desired heading, degrees.
        """
        self._skip_reached_soft_waypoints(location)
        self._turn_early_at_soft_corner(location, heading, wind_direction, wind_speed)
        target = self.route[self.target_index]

        # Advance when the current waypoint is reached.
        if location.distance(target) < self.config.dist_threshold:
            if self.target_index >= len(self.route) - 1:
                self.get_logger().info("Reached final waypoint. Route completed.")
                self.completed = True
                self._publisher_route_completed.publish(Bool(data=True))
                return heading          # hold; nothing left to steer toward
            self.target_index += 1
            self.maneuver = None        # a fresh leg cancels any maneuver in progress
            target = self.route[self.target_index]

        desired_heading = bearing(location, target)
        heading_diff = angular_distance(heading, desired_heading)

        # Continue an active maneuver until it finishes.
        if self.maneuver is not None:
            if self.maneuver.complete:
                self.maneuver = None
            else:
                if self.maneuver.is_stuck(heading):
                    self.get_logger().info("Tack maneuver stuck, switching to jibe")
                    self.maneuver = JibeManeuver(
                        target_heading=desired_heading,
                        current_heading=heading,
                        heading_step_size=self.config.maneuver_heading_step_size,
                        wind_direction=wind_direction,
                        heading_tolerance=self.config.heading_tolerance,
                        command_step_size_multiplier=self.config.command_step_size_multiplier,
                    )
                return self._maneuver_command(heading, desired_heading)

        # Start a maneuver when the required turn is too large to just steer.
        if abs(heading_diff) > self.config.maneuver_threshold:
            maneuver = self._select_maneuver(
                location, target, heading, wind_direction, wind_speed
            )
            maneuver = self._maneuver_with_room(maneuver, location, heading, desired_heading,
                                                wind_direction, wind_speed)
            if maneuver is None:
                # No room to turn either way yet: keep going, and check again
                # next step. If this course runs onto land, the land check
                # after _step bears away from it — which also makes room.
                if not self._waiting_for_room:
                    self.get_logger().info(
                        f"No room to turn onto {desired_heading:.0f}° yet; holding course"
                    )
                self._waiting_for_room = True
                return heading
            self._waiting_for_room = False
            build = self._build_speed_for_tack(maneuver, heading, wind_direction)
            if build is not None:
                return build
            self.get_logger().info(f"Starting maneuver: heading difference = {heading_diff:.1f}°")
            self.maneuver = maneuver
            return self._maneuver_command(heading, desired_heading)

        # Otherwise sail straight at the waypoint.
        return desired_heading

    def _build_speed_for_tack(self, maneuver: ManeuverAction, heading: float,
                              wind_direction: float) -> float | None:
        """A heading to build speed on before tacking, or None to tack now.

        A tack is carried through the wind by the boat's momentum. In the
        simulation every tack started below about 0.5 m/s stalled head to wind
        (and drifted backwards until the jibe fallback got it round), while
        every one from 0.66 m/s up came through. So below `min_tack_speed` the
        boat first bears away to `build_speed_angle` off the wind on its
        current tack — a close reach, its quickest point of sailing short of a
        beam reach — and tacks once it is fast enough. A boat already further
        off the wind keeps its heading. Jibes need no speed and are not held.
        """
        velocity = getattr(self, "velocity", None)
        if velocity is None or not isinstance(maneuver, TackManeuver):
            return None
        speed = math.hypot(*velocity)
        if speed >= self.config.min_tack_speed:
            if self._building_speed:
                self.get_logger().info(f"Up to {speed:.2f} m/s; tacking")
            self._building_speed = False
            return None

        off_wind = (heading - wind_direction + 180.0) % 360.0 - 180.0   # signed
        if abs(off_wind) >= self.config.build_speed_angle:
            command = heading
        else:
            side = 1.0 if off_wind >= 0 else -1.0
            command = (wind_direction + side * self.config.build_speed_angle) % 360.0
        if not self._building_speed:
            self.get_logger().info(
                f"Too slow to tack ({speed:.2f} m/s < {self.config.min_tack_speed} m/s); "
                f"building speed on {command:.0f}° first"
            )
        self._building_speed = True
        return command

    # ── room to turn ──────────────────────────────────────────────────────────

    def _speed_and_course(self, heading: float) -> tuple[float, float]:
        """Boat speed (m/s) and course over ground (degrees; the heading when slow)."""
        east, north = getattr(self, "velocity", (0.0, 0.0))
        speed = math.hypot(east, north)
        if speed < 0.5:
            return speed, heading
        return speed, math.degrees(math.atan2(east, north)) % 360.0

    def _turn_fits(self, location: WayPoint, heading: float, new_heading: float,
                   wind_direction: float, jibe: bool) -> bool:
        """Whether a tack (or jibe) from here onto new_heading stays clear of land."""
        speed, course = self._speed_and_course(heading)
        turn = turn_angle(heading, new_heading, wind_direction, jibe)
        path = self.turn_model.path(location.east, location.north, course, turn, speed)
        return path_clearance(self._turn_checks, path)

    def _maneuver_with_room(self, maneuver: ManeuverAction, location: WayPoint,
                            heading: float, new_heading: float, wind_direction: float,
                            wind_speed: float) -> ManeuverAction | None:
        """The maneuver to make, if there is room for it; else the other kind; else None.

        The chosen maneuver is checked first. If its predicted path runs onto
        land, the other kind turns the other way round, which may fit — a jibe
        only if the wind allows one. With no map, any maneuver fits.
        """
        if not getattr(self, "_turn_checks", None):
            return maneuver
        jibe = isinstance(maneuver, JibeManeuver)
        if self._turn_fits(location, heading, new_heading, wind_direction, jibe):
            return maneuver
        if not jibe and wind_speed is not None and wind_speed > self.config.safe_jibe_threshold:
            return None
        if self._turn_fits(location, heading, new_heading, wind_direction, not jibe):
            other = "jibing" if not jibe else "tacking"
            self.get_logger().info(f"No room to {'jibe' if jibe else 'tack'} here; {other} instead")
            kind = TackManeuver if jibe else JibeManeuver
            kwargs = dict(
                target_heading=new_heading,
                current_heading=heading,
                heading_step_size=self.config.maneuver_heading_step_size,
                wind_direction=wind_direction,
                heading_tolerance=self.config.heading_tolerance,
                command_step_size_multiplier=self.config.command_step_size_multiplier,
            )
            if kind is TackManeuver:
                kwargs["max_steps_to_cross_wind"] = self.config.max_steps_to_cross_wind_when_tacking
            return kind(**kwargs)
        return None

    def _turn_early_at_soft_corner(self, location: WayPoint, heading: float,
                                   wind_direction: float, wind_speed: float) -> None:
        """Turn before a tacking corner if the corner itself has no room to turn.

        Planned tacking corners lie on the shore margin's edge, so the boat
        arrives at one beside land — and a turn carries it several metres on
        before it comes round. A soft waypoint may be cut, so the turn starts
        early when all of these hold:

          * the corner is within one look-ahead distance (as in `_avoid_land`)
            — cutting from further back would leave the planned route;
          * a turn at the corner would not fit;
          * a turn ordered one steering step from now would not fit either,
            but one ordered now does — so now is the last moment;
          * the straight leg from here to the waypoint after the corner is
            clear of land (the planner checked that leg from the corner, not
            from here).
        """
        if not getattr(self, "_turn_checks", None) or self.maneuver is not None:
            return
        index = self.target_index
        if index >= len(self.route) - 1 or not self.route[index].is_soft:
            return
        corner, after = self.route[index], self.route[index + 1]
        if angular_distance(heading, bearing(corner, after)) <= self.config.maneuver_threshold:
            return                      # an ordinary turn, not a maneuver

        speed, course = self._speed_and_course(heading)
        reach = max(self.config.min_lookahead, speed * self.config.lookahead_time)
        if location.distance(corner) > reach:
            return

        def fits(where: WayPoint) -> bool:
            new_heading = bearing(where, after)
            return any(self._turn_fits(where, heading, new_heading, wind_direction, jibe)
                       for jibe in (False, True))

        step = speed * self.config.step_interval
        ahead = WayPoint(location.east + step * math.sin(math.radians(course)),
                         location.north + step * math.cos(math.radians(course)))
        if fits(corner) or fits(ahead) or not fits(location):
            return
        leg = location.distance(after)
        if self.clearance.free_distance(location.east, location.north,
                                        bearing(location, after), leg) < leg:
            return
        self.get_logger().info(
            f"Turning early, {location.distance(corner):.1f} m before the tacking corner: "
            f"no room to turn there"
        )
        self.target_index = index + 1

    def _maneuver_command(self, heading: float, desired_heading: float) -> float:
        """The active maneuver's heading command, or the desired heading once it is done.

        Compared with None, not by truthiness: a command of 0 degrees (due
        north — dead downwind in a southerly, where a jibe steers) is a
        heading, not a missing one.
        """
        command = self.maneuver.step(heading)
        return desired_heading if command is None else command

    def _skip_reached_soft_waypoints(self, location: WayPoint) -> None:
        """Drop tacking corners the boat no longer needs, having arrived anyway.

        A soft waypoint exists only to get the boat somewhere. Once the next
        hard waypoint is within reach, every soft one before it has been made
        pointless — and steering at them means sailing back down the course to
        visit a point that was never a real destination.

        The scan stops at the first hard waypoint, so a mark can never be
        skipped: being near the *next* mark is what licenses the skip, and
        rounding marks in order is the whole point of a course.
        """
        if not self.route or self.target_index >= len(self.route):
            return
        if not self.route[self.target_index].is_soft:
            return

        for index in range(self.target_index, len(self.route)):
            if self.route[index].is_soft:
                continue
            if location.distance(self.route[index]) < self.config.dist_threshold:
                self.get_logger().info(
                    f"Within reach of waypoint {index}; skipping "
                    f"{index - self.target_index} soft waypoint(s)"
                )
                self.target_index = index
                self.maneuver = None      # the skipped leg's maneuver is moot
            break                          # only the next hard waypoint counts

    def _select_maneuver(self, location: WayPoint, target: WayPoint,
                         current_heading: float, wind_direction: float,
                         wind_speed: float) -> ManeuverAction:
        """Choose a tack or a jibe to swing onto the new heading.

        Reproduces route_planner's choice so the two nodes maneuver alike:
        tack when either the old or new heading is close to the wind, jibe when
        both are well off it (unless the wind is too strong to jibe safely),
        and otherwise take whichever turns less.

        Args:
            location: The boat's position.
            target: The waypoint being steered to.
            current_heading: The boat's current heading, degrees.
            wind_direction: Wind bearing, degrees.
            wind_speed: Wind speed, m/s.

        Returns:
            The maneuver to execute.
        """
        new_heading = bearing(location, target)
        tack = TackManeuver(
            target_heading=new_heading,
            current_heading=current_heading,
            heading_step_size=self.config.maneuver_heading_step_size,
            wind_direction=wind_direction,
            max_steps_to_cross_wind=self.config.max_steps_to_cross_wind_when_tacking,
            heading_tolerance=self.config.heading_tolerance,
            command_step_size_multiplier=self.config.command_step_size_multiplier,
        )
        jibe = JibeManeuver(
            target_heading=new_heading,
            current_heading=current_heading,
            heading_step_size=self.config.maneuver_heading_step_size,
            wind_direction=wind_direction,
            heading_tolerance=self.config.heading_tolerance,
            command_step_size_multiplier=self.config.command_step_size_multiplier,
        )

        current_wind_angle = _signed_angle(wind_direction - current_heading)
        new_wind_angle = _signed_angle(wind_direction - new_heading)

        # Close to the wind on either side: only a tack gets through the eye.
        if abs(current_wind_angle) < 90 or abs(new_wind_angle) < 90:
            return tack
        # Running or broad reach on both: jibe, unless it is blowing too hard.
        if abs(current_wind_angle) > 120 and abs(new_wind_angle) > 120:
            if wind_speed is not None and wind_speed > self.config.safe_jibe_threshold:
                return tack
            return jibe
        # In between: whichever turns less.
        return tack if abs(_signed_angle(new_heading - current_heading)) < 180 else jibe


def _signed_angle(angle: float) -> float:
    """Wrap an angle to (-180, 180] degrees."""
    a = angle % 360
    return a - 360 if a > 180 else a


def main(args: list[str] | None = None) -> None:
    """The main entry point of this node."""
    rclpy.init(args=args)

    node = RouteFollower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    # A second SIGINT (ros2 launch forwards one after the run script's) must
    # not interrupt the cleanup below
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    rclpy.try_shutdown()  # the SIGINT handler may already have shut down


if __name__ == "__main__":
    main()
