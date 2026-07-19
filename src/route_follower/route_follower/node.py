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

from datetime import datetime

import rclpy
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
        self.wind_direction: float | None = None
        self.wind_speed: float | None = None

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
        return Config(
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

    def _throttled_warn(self, message: str, last_logged: datetime) -> datetime:
        """Log a warning at most every `time_between_missing_values_logs` seconds."""
        if (datetime.now() - last_logged).total_seconds() > self.config.time_between_missing_values_logs:
            self.get_logger().warn(message)
            return datetime.now()
        return last_logged

    # ── subscriptions ─────────────────────────────────────────────────────────

    def _on_target_route(self, msg: RouteMsg) -> None:
        """Accept a new route to follow, replacing any current one."""
        self.route = [WayPoint(float(wp.east), float(wp.north)) for wp in msg.waypoints]
        self.target_index = 0
        self.maneuver = None
        self.completed = False
        self.get_logger().info(f"Following new route: {len(self.route)} waypoints")

        # Echo it on current_route, which the simulation renders.
        self._publisher_current_route.publish(msg)

    def _on_boat_info(self, msg: BoatInfo) -> None:
        self.location = WayPoint(float(msg.x), float(msg.y))
        self.heading = float(msg.heading)

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
        if not self.wind_direction or self.wind_speed is None:
            self._last_missing_wind_log_time = self._throttled_warn(
                "Wind data is not valid yet", self._last_missing_wind_log_time
            )
            return

        if self.completed:
            return

        desired_heading = self._step(self.location, self.heading, self.wind_direction, self.wind_speed)
        self._publisher_desired_heading.publish(Heading(heading=desired_heading))

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
                return self.maneuver.step(heading) or desired_heading

        # Start a maneuver when the required turn is too large to just steer.
        if abs(heading_diff) > self.config.maneuver_threshold:
            self.get_logger().info(f"Starting maneuver: heading difference = {heading_diff:.1f}°")
            self.maneuver = self._select_maneuver(
                location, target, heading, wind_direction, wind_speed
            )
            return self.maneuver.step(heading) or desired_heading

        # Otherwise sail straight at the waypoint.
        return desired_heading

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
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
