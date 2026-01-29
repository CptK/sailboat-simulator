"""This module provides the node responsible for planning trajectories within a local navigation frame."""
from datetime import datetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
import warnings

# ROS2 interfaces
import rclpy
from rclpy.node import Node

# Boat messages and services
from boat_msgs.msg import BoatInfo
from boat_msgs.msg import Wind
from boat_msgs.msg import Heading
from boat_msgs.msg import Route as RouteMsg
from boat_msgs.msg import Location as LocationMsg
from std_msgs.msg import Bool

# Planning backend
from .planner import Mission
from .waypoint import WayPoint


class Config(BaseModel):
    """Configuration model for the NearPlanner2 node with validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Time interval between steps in seconds
    step_interval: float = Field(..., gt=0, description="Step interval must be larger than 0")

    # mission
    dist_threshold: float = Field(..., ge=0, description="Distance threshold must be at least 0")
    off_course_threshold: float = Field(..., ge=0, description="Off course threshold must be at least 0")
    min_tack_angle: float = Field(..., ge=0, description="Minimum tack angle must be at least 0")
    safe_jibe_threshold: float = Field(..., ge=0, description="Safe jibe threshold must be at least 0")
    maneuver_heading_step_size: float = Field(
        ..., ge=0, description="Maneuver heading step size must be at least 0"
    )
    command_step_size_multiplier: float = Field(
        ..., ge=0, description="Command step size multiplier must be at least 0"
    )
    heading_tolerance: float = Field(..., ge=0, description="Heading tolerance must be at least 0")
    max_steps_to_cross_wind_when_tacking: int = Field(
        ..., ge=0, description="Max steps to cross wind when tacking must be at least 0"
    )

    # logging
    time_between_missing_values_logs: float = Field(
        5.0,
        ge=0,
        description="Time between logging missing values in seconds. Must be at least 0.",
    )

    @field_validator("dist_threshold")
    @classmethod
    def validate_dist_threshold(cls, dist_threshold: float) -> float:
        if dist_threshold < 1:
            warnings.warn(
                f"Distance threshold is set to a very low value ({dist_threshold}m), requiring high precision for reaching waypoints."  # noqa: E501
            )
        return dist_threshold

    @field_validator("off_course_threshold")
    @classmethod
    def validate_off_course_threshold(cls, off_course_threshold: float) -> float:
        if off_course_threshold < 1:
            warnings.warn(
                f"Off course threshold is set to a very low value ({off_course_threshold}m), requiring high \
                    precision for staying on course and causing potentially frequent route replanning."
            )
        return off_course_threshold

    @model_validator(mode="after")
    def validate_time_to_cross_wind_when_tacking(self):
        """Ensure that max_steps_to_cross_wind_when_tacking is not larger than the step interval."""
        max_steps = self.max_steps_to_cross_wind_when_tacking
        step_interval = self.step_interval
        total_allowed_seconds = max_steps * step_interval
        if total_allowed_seconds < 10:
            warnings.warn(
                f"The maximum time to cross the wind when tacking is set to a very low value ({total_allowed_seconds}s)."  # noqa: E501
            )
        return self
    
    def __str__(self) -> str:
        return self.model_dump_json(indent=4)


class RoutePlanner(Node):
    """The main node of the near planner.

    It receives environmental data (position, nautical chart entries, vessels, etc.) and produces both a good
    route as well as the next required heading.
    It starts navigating as soon as a boat and goal position are published.
    """

    def __init__(self, **kwargs):
        super().__init__("route_planner", automatically_declare_parameters_from_overrides=True, **kwargs)

        # STATE - INIT
        try:
            self.config: Config = self._load_config()
            self.get_logger().info(f"Config loaded successfully:\n{self.config}")
        except ValueError as err:
            self.get_logger().error(f"Config validation failed: {err}")
            return

        self.mission: Mission | None = None
        self.last_boat_location: WayPoint | None = None
        self.last_boat_heading: float | None = None
        self.wind_direction: float | None = None
        self.wind_speed: float | None = None

        # LOGGING - TIMER
        # If values are missing, the node logs warnings which should not be spammed and therfore are throttled
        self._last_missing_route_log_time: datetime = datetime.now()
        self._last_missing_wind_log_time: datetime = datetime.now()
        self._last_missing_location_log_time: datetime = datetime.now()
        self._last_missing_heading_log_time: datetime = datetime.now()

        # SUBSCRIBER - INIT
        self.create_subscription(BoatInfo, "/sense/boat_info", self._on_boat_info, 1)
        self.create_subscription(Wind, "/sense/wind", self._on_wind_change, 1)
        self.create_subscription(RouteMsg, "/planning/target_route", self._on_subgoal_change, 1)

        # PUBLISHER - INIT
        self._publisher_current_heading_as_angle = self.create_publisher(
            Heading, "/planning/desired_heading", 1
        )
        self._publisher_current_route = self.create_publisher(RouteMsg, "/planning/current_route", 1)
        self._publisher_route_completed = self.create_publisher(Bool, "/planning/route_completed", 1)

        # TIMER for desired heading
        self._timer = self.create_timer(self.config.step_interval, self._desired_heading_timer_callback)

    def _load_config(self) -> Config:
        return Config(
            step_interval=self.get_parameter("step_interval").value,
            dist_threshold=self.get_parameter("dist_threshold").value,
            off_course_threshold=self.get_parameter("off_course_threshold").value,
            min_tack_angle=self.get_parameter("min_tack_angle").value,
            safe_jibe_threshold=self.get_parameter("safe_jibe_threshold").value,
            maneuver_heading_step_size=self.get_parameter("maneuver_heading_step_size").value,
            command_step_size_multiplier=self.get_parameter("command_step_size_multiplier").value,
            heading_tolerance=self.get_parameter("heading_tolerance").value,
            max_steps_to_cross_wind_when_tacking=self.get_parameter(
                "max_steps_to_cross_wind_when_tacking"
            ).value,
            time_between_missing_values_logs=self.get_parameter("time_between_missing_values_logs").value,
        )

    def _on_wind_change(self, msg: Wind) -> None:
        if not (0 <= msg.angle <= 360):
            self.get_logger().error(f"Invalid wind angle: {msg.angle}. Must be between 0 and 360 degrees.")
            return

        self.wind_direction = msg.angle
        self.wind_speed = msg.speed

    def _on_boat_info(self, msg: BoatInfo) -> None:
        self.last_boat_location = WayPoint(msg.x, msg.y)
        self.last_boat_heading = msg.heading

    def _on_subgoal_change(self, msg: RouteMsg) -> None:
        waypoints: list[WayPoint] = []
        for wp_msg in msg.waypoints:
            waypoints.append(WayPoint.from_coordinates(wp_msg.east, wp_msg.north))

        self.get_logger().info(f"New target route received")
        self.get_logger().info(f"Waypoints: {waypoints}")

        self.mission = Mission(
            waypoints=waypoints,
            dist_threshold=self.config.dist_threshold,
            off_course_threshold=self.config.off_course_threshold,
            min_tack_angle=self.config.min_tack_angle,
            safe_jibe_threshold=self.config.safe_jibe_threshold,
            maneuver_heading_step_size=self.config.maneuver_heading_step_size,
            command_step_size_multiplier=self.config.command_step_size_multiplier,
            heading_tolerance=self.config.heading_tolerance,
            max_steps_to_cross_wind_when_tacking=self.config.max_steps_to_cross_wind_when_tacking,
            logger=self.get_logger(),
        )

        # Plan the route if we have the required data, otherwise it will be planned on first step
        if self.last_boat_location is not None and self.wind_direction is not None:
            self.mission.plan(self.last_boat_location, self.wind_direction)
            self.get_logger().info("Initial route planned")

    def _desired_heading_timer_callback(self) -> None:
        if self.mission is None:
            if (
                datetime.now() - self._last_missing_route_log_time
            ).total_seconds() > self.config.time_between_missing_values_logs:
                self._last_missing_route_log_time = datetime.now()
                self.get_logger().warn("No mission set yet")
            return

        if self.last_boat_location is None:
            if (
                datetime.now() - self._last_missing_location_log_time
            ).total_seconds() > self.config.time_between_missing_values_logs:
                self._last_missing_location_log_time = datetime.now()
                self.get_logger().warn("No location published yet")
            return

        if self.last_boat_heading is None:
            if (
                datetime.now() - self._last_missing_heading_log_time
            ).total_seconds() > self.config.time_between_missing_values_logs:
                self._last_missing_heading_log_time = datetime.now()
                self.get_logger().warn("Heading of the boat is unknown")
            return

        if not self.wind_direction or self.wind_speed is None:
            self.get_logger().warn("Wind data is not valid yet")
            return

        # Plan the route if it hasn't been planned yet
        if not getattr(self.mission, 'route', None):
            self.mission.plan(self.last_boat_location, self.wind_direction)
            self.get_logger().info("Route planned")

        desired_heading = self.mission.step(self.last_boat_location, self.last_boat_heading, self.wind_direction, self.wind_speed)
        self._publisher_current_heading_as_angle.publish(Heading(heading=desired_heading))

        # Publish current route as RouteMsg
        route_msg = RouteMsg()
        for wp in self.mission.route:
            wp_msg = LocationMsg(east=wp.east, north=wp.north)
            route_msg.waypoints.append(wp_msg)

        self._publisher_current_route.publish(route_msg)

        # Publish route completion status
        if self.mission.completed:
            self._publisher_route_completed.publish(Bool(data=True))


def main(args: list[str] | None = None) -> None:
    """The main entry point of this node."""
    rclpy.init(args=args)

    node = RoutePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "main":
    main()
