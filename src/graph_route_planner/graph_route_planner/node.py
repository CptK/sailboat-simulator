"""The node that expands a coarse mission into a sailable route.

A mission is the via points you want to round, in order — the corners of a
course, say. This node plans the water between them: each consecutive pair is
routed around land and islands over a visibility graph, and the segments are
concatenated into one detailed route.

It sits between whoever sets the course and `route_planner`, which follows it:

    /planning/mission ─┐
                       ├─> graph_route_planner ──> /planning/target_route
    /sense/wind       ─┘                                    │
                                                            v
                                                      route_planner ──> /planning/desired_heading

The mission's own via points are always kept, in order. The planner only fills
in how to get from each one to the next.

Frames: the map is projected to metres about its own bounding-box centre, and
those metres are published directly as Location.east / Location.north. This
assumes the map's centre is the world origin the simulation reports positions
in. Nothing in the stack carries a geodetic reference to check that against, so
if the two ever diverge the routes will be silently offset.
"""

from datetime import datetime

import rclpy
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# Boat messages
from boat_msgs.msg import Location as LocationMsg
from boat_msgs.msg import Route as RouteMsg
from boat_msgs.msg import Wind

# Planning backend — plain Python, no ROS.
from graph_route_planner import DEFAULT_MAP
from graph_route_planner.map_loader import SailMap, load_kml
from graph_route_planner.planner import (
    NoWaterError,
    PlannerConfig,
    Position,
    plan_route,
)
from graph_route_planner.sailing import SailingModel

#: The route is state, not a stream. It is published once per mission, so a
#: follower that starts late — or restarts — would otherwise wait indefinitely
#: for a route that has already been sent. Latching hands the current route to
#: every late subscriber. A TRANSIENT_LOCAL publisher still satisfies a VOLATILE
#: subscriber, so this stays compatible with route_planner's default-QoS
#: subscription.
ROUTE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class Config(BaseModel):
    """Configuration model for the graph route planner node, with validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Map
    map_path: str = Field("", description="KML map to plan on; empty uses the bundled default")

    # Planning cadence
    plan_interval: float = Field(..., gt=0, description="Plan interval must be larger than 0")

    # Search tuning — see PlannerConfig for what each one does.
    margin: float = Field(..., gt=0, description="Shore clearance must be larger than 0")
    grid_spacing: float = Field(..., gt=0, description="Grid spacing must be larger than 0")
    merge_threshold: float = Field(..., ge=0, description="Merge threshold must be at least 0")
    min_leg_distance: float = Field(..., ge=0, description="Minimum leg must be at least 0")
    max_leg_distance: float = Field(
        0.0,
        ge=0,
        description="Longest leg in metres; 0 derives it from the map, which never truncates",
    )

    # Logging
    time_between_missing_values_logs: float = Field(
        5.0, ge=0, description="Seconds between repeated warnings about missing inputs"
    )

    @field_validator("plan_interval")
    @classmethod
    def validate_plan_interval(cls, plan_interval: float) -> float:
        # Planning is synchronous and takes ~0.1-0.6 s per segment on a
        # pond-sized map, so a fast interval starves the executor rather than
        # doing anything useful.
        if plan_interval < 1.0:
            import warnings

            warnings.warn(
                f"plan_interval is {plan_interval}s, but planning one mission can take "
                f"several hundred milliseconds per leg and blocks this node while it runs."
            )
        return plan_interval

    def to_planner_config(self) -> PlannerConfig:
        """Translate to the planning backend's own config.

        Returns:
            The equivalent PlannerConfig; max_leg_distance 0 becomes None, which
            makes the planner derive the limit from the map.
        """
        return PlannerConfig(
            margin=self.margin,
            grid_spacing=self.grid_spacing,
            merge_threshold=self.merge_threshold,
            min_leg_distance=self.min_leg_distance,
            max_leg_distance=self.max_leg_distance or None,
        )

    def __str__(self) -> str:
        return self.model_dump_json(indent=4)


class GraphRoutePlanner(Node):
    """Expands a mission's via points into a route that stays on water.

    The mission is planned once, when it arrives, using the wind known at that
    moment. It is deliberately not replanned afterwards: `route_planner` builds
    a fresh Mission from every route it receives, so republishing mid-course
    would send the boat back to the first via point.
    """

    def __init__(self, **kwargs):
        super().__init__(
            "graph_route_planner", automatically_declare_parameters_from_overrides=True, **kwargs
        )

        # STATE - INIT
        try:
            self.config: Config = self._load_config()
            self.get_logger().info(f"Config loaded successfully:\n{self.config}")
        except ValueError as err:
            self.get_logger().error(f"Config validation failed: {err}")
            return

        try:
            self.map: SailMap = self._load_map()
        except (OSError, ValueError) as err:
            self.get_logger().error(f"Could not load map: {err}")
            return

        self.wind_direction: float | None = None

        # A mission waits here until the wind is known, since the route depends
        # on it. Cleared once planned, so each mission is expanded once.
        self._pending_mission: list[Position] | None = None

        # LOGGING - TIMER
        # Missing inputs are normal at startup; warn, but not on every tick.
        self._last_missing_wind_log_time: datetime = datetime.now()

        # SUBSCRIBER - INIT
        self.create_subscription(RouteMsg, "/planning/mission", self._on_mission_change, 1)
        self.create_subscription(Wind, "/sense/wind", self._on_wind_change, 1)

        # PUBLISHER - INIT
        self._publisher_target_route = self.create_publisher(
            RouteMsg, "/planning/target_route", ROUTE_QOS
        )

        # TIMER for planning
        self._timer = self.create_timer(self.config.plan_interval, self._plan_timer_callback)

    def _load_config(self) -> Config:
        return Config(
            map_path=self.get_parameter("map_path").value,
            plan_interval=self.get_parameter("plan_interval").value,
            margin=self.get_parameter("margin").value,
            grid_spacing=self.get_parameter("grid_spacing").value,
            merge_threshold=self.get_parameter("merge_threshold").value,
            min_leg_distance=self.get_parameter("min_leg_distance").value,
            max_leg_distance=self.get_parameter("max_leg_distance").value,
            time_between_missing_values_logs=self.get_parameter(
                "time_between_missing_values_logs"
            ).value,
        )

    def _load_map(self) -> SailMap:
        path = self.config.map_path or str(DEFAULT_MAP)
        sail_map = load_kml(path)
        bodies = len(sail_map.components)
        self.get_logger().info(
            f"Loaded map '{sail_map.name}': {len(sail_map.features)} features, "
            f"{bodies} navigable body(ies), {sail_map.water.area:.0f} m² of water"
        )
        if bodies > 1:
            # Only one body is reachable from any given point; the rest need a portage.
            self.get_logger().info(
                f"Map has {bodies} separate bodies of water — via points must all lie "
                f"in the same one."
            )
        return sail_map

    # ── subscriptions ─────────────────────────────────────────────────────────

    def _on_wind_change(self, msg: Wind) -> None:
        if not (0 <= msg.angle <= 360):
            self.get_logger().error(f"Invalid wind angle: {msg.angle}. Must be between 0 and 360 degrees.")
            return
        self.wind_direction = msg.angle

    def _on_mission_change(self, msg: RouteMsg) -> None:
        """Accept a new mission: the via points to round, in order."""
        via = [(float(wp.east), float(wp.north)) for wp in msg.waypoints]

        if len(via) < 2:
            self.get_logger().error(
                f"Mission has {len(via)} via point(s); at least 2 are needed to plan between them."
            )
            return

        # Reject the whole mission if any via point is unreachable, rather than
        # silently planning a course that skips one.
        off_water = [p for p in via if self.map.component_for(p) is None]
        if off_water:
            self.get_logger().error(f"Mission via points not on water: {off_water} — ignoring mission.")
            return

        self.get_logger().info(f"New mission received: {len(via)} via points {via}")
        self._pending_mission = via

    # ── planning ──────────────────────────────────────────────────────────────

    def _plan_timer_callback(self) -> None:
        if self._pending_mission is None:
            return

        if self.wind_direction is None:
            self._last_missing_wind_log_time = self._throttled_warn(
                "Mission waiting: no wind published yet", self._last_missing_wind_log_time
            )
            return

        via = self._pending_mission
        model = SailingModel.from_bearing(self.wind_direction)
        route = self._plan_mission(via, model)

        # Whether it worked or not, this mission has had its turn; a new one
        # must be published to try again.
        self._pending_mission = None

        if route is None:
            return

        self._publish_route(route)
        self.get_logger().info(
            f"Published route: {len(via)} via points expanded to {len(route)} waypoints, "
            f"wind from {self.wind_direction:.0f}°"
        )

    def _plan_mission(self, via: list[Position], model: SailingModel) -> list[Position] | None:
        """Plan the water between each consecutive pair of via points.

        Args:
            via: The via points to round, in order.
            model: The boat's sailing model for the current wind.

        Returns:
            The concatenated route, or None if any leg could not be planned.
            The via points themselves always appear, in order.
        """
        config = self.config.to_planner_config()
        route: list[Position] = []

        for index, (start, goal) in enumerate(zip(via, via[1:])):
            try:
                leg = plan_route(self.map.water, start, goal, model=model, config=config)
            except NoWaterError as err:
                self.get_logger().error(f"Leg {index} ({start} -> {goal}) cannot be planned: {err}")
                return None
            except ValueError as err:
                self.get_logger().error(f"Invalid planner configuration: {err}")
                return None

            if not leg.found:
                self.get_logger().error(
                    f"Leg {index} ({start} -> {goal}): no route found — try a smaller margin."
                )
                return None

            self.get_logger().info(
                f"  leg {index}: {start} -> {goal} = {len(leg.waypoints)} waypoints, "
                f"{leg.duration:.1f} time units"
            )
            # Drop the leg's first point: it is the previous leg's last.
            route.extend(leg.waypoints if not route else leg.waypoints[1:])

        return route

    def _publish_route(self, waypoints: list[Position]) -> None:
        route_msg = RouteMsg()
        for east, north in waypoints:
            route_msg.waypoints.append(LocationMsg(east=east, north=north))
        self._publisher_target_route.publish(route_msg)

    def _throttled_warn(self, message: str, last_logged: datetime) -> datetime:
        """Log a warning at most every `time_between_missing_values_logs` seconds."""
        if (datetime.now() - last_logged).total_seconds() > self.config.time_between_missing_values_logs:
            self.get_logger().warn(message)
            return datetime.now()
        return last_logged


def main(args: list[str] | None = None) -> None:
    """The main entry point of this node."""
    rclpy.init(args=args)

    node = GraphRoutePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
