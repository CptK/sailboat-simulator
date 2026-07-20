"""The node that expands a coarse mission into a sailable route, and keeps it valid.

A mission is the via points you want to round, in order — the corners of a
course, say. This node plans the water between them: each consecutive pair is
routed around land and islands over a visibility graph, and the segments are
concatenated into one detailed route, starting from where the boat actually is.

It also owns *replanning*. If the boat drifts off the planned route, or the wind
shifts enough to make a leg unsailable, it plans again from the boat's current
position through the via points still ahead, and publishes the new route. The
steering node simply receives it — deciding the path is this node's job alone.

    /planning/mission ─┐
    /sense/wind        ─┼─> graph_route_planner ──> /planning/target_route
    /sense/boat_info   ─┘                                    │
                                                             v
                                                     route_follower ──> /planning/desired_heading

Frames: the map is projected to metres about its own bounding-box centre, and
those metres are published directly as Location.east / Location.north. This
assumes the map's centre is the world origin the simulation reports positions
in. Nothing in the stack carries a geodetic reference to check that against, so
if the two ever diverge the routes will be silently offset.
"""

import math
from datetime import datetime, timedelta

import rclpy
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shapely.geometry import LineString, Point

# Boat messages
from boat_msgs.msg import BoatInfo
from boat_msgs.msg import Location as LocationMsg
from boat_msgs.msg import Route as RouteMsg
from boat_msgs.msg import Wind

# Planning backend — plain Python, no ROS.
from graph_route_planner import DEFAULT_MAP
from graph_route_planner.map_loader import SailMap, load_kml
from graph_route_planner.planner import (
    CachedPlanner,
    NoWaterError,
    PlannerConfig,
    Position,
)
from graph_route_planner.sailing import SailingModel

#: The route is state, not a stream. It is published once per plan, so a
#: follower that starts late — or restarts — would otherwise wait indefinitely
#: for a route that has already been sent. Latching hands the current route to
#: every late subscriber. A TRANSIENT_LOCAL publisher still satisfies a VOLATILE
#: subscriber, so this stays compatible with default-QoS subscriptions.
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

    # Boat
    min_tack_angle: float = Field(
        ..., gt=0, lt=90,
        description="Closest the boat can sail to the wind, degrees; legs inside this must tack",
    )

    # Cadence and replanning
    plan_interval: float = Field(..., gt=0, description="Plan interval must be larger than 0")
    dist_threshold: float = Field(
        ..., ge=0, description="Radius in metres within which a via point counts as rounded"
    )
    off_course_threshold: float = Field(
        ..., ge=0, description="Distance from the planned route, in metres, that triggers a replan"
    )
    min_time_between_replanning: float = Field(
        ..., ge=0, description="Minimum seconds between two replans"
    )
    no_go_replan_margin: float = Field(
        5.0, ge=0,
        description="Degrees inside the no-go zone before drift triggers a replan",
    )

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
    segment_cost: float = Field(
        1.0, ge=0,
        description="Fixed time charge per route leg, biasing the search toward fewer turns",
    )

    # Logging
    time_between_missing_values_logs: float = Field(
        5.0, ge=0, description="Seconds between repeated warnings about missing inputs"
    )

    @model_validator(mode="after")
    def validate_no_go_margin(self) -> "Config":
        if self.no_go_replan_margin >= self.min_tack_angle:
            raise ValueError(
                f"no_go_replan_margin ({self.no_go_replan_margin}) must be below "
                f"min_tack_angle ({self.min_tack_angle}), or drift can never "
                f"trigger a replan at all"
            )
        return self

    @field_validator("plan_interval")
    @classmethod
    def validate_plan_interval(cls, plan_interval: float) -> float:
        # Planning is synchronous and takes ~0.1-0.6 s per leg on a pond-sized
        # map, so a fast interval starves the executor rather than helping.
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
            segment_cost=self.segment_cost,
        )

    def __str__(self) -> str:
        return self.model_dump_json(indent=4)


class GraphRoutePlanner(Node):
    """Plans a mission's route across the water, and replans it when it goes stale.

    Every published route starts at the boat's current position and runs through
    the via points still ahead of it. That matters because the follower restarts
    at the first waypoint of whatever route it receives: a replan that began at
    the original start would send the boat back to the beginning of the mission.
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

        # The planner keeps its visibility graph between calls. Building it is
        # ~96% of a plan and depends only on the map and the search settings,
        # both fixed for this node's lifetime — so it happens once, on the first
        # plan, and every later plan reuses it. A wind change re-costs it; a new
        # boat position is attached with a single O(n) pass.
        self.planner = CachedPlanner(self.map.water, self.config.to_planner_config())

        self.boat: Position | None = None
        self.wind_direction: float | None = None

        # The mission and how far through it the boat has got. via_index is the
        # next via point to round; everything before it is done.
        self.mission: list[Position] = []
        self.via_index: int = 0
        self.mission_complete: bool = False

        self.planned_route: list[Position] | None = None
        self.last_plan_time: datetime = datetime.min

        # Index of the route waypoint the boat is steering at, mirroring
        # route_follower's own target_index. Replanning needs it to tell the leg
        # being sailed from the ones already behind the boat.
        self.route_index: int = 0

        # LOGGING - TIMER
        self._last_missing_wind_log_time: datetime = datetime.now()
        self._last_missing_boat_log_time: datetime = datetime.now()

        # SUBSCRIBER - INIT
        self.create_subscription(RouteMsg, "/planning/mission", self._on_mission_change, 1)
        self.create_subscription(Wind, "/sense/wind", self._on_wind_change, 1)
        self.create_subscription(BoatInfo, "/sense/boat_info", self._on_boat_info, 1)

        # PUBLISHER - INIT
        self._publisher_target_route = self.create_publisher(
            RouteMsg, "/planning/target_route", ROUTE_QOS
        )

        # TIMER for planning
        self._timer = self.create_timer(self.config.plan_interval, self._plan_timer_callback)

    def _load_config(self) -> Config:
        return Config(
            map_path=self.get_parameter("map_path").value,
            min_tack_angle=self.get_parameter("min_tack_angle").value,
            plan_interval=self.get_parameter("plan_interval").value,
            dist_threshold=self.get_parameter("dist_threshold").value,
            off_course_threshold=self.get_parameter("off_course_threshold").value,
            min_time_between_replanning=self.get_parameter("min_time_between_replanning").value,
            no_go_replan_margin=self.get_parameter("no_go_replan_margin").value,
            margin=self.get_parameter("margin").value,
            grid_spacing=self.get_parameter("grid_spacing").value,
            merge_threshold=self.get_parameter("merge_threshold").value,
            min_leg_distance=self.get_parameter("min_leg_distance").value,
            max_leg_distance=self.get_parameter("max_leg_distance").value,
            segment_cost=self.get_parameter("segment_cost").value,
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
            self.get_logger().info(
                f"Map has {bodies} separate bodies of water — the boat and every via "
                f"point must lie in the same one."
            )
        return sail_map

    def _sailing_model(self) -> SailingModel:
        """The boat's sailing model for the wind now blowing."""
        return SailingModel.from_bearing(
            self.wind_direction or 0.0, no_go_deg=self.config.min_tack_angle
        )

    def _throttled_warn(self, message: str, last_logged: datetime) -> datetime:
        """Log a warning at most every `time_between_missing_values_logs` seconds."""
        if (datetime.now() - last_logged).total_seconds() > self.config.time_between_missing_values_logs:
            self.get_logger().warn(message)
            return datetime.now()
        return last_logged

    # ── subscriptions ─────────────────────────────────────────────────────────

    def _on_wind_change(self, msg: Wind) -> None:
        if not (0 <= msg.angle <= 360):
            self.get_logger().error(f"Invalid wind angle: {msg.angle}. Must be between 0 and 360 degrees.")
            return
        self.wind_direction = msg.angle

    def _on_boat_info(self, msg: BoatInfo) -> None:
        self.boat = (float(msg.x), float(msg.y))
        self._update_progress()
        self._advance_route_index()

    def _on_mission_change(self, msg: RouteMsg) -> None:
        """Accept a new mission: the via points to round, in order."""
        via = [(float(wp.east), float(wp.north)) for wp in msg.waypoints]

        if not via:
            self.get_logger().error("Mission has no via points — ignoring it.")
            return

        off_water = [p for p in via if self.map.component_for(p) is None]
        if off_water:
            self.get_logger().error(f"Mission via points not on water: {off_water} — ignoring mission.")
            return

        self.get_logger().info(f"New mission received: {len(via)} via points {via}")
        self.mission = via
        self.via_index = 0
        self.mission_complete = False
        self.planned_route = None          # forces an immediate plan
        self.route_index = 0
        self.last_plan_time = datetime.min

    # ── mission progress ──────────────────────────────────────────────────────

    def _update_progress(self) -> None:
        """Advance past via points the boat has rounded.

        Uses the same distance threshold as the follower, so the two agree on
        when a via point counts as reached.
        """
        if not self.mission or self.mission_complete or self.boat is None:
            return

        while self.via_index < len(self.mission):
            target = self.mission[self.via_index]
            if Point(self.boat).distance(Point(target)) >= self.config.dist_threshold:
                break
            self.via_index += 1
            self.get_logger().info(
                f"Rounded via point {self.via_index}/{len(self.mission)} {target}"
            )

        if self.via_index >= len(self.mission):
            self.mission_complete = True
            self.get_logger().info("All via points rounded; mission complete.")

    def _advance_route_index(self) -> None:
        """Advance past route waypoints the boat has reached.

        Uses route_follower's threshold so both agree on which leg is being
        sailed. The index only moves forward: a boat blown back past a waypoint
        is off course, which is a reason to replan rather than to rewind.
        """
        route, boat = self.planned_route, self.boat
        if route is None or boat is None:
            return

        while (self.route_index < len(route) - 1
               and Point(boat).distance(Point(route[self.route_index]))
               < self.config.dist_threshold):
            self.route_index += 1

    # ── planning ──────────────────────────────────────────────────────────────

    def _plan_timer_callback(self) -> None:
        if not self.mission or self.mission_complete:
            return

        if self.boat is None:
            self._last_missing_boat_log_time = self._throttled_warn(
                "Mission waiting: no boat position published yet", self._last_missing_boat_log_time
            )
            return

        if self.wind_direction is None:
            self._last_missing_wind_log_time = self._throttled_warn(
                "Mission waiting: no wind published yet", self._last_missing_wind_log_time
            )
            return

        if self.planned_route is None:
            self._plan_and_publish("initial plan")
            return

        # Replanning is rate-limited so a persistent trigger cannot thrash.
        if (datetime.now() - self.last_plan_time) < timedelta(
            seconds=self.config.min_time_between_replanning
        ):
            return

        reason = self._replan_reason()
        if reason is not None:
            self._plan_and_publish(reason)

    def _replan_reason(self) -> str | None:
        """Why the current route should be replanned, or None to keep it.

        Three triggers, and the order matters — the cheapest and most specific
        first:

        1. The boat can no longer sail at the waypoint it is steering at. This
           is the one that catches leeway. The route's own waypoints never move,
           so checking only those can never notice the boat sliding downwind;
           the bearing that goes bad is the live one, from the boat to its
           target. It degrades fastest just as the waypoint is nearly reached,
           because a fixed sideways offset subtends a larger angle the closer
           the boat gets.

           It fires only once the bearing is `no_go_replan_margin` degrees past
           the tack limit. Upwind legs are laid at exactly that limit, so the
           live bearing rides the threshold for the whole beat; without the
           margin every wobble replans, and each replan restarts the follower's
           maneuver, so the boat thrashes instead of sailing.
        2. The boat has strayed too far from the leg it should be on. Measured
           against the *remaining* route only. Measuring against the whole
           polyline silently forgives drift on a course that doubles back near
           itself — this mission ends where it starts, so the last leg would be
           masked by the first.
        3. The wind has shifted enough that a leg still ahead cannot be sailed.
           Legs already behind the boat are irrelevant.
        """
        route, boat = self.planned_route, self.boat
        if route is None or boat is None or len(route) < 2:
            return None

        model = self._sailing_model()
        index = min(self.route_index, len(route) - 1)

        target = route[index]
        replan_below = self.config.min_tack_angle - self.config.no_go_replan_margin
        if Point(boat).distance(Point(target)) >= self.config.dist_threshold:
            angle = math.degrees(model.angle_off_wind(boat, target))
            if angle < replan_below:
                # One decimal: at .0f a bearing of 44.6 prints as "45 deg, past
                # the 45 deg limit", which reads like a contradiction in the log.
                return (f"drifted into the no-go zone: waypoint {index} now bears "
                        f"{angle:.1f}° off the wind, past the {replan_below:.0f}° replan "
                        f"limit ({self.config.min_tack_angle:.0f}° tack limit less "
                        f"{self.config.no_go_replan_margin:.0f}° margin)")

        # The leg being sailed starts at the waypoint before the target.
        remaining = route[max(index - 1, 0):]
        if len(remaining) >= 2:
            drift = LineString(remaining).distance(Point(boat))
            if drift > self.config.off_course_threshold:
                return f"off course by {drift:.1f} m"

        for u, v in zip(remaining, remaining[1:]):
            if not model.heading_is_sailable(u, v):
                return "wind shift made a leg unsailable"

        return None

    def _plan_and_publish(self, reason: str) -> None:
        """Plan from the boat through the remaining via points, and publish it."""
        boat = self.boat
        if boat is None:
            return

        # Always start at the boat: the follower restarts at waypoint 0 of any
        # route it receives, so a route starting anywhere else would send it back.
        points: list[Position] = [boat] + self.mission[self.via_index:]
        if len(points) < 2:
            return

        model = self._sailing_model()
        planned = self._plan_through(points, model)
        self.last_plan_time = datetime.now()

        if planned is None:
            return

        route, is_soft = planned
        self.planned_route = route
        self.route_index = 0
        self._publish_route(route, is_soft)
        self.get_logger().info(
            f"Published route ({reason}): {len(points) - 1} via point(s) ahead expanded to "
            f"{len(route)} waypoints, wind from {self.wind_direction:.0f}°, "
            f"tack limit {self.config.min_tack_angle:.0f}°"
        )

    def _plan_through(self, points: list[Position],
                      model: SailingModel) -> tuple[list[Position], list[bool]] | None:
        """Plan the water between each consecutive pair of points.

        Args:
            points: Positions to visit in order, starting at the boat.
            model: The boat's sailing model for the current wind.

        Returns:
            ``(route, is_soft)`` of equal length, or None if any leg could not be
            planned. A waypoint is soft when the search invented it to get
            somewhere — a tacking corner — and hard when it is a mission via
            point that must actually be rounded. The follower needs the
            difference: it may cut the corners, but not the course.
        """
        route: list[Position] = []
        is_soft: list[bool] = []

        for index, (start, goal) in enumerate(zip(points, points[1:])):
            try:
                leg = self.planner.plan(start, goal, model)
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
            added = leg.waypoints if not route else leg.waypoints[1:]
            route.extend(added)
            # Everything the search produced is soft, except this leg's goal,
            # which is the via point the leg was planned to reach. The boat's own
            # starting position stays soft: it is where the route begins, not
            # somewhere it has to go.
            is_soft.extend([True] * len(added))
            if added:
                is_soft[-1] = False

        return route, is_soft

    def _publish_route(self, waypoints: list[Position], is_soft: list[bool]) -> None:
        route_msg = RouteMsg()
        for (east, north), soft in zip(waypoints, is_soft):
            route_msg.waypoints.append(LocationMsg(east=east, north=north, is_soft=soft))
        self._publisher_target_route.publish(route_msg)


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
