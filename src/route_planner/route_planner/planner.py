"""A mission planner that generates a path for a robot to follow."""

from datetime import datetime
from datetime import timedelta

from route_planner.maneuver_action import JibeManeuver
from route_planner.maneuver_action import ManeuverAction
from route_planner.maneuver_action import TackManeuver
from route_planner.segment_action import StraightWayAction
from route_planner.segment_action import TackingWayAction
from route_planner.segment_action import WayAction
from route_planner.utils import angular_distance
from route_planner.utils import bearing
from route_planner.utils import is_sailable_direction
from route_planner.utils import point_to_line_distance
from route_planner.waypoint import WayPoint


class Mission:
    """A mission planner that generates a path for a robot to follow.

    Args:
        waypoints (list[WayPoint]): A list of waypoints that the planner will use to generate a path.
            The first waypoint is not the current location of the robot, but the starting point of the path.
        dist_threshold (float): The distance threshold at which a waypoint is considered reached (meters).
        off_course_threshold (float): The threshold at which the robot is considered off course (meters).
        min_tack_angle (float): The minimum angle between the wind direction and the path direction for the
            path to be considered sailable (degrees).
        safe_jibe_threshold (float): The wind speed threshold at which jibing is considered safe (km/h).
        maneuver_heading_step_size (float): The step size in degrees that the robot will use to adjust its
            heading during a maneuver.
        command_step_size_multiplier (float): The multiplier that is applied to the maneuver heading step size
            to determine the step size of the command that is sent to the robot.
        heading_tolerance (float): The tolerance in degrees that is allowed for the robot's heading to be off
            from the target heading.
        max_steps_to_cross_wind_when_tacking (int): The maximum number of steps that may be used to cross the
            wind when tacking before the maneuver is considered stuck.
    """

    def __init__(
        self,
        waypoints: list[WayPoint],
        dist_threshold: float = 0.5,
        off_course_threshold: float = 3,
        min_tack_angle: float = 45,
        safe_jibe_threshold: float = 25,
        maneuver_heading_step_size: float = 10,
        command_step_size_multiplier: float = 1.5,
        heading_tolerance: float = 5,
        max_steps_to_cross_wind_when_tacking: int = 40,
        logger=None,
    ) -> None:
        """Initializes the mission planner with the given waypoints and parameters."""
        self.hard_waypoints = waypoints
        self.dist_threshold = dist_threshold
        self.off_course_threshold = off_course_threshold
        self.min_tack_angle = min_tack_angle
        self.safe_jibe_threshold = safe_jibe_threshold
        self.maneuver_heading_step_size = maneuver_heading_step_size
        self.command_step_size_multiplier = command_step_size_multiplier
        self.heading_tolerance = heading_tolerance
        self.max_steps_to_cross_wind_when_tacking = max_steps_to_cross_wind_when_tacking
        self.logger = logger
        self.min_time_between_replanning = timedelta(seconds=5)
        self.last_replanning_time = datetime.now() - timedelta(seconds=60)

        self.completed = False
        self.maneuver_action: ManeuverAction | None = None
        self.way_actions: list[WayAction]
        self.route: list[WayPoint]

    def plan(self, location: WayPoint, wind_direction: float) -> list[WayPoint]:
        """Method for planning the mission.

        Args:
            location (WayPoint): The current location of the robot.
        """
        # set the start of the segment to the current location of the robot, as we start from there
        self.way_actions = self._build_submissions(location, wind_direction)
        self._previous_waypoint = location
        self.route = []

        # This error should never be raised because the node should make sure that this method is only called
        # when there is a valid wind direction
        if wind_direction is None:
            raise ValueError("Wind direction is None. Cannot plan route.")

        for i, way_action in enumerate(self.way_actions):
            points = way_action.plan(wind_direction)

            if len(self.route) > 0:
                # remove the first point as it is the same as the last point of the previous route
                is_soft = self.route[-1].is_soft and way_action.start.is_soft
                self.route[-1].is_soft = is_soft
                points = points[1:]

            self.route.extend(points)

        self.route = self.route[
            1:
        ]  # remove the first point as it is the same as the current location of the robot

        self.last_replanning_time = datetime.now()
        return self.route

    def step(self, location: WayPoint, heading: float, wind_direction: float, wind_speed: float) -> float:
        """Execute one navigation step using heading-difference based maneuvering."""
        # This error should never be raised because the node should make sure that this method is only called
        # when there is a valid wind direction
        if wind_direction is None:
            raise ValueError("Wind direction is None. Cannot plan route.")

        if self.completed:
            return 0.0

        maneuver_threshold = 70.0  # degrees

        # Check if waypoint reached FIRST (before calculating heading to potentially wrong waypoint)
        # When close to a waypoint, bearing calculations become unstable and could trigger spurious maneuvers.
        # First check if we're close enough to the next hard waypoint while on soft waypoints.
        # This prevents unnecessary short tacks when we're already within reach of the hard waypoint.
        if (
            self.current_waypoint.is_soft
            and len(self.hard_waypoints) > 0
            and self.hard_waypoints[0].distance(location) < self.dist_threshold
        ):
            self._log("Within reach of hard waypoint, skipping remaining soft waypoints")
            # Remove all soft waypoints until we reach the hard waypoint
            while len(self.route) > 1 and self.route[0].is_soft:
                self.route.pop(0)

        if self.current_waypoint.distance(location) < self.dist_threshold:
            self._log("Reached waypoint")

            # Check if this was the last waypoint
            if len(self.route) == 1:
                self._log("Reached final waypoint. Mission is completed.")
                self.completed = True
                return 0.0

            # Remove hard waypoint if needed
            if not self.current_waypoint.is_soft:
                self._log("Reached hard waypoint, removing it from the route")
                self.way_actions.pop(0)
                self.hard_waypoints.pop(0)

            # Move to next waypoint
            self.move_to_next_waypoint()

            # Replan after reaching waypoint
            self.plan(location, wind_direction)
            self._log("Replanned route after reaching waypoint")

            # Clear any active maneuver since we're now targeting a new waypoint
            self.maneuver_action = None

        # Now calculate desired heading to the (correct) current waypoint
        desired_heading = bearing(location, self.current_waypoint)

        # Calculate heading difference (handling circular math)
        heading_diff = angular_distance(heading, desired_heading)

        # Handle active maneuver
        if self.maneuver_action is not None:
            if self.maneuver_action.complete:
                self._log("Maneuver completed")
                self.maneuver_action = None
                # Continue with normal navigation logic below
            else:
                # Continue active maneuver
                if self.maneuver_action.is_stuck(heading):
                    self._log("Tack maneuver stuck, switching to jibe")
                    self.maneuver_action = JibeManeuver(
                        target_heading=desired_heading,
                        current_heading=heading,
                        heading_step_size=self.maneuver_heading_step_size,
                        wind_direction=wind_direction,
                        heading_tolerance=self.heading_tolerance,
                        command_step_size_multiplier=self.command_step_size_multiplier,
                    )

                return self.maneuver_action.step(heading) or 0.0

        # No active maneuver - check if we need to start one based on heading difference
        if abs(heading_diff) > maneuver_threshold:
            self._log(f"Starting maneuver: heading difference = {heading_diff:.1f}°")
            self.maneuver_action = self._get_maneuver_action(
                location,
                self.current_waypoint,
                current_heading=heading,
                wind_direction=wind_direction,
                wind_speed=wind_speed
            )
            return self.maneuver_action.step(heading) or 0.0

        # Check if replanning needed due to wind/course changes
        if ((datetime.now() - self.last_replanning_time) > self.min_time_between_replanning) and \
            any([wind := self._critical_wind_change(location, wind_direction), off_course := self._off_course(location)]):
            cause = (
                "wind change and off course"
                if wind and off_course
                else "wind change"
                if wind
                else "off course"
            )
            self._log(f"Replanning route due to {cause}")
            self.plan(location, wind_direction)
            return bearing(location, self.current_waypoint)

        # Normal navigation - heading is acceptable, continue straight
        return desired_heading

    @property
    def current_waypoint(self) -> WayPoint:
        """Returns the waypoint that the robot is currently navigating to."""
        return self.route[0]

    @property
    def next_waypoint(self) -> WayPoint | None:
        """Returns the next waypoint that the robot will navigate to."""
        return self.route[1] if hasattr(self, "route") and len(self.route) > 1 else None

    @property
    def previous_waypoint(self) -> WayPoint:
        """Returns the previous waypoint that the robot navigated to."""
        return self._previous_waypoint

    @property
    def current_submission(self) -> WayAction | None:
        """Returns the current submission that the robot is executing."""
        return self.way_actions[0] if self.way_actions else None

    def move_to_next_waypoint(self) -> WayPoint:
        """Removes the first waypoint from the list (reached) and returns the next waypoint.

        Also updates the current submission to the next submission in the list.
        """
        self._previous_waypoint = self.route.pop(0)
        return self.route[0]

    def _build_submissions(self, start_location: WayPoint, wind_direction: float) -> list[WayAction]:
        submissions: list[WayAction] = []

        initial_way_action = self._get_way_action(start_location, self.hard_waypoints[0], wind_direction)
        submissions.append(initial_way_action)

        for i in range(len(self.hard_waypoints) - 1):
            start = self.hard_waypoints[i]
            target = self.hard_waypoints[i + 1]
            way_action = self._get_way_action(start, target, wind_direction)
            submissions.append(way_action)

        return submissions

    def _get_way_action(
        self,
        start:  WayPoint,
        end: WayPoint,
        wind_direction: float,
    ) -> WayAction:
        """Returns the way action to navigate from the start location to the end location.

        If the wind conditions allow for a direct path, return a WayAction with the direct path, otherwise,
        return a WayAction with a tacking path.
        """
        # This error should never be raised because the node should make sure that this method is only called
        # when there is a valid wind direction
        if wind_direction is None:
            raise ValueError("Wind direction is None. Cannot plan route.")

        # If wind is highly variable, be more conservative with straight paths
        min_tack_angle = self.min_tack_angle + 5  # be more conservative

        if is_sailable_direction(
            start,
            end,
            wind_direction,
            minimum_tack_angle=min_tack_angle,
        ):
            return StraightWayAction(start, end, self.logger)
        return TackingWayAction(start, end, min_tack_angle, self.logger)

    def _get_maneuver_action(
        self,
        current_postion: WayPoint,
        target: WayPoint,
        current_heading: float,
        wind_direction: float,
        wind_speed: float,
    ) -> ManeuverAction:
        # This error should never be raised because the node should make sure that this method is only called
        # when there is a valid wind direction
        if wind_direction is None:
            raise ValueError("Wind direction is None. Cannot plan route.")

        new_heading: float = bearing(current_postion, target)

        tack_maneuver = TackManeuver(
            target_heading=new_heading,
            current_heading=current_heading,
            heading_step_size=self.maneuver_heading_step_size,
            wind_direction=wind_direction,
            max_steps_to_cross_wind=self.max_steps_to_cross_wind_when_tacking,
            heading_tolerance=self.heading_tolerance,
            command_step_size_multiplier=self.command_step_size_multiplier,
        )
        jibe_maneuver = JibeManeuver(
            target_heading=new_heading,
            current_heading=current_heading,
            heading_step_size=self.maneuver_heading_step_size,
            wind_direction=wind_direction,
            heading_tolerance=self.heading_tolerance,
            command_step_size_multiplier=self.command_step_size_multiplier,
        )

        # Calculate wind angles relative to current and new headings
        # Using modulo 360 to keep angles in range
        current_wind_angle = (wind_direction - current_heading) % 360
        new_wind_angle = (wind_direction - new_heading) % 360

        # Convert angles > 180 to their negative complement
        # This gives us angles from -179 to +180, which is easier to reason about
        if current_wind_angle > 180:
            current_wind_angle -= 360  # pragma: no cover
        if new_wind_angle > 180:
            new_wind_angle -= 360  # pragma: no cover

        # If either position is close to the wind (< 90 degrees absolute angle), prefer tacking
        if abs(current_wind_angle) < 90 or abs(new_wind_angle) < 90:
            return tack_maneuver

        # If running or broad reach
        elif abs(current_wind_angle) > 120 and abs(new_wind_angle) > 120:
            # Strong winds make jibing dangerous
            if wind_speed is not None and wind_speed > self.safe_jibe_threshold:
                return tack_maneuver
            return jibe_maneuver

        # In intermediate cases, choose based on which maneuver requires less turning
        else:
            angle_difference = (new_heading - current_heading) % 360
            if angle_difference > 180:
                angle_difference -= 360  # pragma: no cover
            return tack_maneuver if abs(angle_difference) < 180 else jibe_maneuver

    def _off_course(self, location: WayPoint) -> bool:
        """Returns whether the robot is off course.

        Calculate the line from the location orthogonal to the line between the previous and current waypoint.
        If the distance is greater than the threshold, the robot is off course.
        """
        return (
            point_to_line_distance(self.previous_waypoint, self.current_waypoint, location)
            > self.off_course_threshold
        )

    def _critical_wind_change(self, position: WayPoint, wind_direction: float) -> bool:
        """Returns whether the wind conditions have changed critically."""
        if wind_direction is None:
            # Not enough measurements yet, use latest measurement
            return False  # or use the latest measurement from environment
        return not is_sailable_direction(
            position, self.current_waypoint, wind_direction, self.min_tack_angle
        )

    def _log(self, message: str) -> None:
        """Logs a message if a logger is provided."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


if __name__ == "__main__":  # pragma: no cover
    points = [[1, 1], [1, 5], [5, 5], [5, 1], [1, 1]]
    points = [[1,1], [5,5]]
    waypoints = [WayPoint.from_coordinates(e, n) for e, n in points]
    location = WayPoint(0.0, 0.0)

    wind_direction=340
    wind_speed=5

    mission = Mission(waypoints, dist_threshold=0.001)
    route = mission.plan(location, wind_direction)
    waypoints = mission.route

    hard_waypoints = [waypoint for waypoint in waypoints if not waypoint.is_soft]
    soft_waypoints = [waypoint for waypoint in waypoints if waypoint.is_soft]

    def plot_mission(
        waypoints: list[WayPoint],
        hard_waypoints: list[WayPoint],
        soft_waypoints: list[WayPoint],
        location: WayPoint,
        wind_direction: float,
        wind_speed: float,
        file_name: str = "mission_plot.png",
    ) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        hard_locations = [(w.east, w.north) for w in hard_waypoints]
        soft_locations = [(w.east, w.north) for w in soft_waypoints]
        all_locations = [(w.east, w.north) for w in waypoints]

        x_min = min(location.east, min([waypoint.east for waypoint in waypoints])) - 1
        x_max = max(location.east, max([waypoint.east for waypoint in waypoints])) + 1
        y_min = min(location.north, min([waypoint.north for waypoint in waypoints])) - 1
        y_max = max(location.north, max([waypoint.north for waypoint in waypoints])) + 1

        # Convert from compass/nautical angles (0° = North) to mathematical angles (0° = East)
        math_angle = np.radians(90 - wind_direction)  # Convert to mathematical angle
        wind_vector = np.array([np.cos(math_angle), np.sin(math_angle)])

        # Create grid of starting points perpendicular to wind direction
        perp_vector = np.array([-wind_vector[1], wind_vector[0]])

        # Create lines parallel to wind direction
        for t in np.arange(-100, 100, 0.5):
            start_point = t * perp_vector
            line_x = [start_point[0] + s * wind_vector[0] for s in [-100, 100]]
            line_y = [start_point[1] + s * wind_vector[1] for s in [-100, 100]]
            plt.plot(line_x, line_y, color="lightgray", linestyle="-", alpha=0.3, zorder=1)

        plt.scatter([location.east], [location.north], c="red")
        plt.scatter(
            [waypoint[0] for waypoint in hard_locations],
            [waypoint[1] for waypoint in hard_locations],
            c="blue",
        )
        plt.scatter(
            [waypoint[0] for waypoint in soft_locations],
            [waypoint[1] for waypoint in soft_locations],
            c="purple",
        )

        plt.plot([location[0] for location in all_locations], [location[1] for location in all_locations])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.savefig(file_name)
        plt.close()

    plot_mission(waypoints, hard_waypoints, soft_waypoints, location, wind_direction, wind_speed, "mission.png")

    # change wind to 45 degrees
    wind_direction = 45
    mission.step(location, 45, wind_direction, wind_speed)

    waypoints = mission.route
    hard_waypoints = [waypoint for waypoint in waypoints if not waypoint.is_soft]
    soft_waypoints = [waypoint for waypoint in waypoints if waypoint.is_soft]
    plot_mission(waypoints, hard_waypoints, soft_waypoints, location, wind_direction, wind_speed, "mission_plot_45.png")

    for i in range(10):
        # sample a location close to the current location
        import random
        new_location = WayPoint(
            location.east + random.uniform(-0.1, 0.11), location.north + random.uniform(-0.11, 0.11)
        )
        heading = random.uniform(0, 360)
        new_heading = mission.step(new_location, heading, wind_direction, wind_speed)
        