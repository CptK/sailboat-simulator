"""This module contains the definition of the WayAction class and its subclasses."""

from abc import ABC
from abc import abstractmethod
import numpy as np
from typing import cast

from route_planner.waypoint import WayPoint


MAX_RECURSION_DEPTH_FOR_OPTIMIZATION = 10


class WayAction(ABC):
    """This class represents an action that the robot can take to move from one point to another."""

    def __init__(self, start: WayPoint, target: WayPoint, logger = None) -> None:
        """Initializes the action with the start and target waypoints.

        Args:
            start (WayPoint): The start waypoint of the action.
            target (WayPoint): The target waypoint of the action.

        Raises:
            ValueError: If the target waypoint is soft.
            ValueError: If the start waypoint is not a WayPoint.
            ValueError: If the target waypoint is not a WayPoint.
        """
        if not isinstance(start, WayPoint):
            raise ValueError("Start must be a WayPoint.")

        if not isinstance(target, WayPoint):
            raise ValueError("Target must be a WayPoint.")

        self.start: WayPoint = start
        self.target: WayPoint = target
        self.__logger = logger

        if target.is_soft:
            raise ValueError("Target waypoint cannot be soft.")
        # start waypoint can be soft, as it can be the current location of the robot

    @abstractmethod
    def plan(self, wind_direction: float) -> list[WayPoint]:  # pragma: no cover
        """This method generates a path that the robot can take to move from one point to another.

        Returns:
            The route that needs to be followed by the robot.
        """
        pass

    def update_start(self, start: WayPoint):  # pragma: no cover
        """Updates the start location of the action.

        Args:
            start (CartesianLocation): The new start location.
        """
        self.start = start

    def _log(self, message: str, level: str = "info") -> None:
        """Logs a message using the logger.

        Args:
            message (str): The message to log.
            level (str): The logging level (default is "info").
        """
        if self.__logger:
            if level == "info":
                self.__logger.info(message)
            elif level == "warning":
                self.__logger.warning(message)
            elif level == "error":
                self.__logger.error(message)
            else:
                raise ValueError(f"Unknown logging level: {level}")
        else:
            print(f"[{level.upper()}] {message}")


class StraightWayAction(WayAction):
    """This class represents a straight path from the start to the target waypoint."""

    def plan(self, wind_direction: float) -> list[WayPoint]:
        return [self.start, self.target]


class TackingWayAction(WayAction):
    """This class represents a tacking path from the start to the target waypoint."""

    def __init__(self, start: WayPoint, target: WayPoint, min_tack_angle: float, logger = None) -> None:
        super().__init__(start, target, logger)
        self.tack_angle = min_tack_angle  # Minimum angle between the wind and the boat's heading to tack

    def plan(self, wind_direction: float) -> list[WayPoint]:
        start_x, start_y = self.start.east, self.start.north
        target_x, target_y = self.target.east, self.target.north

        # Calculate angle of line between start and target
        direction = np.array([target_x - start_x, target_y - start_y])

        # Continue with normal tacking logic - this code stays exactly the same
        wind_angle = np.radians(wind_direction)
        tack_angle = np.radians(self.tack_angle)

        # Calculate tack vectors ensuring minimum angle to wind
        # We need to use sin, cos and not cos, sin because of 0° = North and 90° = East
        right_tack = np.array([np.sin(wind_angle + tack_angle), np.cos(wind_angle + tack_angle)])
        left_tack = np.array([np.sin(wind_angle - tack_angle), np.cos(wind_angle - tack_angle)])

        # Choose tack direction based on progress while ensuring minimum angle
        right_progress = np.dot(right_tack, direction)
        left_progress = np.dot(left_tack, direction)

        if max(right_progress, left_progress) <= 0:
            # Both tacks are unsuitable - this shouldn't be a tacking problem
            # Fall back to direct path (caller should have used StraightWayAction)
            self._log("Both tacks unsuitable, falling back to direct path.", level="warning")
            return [self.start, self.target]

        tack_1, tack_2 = (
            (right_tack, left_tack) if right_progress > left_progress else (left_tack, right_tack)
        )

        # Calculate line to target for intersection checks
        line_to_target = direction / np.linalg.norm(direction)

        route = [self.start]
        max_tack_distance = np.linalg.norm(direction) / 3  # TODO: Make this a parameter

        while not self._overshoot(route[-1], line_to_target):
            current = np.array([route[-1].east, route[-1].north])

            if len(route) % 2 == 1:
                # Tack out at the chosen angle
                next_point = current + max_tack_distance * tack_1
                route.append(WayPoint.from_coordinates(east=next_point[0], north=next_point[1], is_soft=True))
            else:
                next_point = self._line_intersection(route[0], line_to_target, route[-1], tack_2)
                route.append(next_point)

        # Remove overshooting point and add clean final approach
        route = route[:-1]
        final_point = self._line_intersection(route[-1], tack_1, self.target, tack_2)
        route.append(final_point)
        route.append(self.target)

        route = self._optimize_last_part(route, tack_1, tack_2)
        route = self._clean_points(route, threshold=0.001)

        return route

    def _overshoot(self, point: WayPoint, line_direction: np.ndarray) -> bool:
        """Check if point overshoots target."""
        target_vec = np.array([self.target.east - self.start.east, self.target.north - self.start.north])
        point_vec = np.array([point.east - self.start.east, point.north - self.start.north])
        # Project both vectors onto line direction
        target_proj = np.dot(target_vec, line_direction)
        point_proj = np.dot(point_vec, line_direction)
        return cast(bool, point_proj > target_proj)

    def _line_intersection(
        self, point1: WayPoint, direction1: np.ndarray, point2: WayPoint, direction2: np.ndarray
    ) -> WayPoint:
        """Calculate intersection of two lines."""
        A = np.vstack([direction1, direction2]).T  # pylint: disable=invalid-name
        d = np.array([point1.east - point2.east, point1.north - point2.north])  # pylint: disable=invalid-name
        x, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
        intersection = np.array([point1.east, point1.north]) - x[0] * direction1
        return WayPoint.from_coordinates(east=intersection[0], north=intersection[1], is_soft=True)

    def _optimize_last_part(
        self, route: list[WayPoint], tack1: np.ndarray, tack2: np.ndarray, recursion_depth: int = 0
    ) -> list[WayPoint]:
        if recursion_depth > MAX_RECURSION_DEPTH_FOR_OPTIMIZATION or len(route) < 4:
            return route

        # get the distance between two consecutive points
        route_np = np.array([[point.east, point.north] for point in route])
        distances = np.linalg.norm(np.diff(route_np, axis=0), axis=1)
        mean_distance = np.mean(distances)

        if distances[-1] > mean_distance * 2 / 3:
            return route

        # keep a reference to the target and remove it from the route
        target = route[-1]
        route = route[:-1]

        # get the vector of the last tack to the target of the current route
        m1_to_target = target - route[-1]

        # we want to merge the last two points before the target, so we add the vector from the last tack to
        # the second to last point which becomes the merged point
        intermediate = route[-2] + m1_to_target
        route = route[:-2]
        route.append(intermediate)

        # re-add the target to the route
        route.append(target)

        # check if the new route is already optimal or if we need to optimize it further
        return self._optimize_last_part(route, tack1, tack2, recursion_depth + 1)

    def _clean_points(self, route: list[WayPoint], threshold: float) -> list[WayPoint]:
        """Remove points that are too close to each other."""
        done = False
        i = 1
        while not done:
            p1 = np.array([route[i - 1].east, route[i - 1].north])  # pylint: disable=invalid-name
            p2 = np.array([route[i].east, route[i].north])  # pylint: disable=invalid-name
            distance = np.linalg.norm(p1 - p2)
            if distance < threshold:
                route.pop(i)
            else:
                i += 1

            if i >= len(route):
                done = True

        return route


if __name__ == "__main__":  # pragma: no cover
    start = WayPoint.from_coordinates(east=0, north=0)
    target = WayPoint.from_coordinates(east=1, north=1)
    wind_direction = -90  # pylint: disable=C0103

    way_action = TackingWayAction(start, target, 45)

    route = way_action.plan(wind_direction)

    import matplotlib.pyplot as plt

    points = np.array([point.to_numpy() for point in route])
    print(points)

    # Plot wind direction grid
    x_min, x_max = -2, 2  # Plotting bounds
    y_min, y_max = -1, 2

    # Convert from compass/nautical angles (0° = North) to mathematical angles (0° = East)
    math_angle = np.radians(90 - wind_direction)  # Convert to mathematical angle
    wind_vector = np.array([np.cos(math_angle), np.sin(math_angle)])

    # Create grid of starting points perpendicular to wind direction
    perp_vector = np.array([-wind_vector[1], wind_vector[0]])

    # Create lines parallel to wind direction
    for t in np.arange(-2, 2, 0.1):
        start_point = t * perp_vector
        line_x = [start_point[0] + s * wind_vector[0] for s in [-2, 2]]
        line_y = [start_point[1] + s * wind_vector[1] for s in [-2, 2]]
        plt.plot(line_x, line_y, color="lightgray", linestyle="-", alpha=0.3, zorder=1)

    plt.plot(points[:, 0], points[:, 1], label="Tacking path")
    plt.scatter(points[:, 0], points[:, 1])
    plt.plot([start.east, target.east], [start.north, target.north], "r--", label="Straight path")
    plt.legend()
    plt.title(f"Wind direction: {wind_direction}")

    min_x, max_x = min(points[:, 0]), max(points[:, 0])
    min_y, max_y = min(points[:, 1]), max(points[:, 1])

    plt.xlim(min_x - 0.05, max_x + 0.05)
    plt.ylim(min_y - 0.05, max_y + 0.05)

    plt.savefig("path.png")
