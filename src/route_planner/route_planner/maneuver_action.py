"""Module for defining the maneuver actions for the sailboat."""

from abc import ABC
from abc import abstractmethod


class ManeuverAction(ABC):
    """Base class for defining a maneuver action for the sailboat.

    This class is responsible for determining the next command to be sent to the sailboat
    to achieve a specific maneuver. It is designed to be used in a state machine to
    determine the next action to take.

    Attributes:
        target_heading: The target heading to reach.
        current_heading: The current heading of the sailboat.
        heading_step_size: The step size to take when turning the sailboat.
        command_step_size_multiplier: The multiplier to amplify the command step size.
        heading_tolerance: The tolerance for reaching the target heading.
        last_step_heading: The heading at the last step.
        current_step_heading: The heading at the current step.
        complete: A flag indicating if the maneuver is complete.
        clockwise: A flag indicating if the sailboat should turn clockwise.
        current_command: The current command to be sent to the sailboat.
    """

    def __init__(
        self,
        target_heading: float,
        current_heading: float,
        heading_step_size: float,
        wind_direction: float,
        command_step_size_multiplier: float,
        heading_tolerance: float,
    ) -> None:
        """Initialize the maneuver action with the required parameters

        Args:
            target_heading: The target heading to reach.
            current_heading: The current heading of the sailboat.
            heading_step_size: The step size to take when turning the sailboat.
            wind_direction: The direction of the wind.
            command_step_size_multiplier: The multiplier to amplify the command step size.
            heading_tolerance: The tolerance for reaching the target heading.
        """
        self._target_heading: float = target_heading
        self._current_heading: float = current_heading
        self._heading_step_size: float = heading_step_size
        self._wind_direction: float = wind_direction
        self._command_multiplier: float = command_step_size_multiplier
        self._heading_tolerance: float = heading_tolerance

        # Both these variables start at the current heading.
        self._last_step_heading: float = current_heading
        self._current_step_heading: float = current_heading

        self._complete: bool = False
        self._clockwise: bool = self._turn_clockwise()

        # Store the currently active command.
        self._current_command: float = self._calculate_next_command()

    def step(self, current_heading: float) -> float | None:
        """Update the current heading and compute the next command."""
        if self._complete:
            return None

        self._current_heading = current_heading

        # If we’ve progressed enough, update the step.
        if self._has_reached_current_step():
            self._current_command = self._calculate_next_command()

        # Check if we've reached the overall target.
        self._complete = self._reached_target()
        return self._current_command

    @abstractmethod
    def is_stuck(self, current_heading: float) -> bool:  # pragma: no cover
        """Check if the maneuver is stuck."""
        pass

    def _has_reached_current_step(self) -> bool:
        """Return True if progress since the last step is sufficient."""
        # Determine the step (positive for clockwise, negative for counter-clockwise).
        step = self._heading_step_size if self._clockwise else -self._heading_step_size

        # Compute progress from the last commanded step.
        if self._clockwise:
            progress = (self._current_heading - self._last_step_heading) % 360
        else:
            progress = (self._last_step_heading - self._current_heading) % 360

        # We consider the step reached if progress meets the required step minus tolerance.
        return progress >= abs(step) - self._heading_tolerance

    def _calculate_next_command(self) -> float:
        """Calculate the next commanded heading based on the current step."""
        step = self._heading_step_size if self._clockwise else -self._heading_step_size
        next_base_heading = (self._current_step_heading + step) % 360

        # Check if moving to next_base_heading would overshoot the target.
        if self._would_overshoot_target(next_base_heading):
            # Instead of overshooting, update state to target.
            self._last_step_heading = self._current_step_heading
            self._current_step_heading = self._target_heading
            return self._target_heading

        # Update our internal step state.
        self._last_step_heading = self._current_step_heading
        self._current_step_heading = next_base_heading
        return self._amplify_command(next_base_heading)

    def _would_overshoot_target(self, next_heading: float) -> bool:
        """Check if the next heading would overshoot target"""
        diff = (self._target_heading - self._current_step_heading) % 360
        if diff <= self._heading_tolerance:
            return True

        if self._clockwise:
            angle_turned = (next_heading - self._current_heading) % 360
            angle_needed = (self._target_heading - self._current_heading) % 360
        else:
            angle_turned = (self._current_heading - next_heading) % 360
            angle_needed = (self._current_heading - self._target_heading) % 360
        # Use >= to also capture the case where we exactly reach the target.
        return angle_turned >= angle_needed

    def _amplify_command(self, base_heading: float) -> float:
        """Amplify the command by the multiplier to help avoid undershooting."""
        amplification = self._heading_step_size * (self._command_multiplier - 1)
        if self._clockwise:
            return (base_heading + amplification) % 360
        return (base_heading - amplification) % 360

    def _get_current_command(self) -> float:
        """Return the active command; if at target, simply return target."""
        if self._current_step_heading == self._target_heading:
            return self._target_heading
        return self._amplify_command(self._current_step_heading)

    def _reached_target(self) -> bool:
        """Return True if the current heading is within tolerance of the target."""
        diff1 = (self._target_heading - self._current_heading) % 360
        diff2 = (self._current_heading - self._target_heading) % 360
        diff = min(diff1, diff2)

        if diff <= self._heading_tolerance:
            return True

        if self._current_step_heading == self._target_heading:
            # depending on the direction we are turning, we might have already overshot the target
            if self._clockwise:
                dist = (self._current_heading - self._current_step_heading) % 360
            else:
                dist = (self._current_step_heading - self._current_heading) % 360

            # this handles the case where we have a jump over 0
            dist = dist = dist - 360 if dist > 180 else dist
            # if dist is positive, we already overshot the target
            if dist > 0:
                return True

        return False

    @abstractmethod
    def _turn_clockwise(self) -> bool:  # pragma: no cover
        """Implement logic to decide if turning clockwise (True) or counter-clockwise (False)."""
        pass

    @property
    def current_command(self) -> float:
        """Return the current command to be sent to the sailboat."""
        return self._current_command

    @property
    def target_heading(self) -> float:
        """Return the target heading."""
        return self._target_heading

    @property
    def current_heading(self) -> float:
        """Return the current heading."""
        return self._current_heading

    @property
    def heading_step_size(self) -> float:
        """Return the heading step size."""
        return self._heading_step_size

    @property
    def command_step_size_multiplier(self) -> float:
        """Return the command step size multiplier."""
        return self._command_multiplier

    @property
    def heading_tolerance(self) -> float:
        """Return the heading tolerance."""
        return self._heading_tolerance

    @property
    def last_step_heading(self) -> float:
        """Return the heading at the last step."""
        return self._last_step_heading

    @property
    def current_step_heading(self) -> float:
        """Return the heading at the current step."""
        return self._current_step_heading

    @property
    def clockwise(self) -> bool:
        """Return True if the sailboat should turn clockwise."""
        return self._clockwise

    @property
    def complete(self) -> bool:
        """Return True if the maneuver is complete."""
        return self._complete


class TackManeuver(ManeuverAction):
    """Class for defining a tack maneuver for the sailboat."""

    def __init__(
        self,
        target_heading: float,
        current_heading: float,
        heading_step_size: float,
        wind_direction: float,
        command_step_size_multiplier: float,
        heading_tolerance: float,
        max_steps_to_cross_wind: int = 40,
    ) -> None:
        """Initialize the tack maneuver with the required parameters

        Args:
            target_heading: The target heading to reach.
            current_heading: The current heading of the sailboat.
            heading_step_size: The step size to take when turning the sailboat.
            wind_direction: The direction of the wind.
            command_step_size_multiplier: The multiplier to amplify the command step size.
            heading_tolerance: The tolerance for reaching the target heading.
            max_steps_to_cross_wind: The maximum number of steps to cross the wind.
        """
        super().__init__(
            target_heading,
            current_heading,
            heading_step_size,
            wind_direction,
            command_step_size_multiplier,
            heading_tolerance,
        )
        self._max_steps_to_cross_wind = max_steps_to_cross_wind
        self._steps_taken = 0
        # Remember which side of the wind we started on
        self._started_on_port = self._is_on_port_tack(current_heading)

    def step(self, current_heading: float) -> float | None:
        self._steps_taken += 1
        return super().step(current_heading)

    def is_stuck(self, current_heading: float) -> bool:
        """Check if the maneuver is stuck"""
        # Check if we're still on the same tack we started on
        still_on_same_tack = self._is_on_port_tack(current_heading) == self._started_on_port

        # We're stuck if we haven't crossed the wind after max_steps
        return still_on_same_tack and self._steps_taken >= self._max_steps_to_cross_wind

    def _is_on_port_tack(self, heading: float) -> bool:
        """Returns True if wind is coming from port side"""
        wind_angle = (self._wind_direction - heading) % 360
        return wind_angle > 180

    def _turn_clockwise(self) -> bool:
        diff = (self._target_heading - self._current_heading) % 360
        diff = diff - 360 if diff > 180 else diff
        return diff > 0


class JibeManeuver(ManeuverAction):
    """Class for defining a jibe maneuver for the sailboat."""

    def is_stuck(self, current_heading):
        return False

    def _turn_clockwise(self):
        current_wind_angle = (self._current_heading - self._wind_direction) % 360
        return current_wind_angle < 180
