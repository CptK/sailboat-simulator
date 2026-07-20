"""Maneuvers: which way the boat turns, and when a tack has failed.

A maneuver used to ramp the commanded heading towards the target in small
increments. That was removed: the ramp capped the heading error reaching the
rudder controller at roughly 17 degrees, which is far too gentle to carry a bow
through the eye of the wind — the boat lost way and stalled head to wind.

What is left is the part a plain heading command cannot express:

  * A **tack** turns the short way. That is all a tack is, so it simply commands
    the target and lets the controller put the rudder over.
  * A **jibe** must turn *away* from the wind, which in roughly half of cases is
    the long way round. A single heading command always takes the short arc, so
    a jibe steers at dead downwind first and only then at its target.
  * A tack that never comes through the wind gives up and becomes a jibe. That
    fallback lives here because only the maneuver knows which side it started on.
"""

from abc import ABC
from abc import abstractmethod


def _signed_difference(to_angle: float, from_angle: float) -> float:
    """The shortest signed turn from one bearing to another, in (-180, 180]."""
    return (to_angle - from_angle + 180) % 360 - 180


def _arc_contains(start: float, end: float, bearing: float) -> bool:
    """Whether the shortest arc from `start` to `end` passes over `bearing`."""
    sweep = _signed_difference(end, start)
    offset = _signed_difference(bearing, start)
    if sweep >= 0:
        return 0.0 <= offset <= sweep
    return sweep <= offset <= 0.0


class ManeuverAction(ABC):
    """Base class for a maneuver: a turn the boat cannot simply be pointed at.

    Attributes:
        target_heading: The heading to end up on.
        current_heading: The boat's heading as last reported.
        heading_tolerance: How close counts as arrived, in degrees.
        complete: Whether the maneuver has finished.
        clockwise: Which way the turn goes.
        current_command: The heading currently being commanded.
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
        """Initialize the maneuver.

        Args:
            target_heading: The heading to end up on.
            current_heading: The boat's heading now.
            heading_step_size: Unused. Retained so the node's construction and
                its parameter file need not change; see the module docstring.
            wind_direction: The bearing the wind blows from.
            command_step_size_multiplier: Unused, as `heading_step_size`.
            heading_tolerance: How close counts as arrived, in degrees.
        """
        self._target_heading = target_heading
        self._current_heading = current_heading
        self._wind_direction = wind_direction
        self._heading_tolerance = heading_tolerance

        self._complete = False
        self._clockwise = self._turn_clockwise()
        self._current_command = self._command_for(current_heading)

    def step(self, current_heading: float) -> float | None:
        """Update with the boat's heading and return the heading to steer.

        Args:
            current_heading: The boat's heading now, in degrees.

        Returns:
            The heading to command, or None once the maneuver is complete.
        """
        if self._complete:
            return None

        self._current_heading = current_heading
        if abs(_signed_difference(self._target_heading, current_heading)) <= self._heading_tolerance:
            self._complete = True
            return None

        self._current_command = self._command_for(current_heading)
        return self._current_command

    def _command_for(self, current_heading: float) -> float:
        """The heading to steer right now. Defaults to the target."""
        return self._target_heading

    @abstractmethod
    def is_stuck(self, current_heading: float) -> bool:  # pragma: no cover
        """Whether the maneuver has failed and should be replaced."""

    @abstractmethod
    def _turn_clockwise(self) -> bool:  # pragma: no cover
        """Whether the turn goes clockwise."""

    @property
    def current_command(self) -> float:
        return self._current_command

    @property
    def target_heading(self) -> float:
        return self._target_heading

    @property
    def current_heading(self) -> float:
        return self._current_heading

    @property
    def heading_tolerance(self) -> float:
        return self._heading_tolerance

    @property
    def clockwise(self) -> bool:
        return self._clockwise

    @property
    def complete(self) -> bool:
        return self._complete


class TackManeuver(ManeuverAction):
    """Turning through the eye of the wind.

    A tack is the short way round, so the target is commanded directly and the
    rudder controller is given the whole error to work with. Anything gentler
    and the boat runs out of momentum before the bow crosses the wind.
    """

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
        """Initialize the tack.

        Args:
            max_steps_to_cross_wind: Ticks to allow before declaring the tack
                failed. One tick per steering step, so at a 0.2 s interval 40 is
                about eight seconds of refusing to come through the wind.
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
        self._started_on_port = self._is_on_port_tack(current_heading)

    def step(self, current_heading: float) -> float | None:
        self._steps_taken += 1
        return super().step(current_heading)

    def is_stuck(self, current_heading: float) -> bool:
        """Whether the boat has failed to come through the wind in time."""
        still_on_same_tack = self._is_on_port_tack(current_heading) == self._started_on_port
        return still_on_same_tack and self._steps_taken >= self._max_steps_to_cross_wind

    def _is_on_port_tack(self, heading: float) -> bool:
        """Whether the wind is coming over the port side."""
        return (self._wind_direction - heading) % 360 > 180

    def _turn_clockwise(self) -> bool:
        return _signed_difference(self._target_heading, self._current_heading) > 0


class JibeManeuver(ManeuverAction):
    """Turning the stern through the wind.

    The defining constraint is direction: a jibe goes *away* from the wind, and
    when the target lies across the eye that is the long way round. A single
    heading command always takes the short arc, so this steers at dead downwind
    until the stern is through, then at the target.
    """

    #: How close to dead downwind counts as through, in degrees.
    THROUGH_DOWNWIND = 25.0

    def _command_for(self, current_heading: float) -> float:
        downwind = (self._wind_direction + 180.0) % 360.0

        # Only detour via downwind when the short way would cross the wind —
        # otherwise the target is already on this side and steering at downwind
        # first would turn the boat past it.
        if not _arc_contains(current_heading, self._target_heading, self._wind_direction):
            return self._target_heading

        if abs(_signed_difference(downwind, current_heading)) <= self.THROUGH_DOWNWIND:
            return self._target_heading
        return downwind

    def is_stuck(self, current_heading: float) -> bool:
        """A jibe is the fallback; there is nothing further to fall back to."""
        return False

    def _turn_clockwise(self) -> bool:
        return (self._current_heading - self._wind_direction) % 360 < 180
