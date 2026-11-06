"""Sailing geometry: the no-go zone, points of sail, and when a turn is a tack."""

from __future__ import annotations

import math

#: The no-go zone: headings closer to the wind than this are not sailed. One
#: number everywhere — the instruments, the chart's wedge, the prompt and the
#: autopilot's guard, which steers close-hauled at exactly this angle. 47°, not
#: the planner's 50°: the sail controller (controller/sail_controller.py) eases
#: the sail from 20° to 35° at 50° off the wind, so a boat held at 50° sailed
#: with an eased sail at ~1.0 m/s instead of ~1.5 — and slowed onto a shore
#: (Opus run, 2026-11-06). The boat itself stalls below ~42°.
NO_GO_DEG = 47.0
#: The instruments warn that the bow is in the no-go zone only below this: the
#: boat's own close-hauled course settles about 47° off the wind (the
#: controller pulls a few degrees up from 50°) at full speed, which is fine,
#: and a warning there made the pilot tack straight back after every tack.
STALL_WARNING_DEG = 42.0
#: The autopilot's no-go guard replaces a requested heading inside the zone
#: with the nearest heading this far off the wind.
GUARD_DEG = NO_GO_DEG


def point_of_sail(off_wind: float) -> str:
    """The point of sail for a heading this many degrees off the wind."""
    off = abs(off_wind)
    if off < STALL_WARNING_DEG:
        return "too close to the wind: the boat stalls"
    if off < 60:
        return "close-hauled (as close to the wind as the boat sails well)"
    if off < 80:
        return "close reach"
    if off < 110:
        return "beam reach"
    if off < 150:
        return "broad reach"
    return "running (wind from behind)"


def compass_name(degrees: float) -> str:
    """The nearest of the eight compass points, e.g. 'south-west'."""
    names = ("north", "north-east", "east", "south-east", "south", "south-west", "west",
             "north-west")
    return names[round((degrees % 360.0) / 45.0) % 8]


def wrap180(angle: float) -> float:
    """An angle in degrees, wrapped to (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def crosses_wind(current: float, target: float, wind_from: float) -> bool:
    """Whether turning the short way from `current` to `target` passes through
    the wind, i.e. whether the turn is a tack."""
    sweep = wrap180(target - current)
    offset = wrap180(wind_from - current)
    return (0.0 < offset < sweep) if sweep > 0 else (sweep < offset < 0.0)


def guard_heading(requested: float, wind_from: float, current: float,
                  limit: float = GUARD_DEG) -> float | None:
    """The nearest heading at least `limit` off the wind, or None if `requested` is fine.

    A request straight into the wind goes to the side the boat is already on.
    """
    off = wrap180(requested - wind_from)
    if abs(off) >= limit:
        return None
    side = math.copysign(1.0, off) if off else math.copysign(1.0, wrap180(current - wind_from) or 1.0)
    return (wind_from + side * limit) % 360.0
