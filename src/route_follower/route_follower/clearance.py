"""How far the boat can sail on a bearing before it gets too close to land.

The follower uses this to look ahead along the heading it is about to command
and, if that runs onto land, to pick the nearest heading that does not.

Distances are measured to land grown by a clearance (the hull's half-width plus
a buffer), not to the planner's shore margin. Planned legs run along the edge
of that margin by design, so checking against it would fight every route.

Pure geometry over shapely — no ROS — so it is cheap to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.prepared import prep
from shapely.strtree import STRtree


def _segments(geometry) -> list[LineString]:
    """Every edge of a (multi)polygon's rings, as two-point LineStrings."""
    segments = []
    polygons = getattr(geometry, "geoms", [geometry])
    for polygon in polygons:
        if polygon.is_empty:
            continue
        for ring in (polygon.exterior, *polygon.interiors):
            coords = np.asarray(ring.coords)
            segments.extend(LineString(pair) for pair in zip(coords[:-1], coords[1:]))
    return segments


class _Shore:
    """A shoreline indexed for fast ray casting."""

    def __init__(self, water):
        self.water = water
        self.prepared = prep(water) if not water.is_empty else None
        self.segments = _segments(water)
        self.tree = STRtree(self.segments) if self.segments else None

    def contains(self, point: Point) -> bool:
        return self.prepared is not None and self.prepared.contains(point)

    def first_hit(self, ray: LineString, origin: Point) -> float | None:
        """Distance from origin to the first shoreline crossing on ray, or None."""
        if self.tree is None:
            return None
        hits = self.tree.query(ray, predicate="intersects")
        if len(hits) == 0:
            return None
        crossings = shapely.intersection(ray, np.asarray(self.segments, dtype=object)[hits])
        return float(min(origin.distance(c) for c in crossings if not c.is_empty))


@dataclass(frozen=True)
class HeadingChoice:
    """The outcome of a clearance check.

    Attributes:
        heading: The heading to steer, compass degrees.
        overridden: True if it is not the heading that was asked for.
        free_distance: Clear water ahead on that heading, metres (capped at
            the look-ahead distance).
        kind: What the override does: "correct" (same tack), "change tack" or
            "open water" (nothing clear); empty if not overridden.
        reason: Why it was overridden, for the log; empty if it was not.
    """

    heading: float
    overridden: bool
    free_distance: float
    kind: str = ""
    reason: str = ""


class Clearance:
    """Clear-water distances on a map, keeping `clearance` metres from land.

    Args:
        water: The map's navigable water (shapely geometry, local metres).
        clearance: How close to land the boat's reference point may come.
    """

    def __init__(self, water, clearance: float):
        self.clearance = clearance
        self._raw = _Shore(water)
        self._safe = _Shore(water.buffer(-clearance)) if clearance > 0 else self._raw

    def distance_to_land(self, east: float, north: float) -> float:
        """Distance from a point to the shoreline; 0 if the point is on land."""
        point = Point(east, north)
        if not self._raw.contains(point):
            return 0.0
        return float(self._raw.water.boundary.distance(point))

    def free_distance(self, east: float, north: float, bearing: float,
                      max_distance: float) -> float:
        """Clear water straight ahead on `bearing`, up to `max_distance` metres.

        Measured to land grown by the clearance. A boat already inside that
        clearance (closer to land than it should be) is measured to land
        itself instead, so it can still tell a way out from a way in.

        Args:
            east, north: Start of the ray, local metres.
            bearing: Compass degrees, 0 = north, clockwise.
            max_distance: Look no further than this.
        """
        origin = Point(east, north)
        shore = self._safe if self._safe.contains(origin) else self._raw
        if not shore.contains(origin):
            return 0.0          # on land
        rad = math.radians(bearing)
        end = (east + max_distance * math.sin(rad), north + max_distance * math.cos(rad))
        ray = LineString([(east, north), end])
        hit = shore.first_hit(ray, origin)
        return max_distance if hit is None else min(hit, max_distance)


#: Largest turn away from the desired heading that still counts as a correction
#: worth staying on this tack for. Beyond it, bearing away further is worse than
#: tacking: in a channel, it means running back the way the boat came.
MAX_SAME_TACK_CORRECTION = 60.0


def _angle_diff(a: float, b: float) -> float:
    """Absolute angle between two bearings, degrees in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _on_port_tack(heading: float, wind_direction: float) -> bool:
    """Whether the wind comes over the port side on this heading."""
    return (wind_direction - heading) % 360.0 > 180.0


def choose_heading(clearance: Clearance, east: float, north: float, desired: float,
                   wind_direction: float, no_go_angle: float, lookahead: float,
                   current_heading: float, drift: float = 0.0,
                   search_step: float = 5.0) -> HeadingChoice:
    """The desired heading if its path is clear, else the best clear alternative.

    The path checked is the one the boat will actually make good: the heading
    plus `drift` (course over ground minus heading — leeway, mostly).

    Preference when the desired heading is blocked:
      1. the clear sailable heading nearest the desired one on the current tack,
         if it is within MAX_SAME_TACK_CORRECTION of it (a correction, no
         maneuver);
      2. otherwise the clear sailable heading nearest the desired one — on the
         other tack, that is a tack or jibe now, while there is still room and
         speed for it;
      3. if nothing is clear, the sailable heading with the most open water.

    Args:
        clearance: The map's clearance model.
        east, north: The boat's position.
        desired: The heading the follower wants, compass degrees.
        wind_direction: Bearing the wind blows from, degrees.
        no_go_angle: Closest the boat can sail to the wind, degrees.
        lookahead: How far ahead the path must be clear, metres.
        current_heading: The boat's heading now, to tell which tack it is on.
        drift: Course over ground minus heading, degrees.
        search_step: Spacing of the candidate headings, degrees.
    """
    def free(h: float) -> float:
        return clearance.free_distance(east, north, (h + drift) % 360.0, lookahead)

    ahead = free(desired)
    if ahead >= lookahead:
        return HeadingChoice(desired % 360.0, False, ahead)

    port = _on_port_tack(current_heading, wind_direction)
    candidates = []
    for h in np.arange(0.0, 360.0, search_step):
        h = float(h)
        if _angle_diff(h, wind_direction) < no_go_angle:
            continue
        candidates.append((h, free(h)))

    clear = [(h, d) for h, d in candidates if d >= lookahead]
    same_tack = [(h, d) for h, d in clear
                 if _on_port_tack(h, wind_direction) == port
                 and _angle_diff(h, desired) <= MAX_SAME_TACK_CORRECTION]
    blocked = f"land {ahead:.1f} m ahead on {desired:.0f}°"
    if same_tack:
        h, d = min(same_tack, key=lambda c: _angle_diff(c[0], desired))
        return HeadingChoice(h, True, d, "correct", f"{blocked}: correcting to {h:.0f}°")
    if clear:
        h, d = min(clear, key=lambda c: _angle_diff(c[0], desired))
        if _on_port_tack(h, wind_direction) == port:
            return HeadingChoice(h, True, d, "correct",
                                 f"{blocked}: bearing away to {h:.0f}°")
        return HeadingChoice(h, True, d, "change tack",
                             f"{blocked}, nothing clear nearby on this tack: "
                             f"changing tack to {h:.0f}°")
    if candidates:
        h, d = max(candidates, key=lambda c: (c[1], -_angle_diff(c[0], desired)))
        if d > ahead:
            return HeadingChoice(h, True, d, "open water",
                                 f"{blocked}, no clear heading: heading for the most open "
                                 f"water, {h:.0f}°")
    return HeadingChoice(desired % 360.0, False, ahead)


# ── turns ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TurnModel:
    """Where the boat goes during a tack or jibe, roughly.

    Fitted to 21 maneuvers sailed in the simulation (scripts/closed_loop.py):
    the heading barely moves for the first second or so — the rudder slews
    and the controller builds the turn — so the boat carries on along its old
    course, then comes round quickly. Its mean error against those tracks was
    about 1.4 m (worst 2.9 m), which is what `buffer` is for.

    Attributes:
        delay: Seconds the boat holds its old course after the turn is ordered.
        rate: Turn rate once turning, degrees per second.
        settle: Seconds of sailing on the new heading included after the turn.
        buffer: Extra distance from land the predicted path must keep, on top
            of the hull clearance, for the model's error.
        min_speed: Speed assumed for a boat moving slower, m/s.
    """

    delay: float = 1.5
    rate: float = 60.0
    settle: float = 2.0
    buffer: float = 1.5
    min_speed: float = 0.5

    def path(self, east: float, north: float, heading: float, turn: float,
             speed: float, dt: float = 0.2) -> list[tuple[float, float]]:
        """Predicted positions through a turn.

        Args:
            east, north: Where the turn is ordered.
            heading: Course over ground at that moment, compass degrees.
            turn: Signed turn in degrees, positive clockwise; a jibe the long
                way round is more than 180 in magnitude.
            speed: Boat speed, m/s.
            dt: Spacing of the predicted points in seconds.

        Returns:
            Positions from the start to `settle` seconds after the turn.
        """
        speed = max(speed, self.min_speed)
        duration = self.delay + abs(turn) / self.rate + self.settle
        points = [(east, north)]
        for i in range(1, math.ceil(duration / dt - 1e-9) + 1):
            t = i * dt
            swung = min(self.rate * max(0.0, t - self.delay), abs(turn))
            h = math.radians(heading + math.copysign(swung, turn))
            east += speed * dt * math.sin(h)
            north += speed * dt * math.cos(h)
            points.append((east, north))
        return points


def path_clearance(clearances: list[Clearance], points: list[tuple[float, float]]) -> bool:
    """Whether a path keeps clear of land, by the strictest standard it can.

    Tries each clearance in order, strictest first. A boat that already sits
    inside one (closer to land than it allows) cannot meet it, so the next is
    used instead — otherwise a boat near the shore could never turn at all.

    Args:
        clearances: Clearance models, strictest (largest clearance) first.
        points: The path.
    """
    start = Point(points[0])
    for clearance in clearances:
        if clearance._safe.contains(start):
            line = LineString(points) if len(points) > 1 else start
            return clearance._safe.prepared.covers(line)
    return False


def turn_angle(current: float, target: float, wind_direction: float, jibe: bool) -> float:
    """Signed turn (positive clockwise) a maneuver makes from one heading to another.

    A tack turns the short way. A jibe turns away from the wind, which is the
    long way round when the short way would cross it.
    """
    short = (target - current + 180.0) % 360.0 - 180.0
    if not jibe:
        return short
    crosses_wind = _angle_diff(current, wind_direction) + _angle_diff(wind_direction, target) \
        <= abs(short) + 1e-6
    if crosses_wind and abs(short) > 1e-6:
        return short - math.copysign(360.0, short)
    return short
