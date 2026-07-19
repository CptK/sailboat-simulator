"""The boat's sailing model: wind direction, polar diagram, leg costs.

The model is an object rather than module constants so that a process can plan
for more than one wind at a time, and so tests can vary the wind without
patching globals. Nothing here runs at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Coordinates are (east, north) metres, matching `map_loader`, so north is
# (0, +1) and a bearing of 0 degrees points along +y.
Vector = tuple[float, float]
Point = tuple[float, float]


@dataclass
class SailingModel:
    """Boat performance as a function of heading relative to the wind.

    Args:
        wind_from: Unit vector pointing towards where the wind originates,
            i.e. the direction a boat would head to be dead upwind. Defaults
            to a northerly (wind blowing from the top of the map downward).
        no_go_deg: Half-angle of the no-go zone in degrees. Headings closer to
            the wind than this cannot be sailed directly and must be tacked.
    """

    wind_from: Vector = (0.0, 1.0)
    no_go_deg: float = 45.0

    # Derived in __post_init__; not constructor arguments.
    optimal_tack_angle: float = field(init=False)
    optimal_vmg: float = field(init=False)

    def __post_init__(self) -> None:
        norm = math.hypot(*self.wind_from)
        if norm < 1e-9:
            raise ValueError("wind_from must be a non-zero vector")
        self.wind_from = (self.wind_from[0] / norm, self.wind_from[1] / norm)
        if not 0.0 < self.no_go_deg < 90.0:
            raise ValueError(f"no_go_deg must be in (0, 90), got {self.no_go_deg}")
        self.optimal_tack_angle, self.optimal_vmg = self._best_upwind_angle()

    @classmethod
    def from_bearing(cls, wind_bearing_deg: float, no_go_deg: float = 45.0) -> SailingModel:
        """Build a model from the compass bearing the wind blows *from*.

        Args:
            wind_bearing_deg: Bearing in degrees, 0 = north, clockwise. This is
                the convention used elsewhere in the sailboat stack.
            no_go_deg: Half-angle of the no-go zone in degrees.

        Returns:
            A model whose ``wind_from`` matches that bearing.
        """
        r = math.radians(wind_bearing_deg)
        return cls(wind_from=(math.sin(r), math.cos(r)), no_go_deg=no_go_deg)

    @property
    def no_go_rad(self) -> float:
        """The no-go half-angle in radians."""
        return math.radians(self.no_go_deg)

    @property
    def wind_bearing_deg(self) -> float:
        """The compass bearing the wind blows from: 0 = north, clockwise.

        Always in [0, 360). A northerly built from a bearing of exactly 360
        leaves atan2 a hair below zero, and the modulo would wrap that to
        359.999..., which rounds to 360.0 — reporting north as "360°".
        """
        bearing = math.degrees(math.atan2(self.wind_from[0], self.wind_from[1])) % 360.0
        return 0.0 if bearing > 360.0 - 1e-9 else bearing

    @property
    def blowing_towards(self) -> Vector:
        """Unit vector in the direction the wind blows, i.e. downwind."""
        return (-self.wind_from[0], -self.wind_from[1])

    def polar_speed(self, alpha: float) -> float:
        """Boat speed for a heading ``alpha`` radians off the wind.

        The shape is a typical sailboat's: nothing in the no-go zone, ramping
        up close-hauled, fastest on a reach, slightly slower running.

        Args:
            alpha: Angle between heading and the wind-from direction, in
                radians. 0 is dead upwind, pi is dead downwind.

        Returns:
            Speed in knots. Only ratios matter for planning.
        """
        deg = math.degrees(alpha)
        if deg < self.no_go_deg:
            return 0.0
        if deg <= 90.0:
            t = (deg - self.no_go_deg) / (90.0 - self.no_go_deg)
            return 5.0 + 3.0 * t          # 5 -> 8 knots, close-hauled
        if deg <= 135.0:
            t = (deg - 90.0) / 45.0
            return 8.0 + 2.0 * t          # 8 -> 10 knots, reaching
        t = (deg - 135.0) / 45.0
        return 10.0 - 2.0 * t             # 10 -> 8 knots, running

    def angle_off_wind(self, u: Point, v: Point) -> float:
        """Angle between the heading u->v and the wind-from direction.

        Args:
            u: Leg start, ``(east, north)`` in metres.
            v: Leg end, ``(east, north)`` in metres.

        Returns:
            Angle in radians in [0, pi]; 0 for a zero-length leg.
        """
        dx, dy = v[0] - u[0], v[1] - u[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return 0.0
        dot = (dx / dist) * self.wind_from[0] + (dy / dist) * self.wind_from[1]
        return math.acos(max(-1.0, min(1.0, dot)))

    def heading_is_sailable(self, u: Point, v: Point) -> bool:
        """Whether the direct heading u->v lies outside the no-go zone.

        Args:
            u: Leg start, ``(east, north)`` in metres.
            v: Leg end, ``(east, north)`` in metres.

        Returns:
            True if the leg can be sailed without tacking.
        """
        dx, dy = v[0] - u[0], v[1] - u[1]
        if math.hypot(dx, dy) < 1e-9:
            return True
        return self.angle_off_wind(u, v) >= self.no_go_rad

    def sailing_time(self, u: Point, v: Point) -> float:
        """Time to sail from u to v, tacking if the heading is in the no-go zone.

        Args:
            u: Leg start, ``(east, north)`` in metres.
            v: Leg end, ``(east, north)`` in metres.

        Returns:
            Travel time in distance/speed units.
        """
        dx, dy = v[0] - u[0], v[1] - u[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return 0.0

        alpha = self.angle_off_wind(u, v)
        speed = self.polar_speed(alpha)
        if speed > 0.0:
            return dist / speed

        # No-go zone: the boat must tack. Decomposing the tack legs so their
        # lateral components cancel gives an effective speed towards v.
        return dist / (self.optimal_vmg / math.cos(alpha))

    def _best_upwind_angle(self) -> tuple[float, float]:
        """Find the close-hauled angle maximising velocity made good upwind.

        Returns:
            ``(angle_rad, vmg)`` for the best upwind angle.
        """
        best_vmg, best_a = 0.0, self.no_go_rad
        for a_deg in range(int(self.no_go_deg), 90):
            a = math.radians(a_deg)
            vmg = self.polar_speed(a) * math.cos(a)
            if vmg > best_vmg:
                best_vmg, best_a = vmg, a
        return best_a, best_vmg
