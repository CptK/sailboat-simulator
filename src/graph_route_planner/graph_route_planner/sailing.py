"""The boat's sailing model: wind direction, polar diagram, leg costs.

The model is an object rather than module constants so that a process can plan
for more than one wind at a time, and so tests can vary the wind without
patching globals. Nothing here runs at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

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

    def polar_speeds(self, alpha):
        """Boat speed for headings ``alpha`` radians off the wind.

        The shape is a typical sailboat's: nothing in the no-go zone, ramping
        up close-hauled, fastest on a reach, slightly slower running.

        This is the array form and the only implementation; `polar_speed` is a
        wrapper over it. Planning evaluates the polar once per graph edge, and
        on a fine grid that is hundreds of thousands of edges, so the loop
        belongs in numpy rather than in Python.

        Args:
            alpha: Angles between heading and the wind-from direction, in
                radians. 0 is dead upwind, pi is dead downwind. Any shape.

        Returns:
            Speeds in knots, same shape as `alpha`. Only ratios matter.
        """
        deg = np.degrees(np.asarray(alpha, dtype=float))
        speed = np.zeros(deg.shape, dtype=float)

        close = (deg >= self.no_go_deg) & (deg <= 90.0)
        reach = (deg > 90.0) & (deg <= 135.0)
        run = deg > 135.0

        # Each branch is evaluated only where its mask holds, so the no-go zone
        # cannot divide by a zero span and the ramps stay in their own domain.
        span = 90.0 - self.no_go_deg
        np.divide(deg - self.no_go_deg, span, out=speed, where=close)
        speed = np.where(close, 5.0 + 3.0 * speed, speed)          # 5 -> 8, close-hauled
        speed = np.where(reach, 8.0 + 2.0 * (deg - 90.0) / 45.0, speed)   # 8 -> 10, reaching
        speed = np.where(run, 10.0 - 2.0 * (deg - 135.0) / 45.0, speed)   # 10 -> 8, running
        return speed

    def polar_speed(self, alpha: float) -> float:
        """Boat speed for a single heading ``alpha`` radians off the wind.

        Args:
            alpha: Angle off the wind-from direction, in radians.

        Returns:
            Speed in knots.
        """
        return float(self.polar_speeds(alpha))

    def _deltas(self, u, v):
        """Leg vectors and lengths for arrays of endpoints.

        Args:
            u: Leg starts, shape (..., 2).
            v: Leg ends, shape (..., 2).

        Returns:
            ``(dx, dy, dist)``, each with the leading shape of the inputs.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        dx = v[..., 0] - u[..., 0]
        dy = v[..., 1] - u[..., 1]
        return dx, dy, np.hypot(dx, dy)

    def angles_off_wind(self, u, v):
        """Angles between the headings u->v and the wind-from direction.

        Args:
            u: Leg starts, shape (..., 2).
            v: Leg ends, shape (..., 2).

        Returns:
            Angles in radians in [0, pi]; 0 for zero-length legs.
        """
        dx, dy, dist = self._deltas(u, v)
        moving = dist >= 1e-9
        scale = np.where(moving, dist, 1.0)
        dot = (dx / scale) * self.wind_from[0] + (dy / scale) * self.wind_from[1]
        # A zero-length leg has no heading; clipping sends it to acos(1) = 0,
        # matching the degenerate case the scalar form special-cased.
        return np.arccos(np.clip(np.where(moving, dot, 1.0), -1.0, 1.0))

    def angle_off_wind(self, u: Point, v: Point) -> float:
        """Angle between the heading u->v and the wind-from direction.

        Args:
            u: Leg start, ``(east, north)`` in metres.
            v: Leg end, ``(east, north)`` in metres.

        Returns:
            Angle in radians in [0, pi]; 0 for a zero-length leg.
        """
        return float(self.angles_off_wind(u, v))

    def headings_are_sailable(self, u, v):
        """Which of the headings u->v lie outside the no-go zone.

        Args:
            u: Leg starts, shape (..., 2).
            v: Leg ends, shape (..., 2).

        Returns:
            Boolean array. Zero-length legs are sailable: staying put needs no
            heading at all.
        """
        _, _, dist = self._deltas(u, v)
        return (self.angles_off_wind(u, v) >= self.no_go_rad) | (dist < 1e-9)

    def heading_is_sailable(self, u: Point, v: Point) -> bool:
        """Whether the direct heading u->v lies outside the no-go zone.

        Args:
            u: Leg start, ``(east, north)`` in metres.
            v: Leg end, ``(east, north)`` in metres.

        Returns:
            True if the leg can be sailed without tacking.
        """
        return bool(self.headings_are_sailable(u, v))

    def sailing_times(self, u, v):
        """Times to sail u->v, tacking where the heading is in the no-go zone.

        Args:
            u: Leg starts, shape (..., 2).
            v: Leg ends, shape (..., 2).

        Returns:
            Travel times in distance/speed units, 0 for zero-length legs.
        """
        _, _, dist = self._deltas(u, v)
        alpha = self.angles_off_wind(u, v)
        speed = self.polar_speeds(alpha)

        direct = speed > 0.0
        time = np.zeros(np.shape(dist), dtype=float)
        np.divide(dist, speed, out=time, where=direct)

        # No-go zone: the boat must tack. Decomposing the tack legs so their
        # lateral components cancel gives an effective speed towards v, which is
        # optimal_vmg / cos(alpha) — so the time is dist * cos(alpha) / vmg.
        if self.optimal_vmg > 0.0:
            tacked = ~direct
            time = np.where(tacked, dist * np.cos(alpha) / self.optimal_vmg, time)

        return np.where(dist < 1e-9, 0.0, time)

    def sailing_time(self, u: Point, v: Point) -> float:
        """Time to sail from u to v, tacking if the heading is in the no-go zone.

        Args:
            u: Leg start, ``(east, north)`` in metres.
            v: Leg end, ``(east, north)`` in metres.

        Returns:
            Travel time in distance/speed units.
        """
        return float(self.sailing_times(u, v))

    def _best_upwind_angle(self) -> tuple[float, float]:
        """Find the close-hauled angle maximising velocity made good upwind.

        Returns:
            ``(angle_rad, vmg)`` for the best upwind angle.
        """
        angles = np.radians(np.arange(int(self.no_go_deg), 90, dtype=float))
        if angles.size == 0:
            return self.no_go_rad, 0.0

        vmg = self.polar_speeds(angles) * np.cos(angles)
        best = int(np.argmax(vmg))          # first maximum wins, as the loop did
        if vmg[best] <= 0.0:
            return self.no_go_rad, 0.0
        return float(angles[best]), float(vmg[best])
