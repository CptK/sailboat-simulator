"""What a pilot is shown each decision, and what it answers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sailing import NO_GO_DEG, STALL_WARNING_DEG, compass_name, point_of_sail


@dataclass
class Observation:
    t: float                      # simulated seconds since the start
    index: int                    # decision number, from 1
    image: np.ndarray             # RGB, with the bearing scale drawn on it
    jpeg: bytes                   # the image as sent
    heading: float                # compass degrees
    speed: float                  # m/s
    wind_from_rel: float          # where the wind comes from, degrees off the bow (+ = starboard)
    wind_speed: float
    colliding: bool
    east: float                   # not shown to the pilot (it has no map); for logs
    north: float
    instruction: str
    history: list["Decision"] = field(default_factory=list)
    chart: np.ndarray | None = None   # top-down chart (--chart)
    chart_jpeg: bytes | None = None
    chart_width_m: float = 0.0        # metres across the chart
    wind_from: float = 0.0            # where the wind comes from, compass degrees
    arrow_length_m: float = 0.0       # how long the chart's boat arrow is drawn, in metres
    true_land_ahead: float | None = None   # map distance to land along the bow (logs only)

    @property
    def in_no_go(self) -> bool:
        """Whether the bow is so close to the wind that the boat stalls."""
        return abs(self.wind_from_rel) < STALL_WARNING_DEG

    @property
    def close_hauled(self) -> tuple[float, float]:
        """The two sailable headings closest to the wind: (port tack, starboard tack)."""
        return ((self.wind_from + NO_GO_DEG) % 360.0, (self.wind_from - NO_GO_DEG) % 360.0)

    def instruments_text(self, period_note: str) -> str:
        side = "starboard" if self.wind_from_rel > 0 else "port"
        relative = ("dead ahead" if abs(self.wind_from_rel) < 2 else
                    "dead astern" if abs(self.wind_from_rel) > 178 else
                    f"{abs(self.wind_from_rel):.0f}° to {side} of the bow")
        # The zone runs clockwise from wind - NO_GO to wind + NO_GO.
        zone_from = (self.wind_from - NO_GO_DEG) % 360.0
        zone_to = (self.wind_from + NO_GO_DEG) % 360.0
        lines = [
            f"Task: {self.instruction}",
            f"Time: {self.t:.0f} s since start, decision {self.index}. {period_note}",
            "Instruments:",
            f"- compass heading: {self.heading:.0f}°",
            f"- speed: {self.speed:.2f} m/s",
            f"- wind: {self.wind_speed:.1f} m/s from {self.wind_from:.0f}° "
            f"({compass_name(self.wind_from)}), that is {relative}",
            f"- point of sail: {point_of_sail(self.wind_from_rel)}, "
            f"{abs(self.wind_from_rel):.0f}° off the wind",
            f"- no-go zone: the autopilot does not steer headings from {zone_from:.0f}° "
            f"clockwise to {zone_to:.0f}°; to go upwind it sails close-hauled on {zone_from:.0f}° "
            f"or {zone_to:.0f}°. Settling a few degrees inside that range while close-hauled "
            "is normal.",
            f"- touching land: {'YES' if self.colliding else 'no'}",
        ]
        note = next((d.progress for d in reversed(self.history) if d.progress), "")
        if note:
            lines.append(f"Your progress notes from your last decision: {note}")
        if self.history:
            lines.append("Your recent decisions (oldest first):")
            for d in self.history:
                steered = (f" (asked for {d.new_heading:.0f}°, inside the no-go zone: "
                           f"the autopilot steered {d.steered_heading:.0f}° instead)"
                           if d.steered_heading is not None else "")
                if d.maneuver:
                    steered += f" ({d.maneuver})"
                land = f", land ahead {d.land_ahead}" if d.land_ahead else ""
                lines.append(f"- t={d.t:.0f} s, heading {d.heading_before:.0f}°{land}: "
                             f"turn {d.turn_deg:+.0f}°{steered}, noted: {d.observation!r}")
        return "\n".join(lines)


@dataclass
class Decision:
    t: float = 0.0                # when the observation was taken
    heading_before: float = 0.0
    turn_deg: float = 0.0         # + = starboard
    observation: str = ""
    done: bool = False
    latency: float = 0.0          # wall seconds the pilot took
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    error: str = ""
    progress: str = ""            # the pilot's running task notes (--memory)
    land_ahead: str = ""          # the pilot's estimate of land straight ahead
    maneuver: str = ""            # what the autopilot did with it: a tack, and how it went
    steered_heading: float | None = None   # set when the no-go guard changed the course

    @property
    def new_heading(self) -> float:
        """The heading the pilot asked for."""
        return (self.heading_before + self.turn_deg) % 360.0

    @property
    def applied_heading(self) -> float:
        """The heading the autopilot actually steered."""
        return self.new_heading if self.steered_heading is None else self.steered_heading
