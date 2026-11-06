"""What the pilot is asked to do, and how success is judged."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scenario import Scenario


@dataclass
class Task:
    """What the pilot is told to do, and how success is judged.

    target: the buoy (east, north) to reach, or None for a free instruction
    with no automatic success check (the run then ends on "done" or time).
    """

    instruction: str
    target: tuple[float, float] | None = None
    radius: float = 5.0

    def distance(self, east: float, north: float) -> float | None:
        if self.target is None:
            return None
        return math.dist((east, north), self.target)

    def succeeded(self, east: float, north: float) -> bool:
        d = self.distance(east, north)
        return d is not None and d <= self.radius


def make_task(scenario: Scenario, target: str | None, instruction: str | None,
              radius: float) -> Task:
    """The task for --target (a colour or a buoy number) and --instruction."""
    if target is None:
        if not instruction:
            raise SystemExit("Give --target <colour|buoy number> or --instruction '...'.")
        return Task(instruction, None, radius)
    if target.isdigit():
        index = int(target) - 1
        if not 0 <= index < len(scenario.buoys):
            raise SystemExit(f"Buoy {target} does not exist; the scenario has "
                             f"{len(scenario.buoys)} buoys.")
        buoy = scenario.buoys[index]
    else:
        matches = [b for b in scenario.buoys if (b.color or "") == target.lower()]
        if not matches:
            colours = sorted({b.color or "unnamed" for b in scenario.buoys})
            raise SystemExit(f"No {target} buoy in '{scenario.name}' (colours: "
                             f"{', '.join(colours) or 'no buoys'}).")
        if len(matches) > 1:
            numbers = [str(scenario.buoys.index(b) + 1) for b in matches]
            raise SystemExit(f"'{scenario.name}' has {len(matches)} {target} buoys "
                             f"(numbers {', '.join(numbers)}); pick one by number.")
        buoy = matches[0]
    name = buoy.color or "marked"
    return Task(instruction or f"Sail to the {name} buoy.", (buoy.east, buoy.north), radius)
