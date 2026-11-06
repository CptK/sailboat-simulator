"""A privileged pilot that knows where the target is: a baseline that needs no API."""

from __future__ import annotations

import math
import time

from ..observation import Decision, Observation
from ..sailing import NO_GO_DEG, wrap180
from ..task import Task


class ScriptedPilot:
    """A pilot that cheats: it knows where the target is (no camera needed).

    Heads straight for the target, or as close to it as the wind allows. A
    reference for what the harness and controller can do, and a way to run
    everything without an API key. It does not avoid land.
    """

    name = "scripted"

    def __init__(self, task: Task, delay: float = 0.0):
        if task.target is None:
            raise SystemExit("The scripted pilot needs a --target buoy.")
        self.task = task
        self.delay = delay          # seconds to sleep, to stand in for API latency

    def settings(self) -> dict:
        return {"pilot": self.name, "delay": self.delay}

    def decide(self, obs: Observation) -> Decision:
        start = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        tx, ty = self.task.target
        bearing = math.degrees(math.atan2(tx - obs.east, ty - obs.north)) % 360.0
        wind_from = (obs.heading + obs.wind_from_rel) % 360.0
        off_wind = wrap180(bearing - wind_from)
        course = bearing
        if abs(off_wind) < NO_GO_DEG:
            course = (wind_from + math.copysign(NO_GO_DEG, off_wind or 1.0)) % 360.0
        dist = self.task.distance(obs.east, obs.north)
        return Decision(t=obs.t, heading_before=obs.heading,
                        turn_deg=wrap180(course - obs.heading),
                        observation=f"target bears {bearing:.0f}°, {dist:.0f} m away",
                        done=self.task.succeeded(obs.east, obs.north),
                        latency=time.monotonic() - start)
