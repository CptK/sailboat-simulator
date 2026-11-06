"""Stand-in pilots and API client for the tests."""

import time
import types

from sail_pilot.observation import Decision
from sail_pilot.sailing import wrap180


class FakePilot:
    """Answers with fixed decisions; records what it was shown."""

    def __init__(self, turn=0.0, done=False, error="", delay=0.0):
        self.turn, self.done, self.error, self.delay = turn, done, error, delay
        self.seen = []

    def decide(self, obs):
        self.seen.append(obs)
        if self.delay:
            time.sleep(self.delay)
        return Decision(t=obs.t, heading_before=obs.heading, turn_deg=self.turn,
                        done=self.done, error=self.error, observation="fake")


class Plan:
    """A pilot that steers a heading chosen by time; records when it was asked."""

    def __init__(self, heading_at):
        self.heading_at, self.asked = heading_at, []

    def decide(self, obs):
        self.asked.append(obs.t)
        want = self.heading_at(obs.t)
        return Decision(t=obs.t, heading_before=obs.heading,
                        turn_deg=wrap180(want - obs.heading))


def fake_client(text=None, stop_reason="end_turn", raises=None):
    """A stand-in for anthropic.Anthropic: returns `text`; records each request."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if raises:
            raise raises
        content = [] if text is None else [types.SimpleNamespace(type="text", text=text)]
        return types.SimpleNamespace(content=content, stop_reason=stop_reason,
                                     usage=types.SimpleNamespace(input_tokens=900,
                                                                 output_tokens=40))
    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    return client, calls
