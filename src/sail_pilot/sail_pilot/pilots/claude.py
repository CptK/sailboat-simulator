"""The Claude pilot: one Messages API request per decision."""

from __future__ import annotations

import base64
import json
import time

from ..observation import Decision, Observation
from ..render import COURSE_LINE_M, COURSE_TICK_M
from ..sailing import GUARD_DEG, NO_GO_DEG

#: The answer's estimate of land straight ahead, nearest first. "none": no land
#: anywhere ahead within sight.
LAND_AHEAD = ("under 5 m", "5-10 m", "10-20 m", "20-50 m", "over 50 m", "none")

#: The boat's length, for telling the model how large the chart's arrow is.
BOAT_LENGTH_M = 2.7

#: USD per million tokens (input, output), for the cost estimate.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00),
          "claude-opus-5-5": (4.00, 20.00), "claude-opus-5": (5.00, 25.00),
          "claude-sonnet-5": (2.00, 10.00), "claude-fable-5-1": (10.00, 50.00),
          "claude-fable-5": (10.00, 50.00), "claude-opus-4-8": (5.00, 25.00),
          "claude-haiku-4-5": (1.00, 5.00)}

#: How --no-thinking turns thinking off, per model: Sonnet 5.5 takes
#: "between_tools" instead of "disabled"; on Opus 5.5 and Fable thinking cannot
#: be turned off at all (lower --effort instead). Both only at effort high or
#: below. Models not listed get "disabled" and the API's verdict.
THINKING_OFF = {"claude-haiku-5-5": "disabled", "claude-sonnet-5-5": "between_tools",
                "claude-opus-5": "disabled", "claude-sonnet-5": "disabled",
                "claude-opus-5-5": None, "claude-fable-5-1": None, "claude-fable-5": None}

SYSTEM_PROMPT = """\
You are the helmsman of a small autonomous sailboat (2.7 m long). You steer by \
choosing course changes; an autopilot holds the course you set until your next \
decision.

Each decision you get the view from a camera on the mast, looking forward over \
the bow, and the boat's instruments.

The image:
- The scale along the top shows bearings in degrees off the bow: negative is to \
port (left), positive to starboard (right). Short white marks below the scale \
and at the bottom edge show where the bow points. The camera covers about \
{half_fov:.0f} degrees either side of the bow; \
anything else is out of view. To look elsewhere, turn.
- The brown shape at the bottom centre of every image is the bow of your own \
boat. It is never a buoy.
- The marks on the left and right edges of the image show distance over the \
water: where the water surface appears at the height of the "10 m" mark, it \
is 10 m away. The closer to the bottom of the image, the closer; everything \
far away is squeezed just below the horizon. Where land, or a buoy, meets the \
water tells you how far away it is. The marks move with the boat's motion. \
Below land there is often a light blue band: that is the water reflecting the \
land, not land; the land ends where the green ends.
- Buoys are coloured floats with a flag, standing in the water. Size tells \
distance: a buoy 10 m away is a large shape, one 50 m away is only a few pixels \
tall near the horizon. Water is blue (the checker pattern is only the water's \
texture). Land is green and solid.
- The islands are low: you can see buoys and water beyond them across the land, \
so a buoy you see may be on the far side of an island. Never steer a course that \
crosses green land, in the camera image or on the chart: before heading for \
something, check that there is open water all the way to it, and if land is in \
between, go around it, keeping at least 10 m clear of the shore.
- Running aground is the worst outcome: the boat stops and usually cannot get \
free again. When land is close ahead, turn away early.
- If the target is not in view, it is somewhere outside the camera's view: turn \
by about 60-90 degrees at a time to look around, and use your previous notes to \
remember where you have already looked. Never steer towards buoys of another \
colour instead of the target.

Sailing:
- The boat cannot sail closer than {no_go:.0f} degrees to the direction the wind \
comes from: the no-go zone. The instruments give the zone as a range of \
headings, and the two closest sailable headings on either side of it.
- If where you want to go lies inside the no-go zone, sail upwind by tacking: \
hold one of the two closest sailable headings, and switch to the other one \
before you get close to land (check the chart), then back again. Each \
switch is a tack. Asking for the heading of the target itself does not work.
- Every other direction is sailable. The boat needs a few seconds and some room \
to turn; it moves about 1-2 metres per second.{guard_note}

Answer with:
- observation: one short sentence on what you see that matters for the task \
(where the target is relative to the bow, obstacles).
- land_ahead: how far the nearest land is in the direction the bow points \
(within about 10 degrees either side of it). Judge it from where the bottom \
edge of the land meets the water against the distance marks, and from the \
chart. Decide this before turn_deg: if it is under 20 m, turn away from the \
land now, towards open water.
- turn_deg: the course change in degrees, relative to the current heading: \
positive turns to starboard (right), negative to port (left), 0 holds the \
course. Between -180 and 180.
- done: true only when the whole task is complete: its last step has actually \
happened, not just about to. For example, "end between two buoys" is done \
once the boat has sailed between them, not while it is approaching them."""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "land_ahead": {"type": "string", "enum": list(LAND_AHEAD)},
        "turn_deg": {"type": "number"},
        "done": {"type": "boolean"},
    },
    "required": ["observation", "land_ahead", "turn_deg", "done"],
    "additionalProperties": False,
}


GUARD_NOTE = """
- The autopilot never steers into the no-go zone: if you ask for a heading \
inside it, it steers the nearest closest-sailable heading instead, and your \
decision history says so. It never decides to tack for you: the other side is \
yours to choose.
- When your new heading means turning through the wind, the autopilot carries \
out the tack for you: it first bears away to pick up speed if the boat is \
slow, then tacks, and falls back to turning the other way round (a jibe) if \
the tack fails. You are not asked again until it is complete, and your \
decision history says how it went. So decide when to tack, and then let each \
tack run: switching back and forth only stops the boat."""

CHART_PROMPT = """

The chart (second image): a map seen from above, north up. Blue is water, \
green is land, the coloured dots are the buoys, the red arrow is your boat \
pointing along its heading, and the white line is the track you have sailed \
so far. The arrow is drawn much larger than the boat: it is about \
{arrow:.0f} m long, the boat only {boat:.1f} m; the boat's exact position is \
the small white dot near the back of the arrow, so judge distances to land \
from that dot. The yellow line from the boat is your current course: where \
the boat goes in the next {course:.0f} m if it holds this heading, with a \
tick every {tick:.0f} m. If it crosses land, you will hit that land: change \
course before it does. When you pick a new heading, check the same way that \
open water lies along it. The scale bar in the bottom-right corner shows a length in \
metres. The light wedge from your boat points into the wind: it is the no-go \
zone, so every course inside it cannot be sailed; its edges are the two \
closest sailable headings. The arrow labelled "wind" in the top-left corner \
shows where the wind blows to. It is {width:.0f} metres across. Use it to know where you are and \
where things are that the camera cannot see; use the camera for what is close \
ahead."""

MEMORY_PROMPT = """

- progress: your running notes on the task, for yourself: which parts of the \
task are done, what you are doing now, what comes next, and anything you \
noticed that you will need later. You see your latest note at your next \
decision; beyond your last few decisions it is your only memory, so keep it \
complete each time rather than writing only what changed."""

MEMORY_SCHEMA = {
    **DECISION_SCHEMA,
    "properties": {**DECISION_SCHEMA["properties"], "progress": {"type": "string"}},
    "required": [*DECISION_SCHEMA["required"], "progress"],
}


class ClaudePilot:
    """Asks a Claude model for each decision (one independent request each)."""

    name = "claude"

    def __init__(self, model: str = "claude-haiku-5-5", effort: str | None = "low",
                 thinking: bool = True, period_note: str = "", client=None,
                 timeout: float = 30.0, half_fov: float = 43.0, memory: bool = False,
                 guard_deg: float | None = GUARD_DEG):
        import anthropic

        self.guard_deg = guard_deg  # the harness's no-go guard, described in the prompt
        self.memory = memory        # ask for, and show back, a running progress note
        self.half_fov = half_fov    # set from the camera by main(); see PilotRun.half_fov
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.period_note = period_note
        self.client = client or anthropic.Anthropic(timeout=timeout, max_retries=2)

    def settings(self) -> dict:
        return {"pilot": self.name, "model": self.model, "effort": self.effort,
                "thinking": self.thinking, "memory": self.memory}

    def system_prompt(self, obs: Observation) -> str:
        guard = GUARD_NOTE.format(guard=self.guard_deg) if self.guard_deg else ""
        text = SYSTEM_PROMPT.format(half_fov=self.half_fov, guard_note=guard, no_go=NO_GO_DEG)
        if obs.chart_jpeg is not None:
            text += CHART_PROMPT.format(width=obs.chart_width_m, arrow=obs.arrow_length_m,
                                        boat=BOAT_LENGTH_M, course=COURSE_LINE_M,
                                        tick=COURSE_TICK_M)
        return text + (MEMORY_PROMPT if self.memory else "")

    def content(self, obs: Observation) -> list:
        def image(data: bytes) -> dict:
            return {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(data).decode("ascii")}}
        if obs.chart_jpeg is None:
            return [image(obs.jpeg),
                    {"type": "text", "text": obs.instruments_text(self.period_note)}]
        return [{"type": "text", "text": "Camera, looking forward over the bow:"},
                image(obs.jpeg),
                {"type": "text", "text": "Chart, north up:"},
                image(obs.chart_jpeg),
                {"type": "text", "text": obs.instruments_text(self.period_note)}]

    def request(self, obs: Observation) -> dict:
        """The Messages API request for an observation (also used by the tests)."""
        schema = MEMORY_SCHEMA if self.memory else DECISION_SCHEMA
        output_config = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        request = dict(
            model=self.model,
            max_tokens=4096,
            system=self.system_prompt(obs),
            output_config=output_config,
            messages=[{"role": "user", "content": self.content(obs)}],
        )
        if not self.thinking:
            request["thinking"] = {"type": THINKING_OFF.get(self.model, "disabled")}
        return request

    def decide(self, obs: Observation) -> Decision:
        import anthropic

        decision = Decision(t=obs.t, heading_before=obs.heading)
        start = time.monotonic()
        try:
            response = self.client.messages.create(**self.request(obs))
        except anthropic.APIStatusError as err:
            decision.error = f"API error {err.status_code}: {err.message}"
        except anthropic.APIConnectionError as err:
            decision.error = f"connection error: {err}"
        except Exception as err:  # noqa: BLE001 - one bad answer must not end the run
            decision.error = f"{type(err).__name__}: {err}"
        else:
            decision.stop_reason = response.stop_reason or ""
            decision.input_tokens = response.usage.input_tokens
            decision.output_tokens = response.usage.output_tokens
            text = next((b.text for b in response.content if b.type == "text"), None)
            if response.stop_reason == "refusal":
                decision.error = "the model declined to answer"
            elif text is None:
                decision.error = f"no answer (stop reason {response.stop_reason})"
            else:
                try:
                    answer = json.loads(text)
                    decision.observation = str(answer["observation"])
                    decision.turn_deg = max(-180.0, min(180.0, float(answer["turn_deg"])))
                    decision.done = bool(answer["done"])
                    decision.progress = str(answer.get("progress", ""))
                    decision.land_ahead = str(answer.get("land_ahead", ""))
                except (ValueError, KeyError, TypeError) as err:
                    decision.error = f"unreadable answer ({err}): {text[:200]}"
        decision.latency = time.monotonic() - start
        return decision


def estimate_cost(model: str | None, tokens_in: int, tokens_out: int) -> float | None:
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    return (tokens_in * pin + tokens_out * pout) / 1e6
