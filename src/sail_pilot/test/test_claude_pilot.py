"""The Claude pilot's requests and answers, with a fake client: no API calls."""

import base64
import json

import pytest

from sail_pilot.pilots.claude import DECISION_SCHEMA, LAND_AHEAD, ClaudePilot, estimate_cost
from sail_pilot.sailing import NO_GO_DEG

pytest.importorskip("rclpy", reason="needs the ROS environment (source install/setup.bash)")

from pilot_helpers import fake_client  # noqa: E402

ANSWER = json.dumps({"observation": "x", "land_ahead": "none", "turn_deg": 0, "done": False})


def test_claude_request_carries_image_instruments_and_schema(make_run):
    obs = make_run().observe(3, [])
    client, calls = fake_client(json.dumps({"observation": "x", "turn_deg": 10, "done": False}))
    pilot = ClaudePilot(model="claude-haiku-5-5", effort="low", thinking=False, client=client,
                        half_fov=41.6)
    pilot.decide(obs)
    req = calls[0]
    assert req["model"] == "claude-haiku-5-5"
    image, text = req["messages"][0]["content"]
    assert image["source"]["media_type"] == "image/jpeg"
    assert base64.standard_b64decode(image["source"]["data"]) == obs.jpeg
    assert "Task: Sail to the orange buoy." in text["text"]
    assert req["output_config"] == {"format": {"type": "json_schema",
                                               "schema": DECISION_SCHEMA}, "effort": "low"}
    assert req["thinking"] == {"type": "disabled"}
    assert "about 42 degrees" in req["system"]


def test_claude_answer_is_parsed_and_clamped(make_run):
    obs = make_run().observe(1, [])
    client, _ = fake_client(json.dumps({"observation": "buoy ahead", "turn_deg": 400,
                                        "done": True}))
    d = ClaudePilot(client=client).decide(obs)
    assert (d.turn_deg, d.done, d.observation, d.error) == (180.0, True, "buoy ahead", "")
    assert (d.input_tokens, d.output_tokens) == (900, 40)


@pytest.mark.parametrize("client_args, message", [
    (dict(text=None, stop_reason="refusal"), "declined"),
    (dict(text="not json"), "unreadable"),
    (dict(text=None, stop_reason="max_tokens"), "no answer"),
    (dict(raises=RuntimeError("socket closed")), "RuntimeError"),
])
def test_claude_failures_become_errors_not_crashes(make_run, client_args, message):
    client, _ = fake_client(**client_args)
    d = ClaudePilot(client=client).decide(make_run().observe(1, []))
    assert message in d.error and d.turn_deg == 0.0


def test_request_with_chart_has_both_images_labelled(make_run):
    obs = make_run(chart=True).observe(1, [])
    client, calls = fake_client(ANSWER)
    ClaudePilot(client=client).decide(obs)
    content = calls[0]["messages"][0]["content"]
    assert [c["type"] for c in content] == ["text", "image", "text", "image", "text"]
    assert "Camera" in content[0]["text"] and "Chart" in content[2]["text"]
    assert base64.standard_b64decode(content[3]["source"]["data"]) == obs.chart_jpeg
    assert "north up" in calls[0]["system"] and "metres across" in calls[0]["system"]


# --- memory ---------------------------------------------------------------------

def test_memory_asks_for_a_progress_note_and_shows_it_back(make_run):
    run = make_run()
    client, calls = fake_client(json.dumps({"observation": "x", "turn_deg": 0, "done": False,
                                            "progress": "passed red; heading for blue"}))
    pilot = ClaudePilot(client=client, memory=True)
    first = pilot.decide(run.observe(1, []))
    assert first.progress == "passed red; heading for blue"
    req = calls[0]
    assert "progress" in req["output_config"]["format"]["schema"]["required"]
    assert "running notes" in req["system"]
    nxt = run.observe(2, [first])
    assert "Your progress notes from your last decision: passed red; heading for blue" \
        in nxt.instruments_text("")


def test_without_memory_the_schema_is_unchanged(make_run):
    client, calls = fake_client(ANSWER)
    ClaudePilot(client=client).decide(make_run().observe(1, []))
    assert calls[0]["output_config"]["format"]["schema"] == DECISION_SCHEMA
    assert "running notes" not in calls[0]["system"]


# --- the prompt -------------------------------------------------------------------

def test_the_prompt_describes_the_guard_only_when_it_is_on(make_run):
    obs = make_run().observe(1, [])
    client, calls = fake_client(ANSWER)
    ClaudePilot(client=client).decide(obs)
    ClaudePilot(client=client, guard_deg=None).decide(obs)
    assert "never steers into the no-go zone" in calls[0]["system"]
    assert "never steers into the no-go zone" not in calls[1]["system"]


def test_one_no_go_angle_everywhere(make_run):
    obs = make_run(chart=True).observe(1, [])
    client, calls = fake_client(ANSWER)
    ClaudePilot(client=client).decide(obs)
    system = calls[0]["system"]
    assert f"closer than {NO_GO_DEG:.0f} degrees" in system
    assert "45 degrees" not in system
    assert "tacking" in system and "light wedge" in system


def test_land_ahead_is_asked_before_the_turn_and_parsed(make_run):
    props = list(DECISION_SCHEMA["properties"])
    assert props.index("land_ahead") < props.index("turn_deg")
    assert DECISION_SCHEMA["properties"]["land_ahead"]["enum"] == list(LAND_AHEAD)
    obs = make_run().observe(1, [])
    client, calls = fake_client(json.dumps({"observation": "x", "land_ahead": "5-10 m",
                                            "turn_deg": 30, "done": False}))
    d = ClaudePilot(client=client).decide(obs)
    assert d.land_ahead == "5-10 m"
    assert "land_ahead" in calls[0]["system"] and "distance over the" in calls[0]["system"]
    nxt = make_run().observe(2, [d]).instruments_text("")
    assert "land ahead 5-10 m" in nxt


def test_prompt_explains_tacks_and_when_a_task_is_done(make_run):
    client, calls = fake_client(ANSWER)
    ClaudePilot(client=client).decide(make_run().observe(1, []))
    system = calls[0]["system"]
    assert "the autopilot carries out the tack for you" in system
    assert "not while it is approaching them" in system


# --- models -----------------------------------------------------------------------

@pytest.mark.parametrize("model, off", [
    ("claude-haiku-5-5", "disabled"),
    ("claude-sonnet-5-5", "between_tools"),
    ("claude-some-future-model", "disabled"),
])
def test_no_thinking_uses_each_models_setting(make_run, model, off):
    client, calls = fake_client(ANSWER)
    ClaudePilot(model=model, thinking=False, client=client).decide(make_run().observe(1, []))
    assert calls[0]["thinking"] == {"type": off}
    assert calls[0]["model"] == model


def test_cost_estimate_knows_the_current_models():
    assert estimate_cost("claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(3.0)
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert estimate_cost("unknown", 10, 10) is None
