"""The run loop: decisions, the no-go guard, groundings, tacks, and tasks."""

import pytest

from sail_pilot.sailing import wrap180
from sail_pilot.task import Task, make_task

pytest.importorskip("rclpy", reason="needs the ROS environment (source install/setup.bash)")

from pilot_helpers import FakePilot, Plan  # noqa: E402

from sail_pilot.harness import HISTORY, MAX_CONSECUTIVE_ERRORS, SETTLED_SPEED  # noqa: E402
from sail_pilot.pilots.scripted import ScriptedPilot  # noqa: E402


# --- the loop -----------------------------------------------------------------

def test_scripted_pilot_reaches_the_orange_buoy(make_run):
    run = make_run()
    task = run.task
    result = run.run(ScriptedPilot(task), mode="paused", period=2.0, max_time=120)
    assert result.success and result.end_reason == "reached the target"
    assert result.final_distance <= task.radius
    assert result.groundings == 0


def test_a_decision_sets_the_course_relative_to_the_heading_it_saw(make_run):
    run = make_run(heading=0.0)
    pilot = FakePilot(turn=90.0)
    run.run(pilot, mode="paused", period=100.0, max_time=12.0)
    assert run.desired_heading == pytest.approx((pilot.seen[0].heading + 90.0) % 360, abs=0.1)
    assert abs(wrap180(run.sim.get_status()["heading"] - run.desired_heading)) < 15


def test_failed_requests_keep_the_course_and_end_the_run(make_run):
    run = make_run(heading=0.0)
    before = run.desired_heading
    result = run.run(FakePilot(turn=90.0, error="boom"), mode="paused", period=1.0,
                     max_time=60.0)
    assert run.desired_heading == before
    assert "failed requests in a row" in result.end_reason
    assert len(result.decisions) == MAX_CONSECUTIVE_ERRORS


def test_done_ends_the_run_without_counting_as_success(make_run):
    result = make_run().run(FakePilot(done=True), mode="paused", max_time=60.0)
    assert result.end_reason == "the pilot said it was done"
    assert not result.success and len(result.decisions) == 1


def test_history_is_shown_to_the_pilot(make_run):
    pilot = FakePilot(turn=5.0)
    make_run().run(pilot, mode="paused", period=1.0, max_time=10.0)
    last = pilot.seen[-1]
    assert len(last.history) == HISTORY
    assert "Your recent decisions" in last.instruments_text("")


def test_realtime_mode_sails_on_while_the_pilot_thinks(make_run):
    run = make_run(heading=0.0)
    pilot = FakePilot(turn=10.0, delay=0.4)
    result = run.run(pilot, mode="realtime", period=0.0, max_time=2.5)
    assert result.wall_time == pytest.approx(result.sim_time, abs=0.6)
    times = [obs.t for obs in pilot.seen]
    assert len(times) >= 3
    # The next request goes out when the answer arrives: ~0.4 s of sailing apart.
    assert min(b - a for a, b in zip(times, times[1:])) >= 0.3


# --- tasks --------------------------------------------------------------------

def test_task_from_a_colour_or_number(scenario, orange):
    task = make_task(scenario, "orange", None, 5.0)
    assert task.target == orange and task.instruction == "Sail to the orange buoy."
    number = str(next(i for i, b in enumerate(scenario.buoys, 1) if b.color == "orange"))
    assert make_task(scenario, number, None, 5.0).target == orange
    with pytest.raises(SystemExit, match="numbers"):
        make_task(scenario, "yellow", None, 5.0)       # two yellow buoys
    with pytest.raises(SystemExit, match="colours"):
        make_task(scenario, "purple", None, 5.0)
    free = make_task(scenario, None, "Circle the island.", 5.0)
    assert free.target is None and not free.succeeded(0, 0)


# --- the no-go guard -------------------------------------------------------------

def test_guard_replaces_a_no_go_request_and_says_so(make_run):
    # Landgraben's wind comes from 180°. Heading 90°, turn +80° asks for 170°.
    run = make_run(heading=90.0)
    pilot = FakePilot(turn=80.0)
    result = run.run(pilot, mode="paused", period=100.0, max_time=1.0)
    d = result.decisions[0]
    assert d.new_heading == pytest.approx(170.0, abs=1)
    assert d.steered_heading == pytest.approx(133.0, abs=1)
    assert run.desired_heading == pytest.approx(133.0, abs=1)
    text = run.observe(2, [d]).instruments_text("")
    assert "the autopilot steered 133° instead" in text


def test_guard_can_be_turned_off(make_run):
    run = make_run(heading=90.0, guard=False)
    result = run.run(FakePilot(turn=80.0), mode="paused", period=100.0, max_time=1.0)
    assert result.decisions[0].steered_heading is None
    assert run.desired_heading == pytest.approx(170.0, abs=1)


# --- stopping on grounding ------------------------------------------------------

#: No target, so no buoy on the way can end these runs early.
HOLD = Task("Hold your course.")

def test_a_run_ends_once_the_boat_is_stuck_on_land(make_run):
    # Holding 270° from the start runs the boat onto the west shore and stops it.
    run = make_run(HOLD, heading=270.0)
    result = run.run(FakePilot(), mode="paused", period=100.0, max_time=120.0,
                     stop_on_grounding=10.0)
    assert result.end_reason == "stuck on land for 10 s"
    assert result.sim_time < 120.0


def test_scraping_along_the_shore_is_not_stuck(make_run):
    # Holding 100° grinds along the south-east shore but keeps moving.
    run = make_run(HOLD, heading=100.0)
    result = run.run(FakePilot(), mode="paused", period=100.0, max_time=60.0,
                     stop_on_grounding=10.0)
    assert result.groundings > 0 and result.end_reason == "time limit"


def test_stop_on_grounding_off_sails_on(make_run):
    run = make_run(HOLD, heading=270.0)
    result = run.run(FakePilot(), mode="paused", period=100.0, max_time=80.0,
                     stop_on_grounding=0.0)
    assert result.end_reason == "time limit"


# --- tacks carried out by the autopilot ---------------------------------------------

def test_a_tack_is_carried_out_without_asking_the_pilot(make_run):
    run = make_run(start=(0.0, -55.0), heading=130.0)
    pilot = Plan(lambda t: 130.0 if t < 20 else 230.0)
    run.run(pilot, mode="paused", period=2.0, max_time=40.0)
    (tack,) = run.maneuvers
    assert tack["outcome"].startswith("tacked onto 230") and tack["speed"] >= 0.8
    during = [t for t in pilot.asked if tack["start"] < t < tack["end"] - 1e-6]
    assert during == []                                       # not asked mid-tack
    after = [t for t in pilot.asked if t >= tack["end"] - 1e-6]
    assert all(b - a >= 1.9 for a, b in zip(after, after[1:]))   # no burst afterwards
    assert abs(wrap180(run.sim.get_status()["heading"] - 230.0)) < 10
    # The pilot is asked only once the boat is moving again.
    assert tack["final_speed"] >= SETTLED_SPEED


def test_a_slow_boat_picks_up_speed_before_tacking(make_run):
    run = make_run(start=(0.0, -55.0), heading=130.0)        # at rest
    trace = []
    run.run(Plan(lambda t: 230.0), mode="paused", period=2.0, max_time=40.0, trace=trace)
    (tack,) = run.maneuvers
    assert tack["speed"] < 0.8 and tack["outcome"].startswith("tacked onto 230")
    # It first bore away to 65° off the wind (115°) to pick up speed, then tacked.
    early_commands = {row[4] for row in trace if row[0] < 3.0}
    assert 115.0 in early_commands


def test_the_history_says_how_the_tack_went(make_run):
    run = make_run(start=(0.0, -55.0), heading=130.0)
    pilot = Plan(lambda t: 130.0 if t < 20 else 230.0)
    result = run.run(pilot, mode="paused", period=2.0, max_time=40.0)
    tacked = [i for i, d in enumerate(result.decisions) if d.maneuver]
    assert tacked and "the autopilot tacked onto 230°" in result.decisions[tacked[0]].maneuver
    # The next decision after the tack sees it in its history.
    text = run.observe(99, result.decisions[:tacked[0] + 2]).instruments_text("")
    assert "the autopilot tacked onto 230°" in text


def test_an_ordinary_turn_is_not_a_tack(make_run):
    run = make_run(start=(0.0, -55.0), heading=130.0)
    run.run(Plan(lambda t: 90.0), mode="paused", period=2.0, max_time=10.0)
    assert run.maneuvers == [] and run.maneuver is None
