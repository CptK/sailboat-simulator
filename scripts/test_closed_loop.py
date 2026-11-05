"""Tests for the deterministic closed-loop harness (scripts/closed_loop.py).

    source install/setup.bash
    pytest scripts/test_closed_loop.py

Needs the ROS environment (the nodes import boat_msgs); skipped without it.
"""

import datetime
import sys
from pathlib import Path

import pytest

pytest.importorskip("boat_msgs.msg", reason="source install/setup.bash first")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import closed_loop  # noqa: E402
from closed_loop import ClosedLoop, load_scenario  # noqa: E402


def _states(result):
    return [(r.t, r.east, r.north, r.heading, r.speed) for r in result.trace]


def test_same_scenario_gives_the_same_run():
    """The point of the harness: no dependence on wall-clock time or CPU load."""
    a = ClosedLoop(load_scenario("landgraben")).run(max_time=60)
    b = ClosedLoop(load_scenario("landgraben")).run(max_time=60)
    assert _states(a) == _states(b)
    assert a.replans == b.replans
    assert len(a.trace) == 601                   # 10 Hz for 60 s, both ends included


def test_node_modules_get_their_clock_back():
    ClosedLoop(load_scenario("landgraben")).run(max_time=1)
    assert closed_loop.planner_module.datetime is datetime.datetime
    assert closed_loop.follower_module.datetime is datetime.datetime


def test_landgraben_lap_completes_without_touching_land():
    """The whole stack works through the harness: plan, follow, control, finish."""
    result = ClosedLoop(load_scenario("landgraben")).run(max_time=600)
    assert result.completed, result.summary()
    assert result.groundings == [], result.summary()
    assert result.replans[0][1] == "initial plan"
    assert result.final_state[0] == pytest.approx(0, abs=6)     # back at the start
    assert result.final_state[1] == pytest.approx(-70, abs=6)


def test_kornaten_tack_beside_islet_keeps_clear_of_land():
    """Regression for the my_course grounding: the upwind leg after buoy 3.

    Starts on the approach to buoy 3 at (123, -119), so the boat rounds it and
    starts the beat north through the islets — where the live run went aground:
    it tacked at a planned corner beside the islet at (150, -126) and the turn
    carried it on. The follower now checks a turn has room before making it.
    """
    loop = ClosedLoop(load_scenario("my_course"), start=(250, -200), heading=307, skip_via=2)
    result = loop.run(max_time=160)
    assert result.groundings == [], result.summary()
    assert result.min_land_distance >= 1.0, result.summary()
