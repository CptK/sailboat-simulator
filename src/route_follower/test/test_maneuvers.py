"""Maneuver selection and the jibe fallback."""

import pytest

from route_follower.maneuver_action import JibeManeuver, TackManeuver

from conftest import step, wp


def test_a_large_turn_starts_a_maneuver(follower):
    boat = wp(0.0, 0.0)
    node = follower(route=[boat, wp(0.0, 50.0)], target_index=1, boat=boat, heading=180.0)
    step(node)
    assert node.maneuver is not None


def test_a_small_turn_just_steers(follower):
    boat = wp(0.0, 0.0)
    node = follower(route=[boat, wp(0.0, 50.0)], target_index=1, boat=boat, heading=5.0)
    step(node)
    assert node.maneuver is None


def test_crossing_the_wind_selects_a_tack(follower):
    """Wind from the north, turning from port bow to starboard bow."""
    boat = wp(0.0, 0.0)
    node = follower(route=[boat, wp(40.0, 40.0)], target_index=1, boat=boat,
                    heading=315.0, wind_direction=0.0)
    step(node)
    assert isinstance(node.maneuver, TackManeuver)


def test_a_downwind_turn_in_light_air_selects_a_jibe(follower):
    boat = wp(0.0, 0.0)
    node = follower(route=[boat, wp(-40.0, -40.0)], target_index=1, boat=boat,
                    heading=135.0, wind_direction=0.0, wind_speed=4.0)
    step(node)
    assert isinstance(node.maneuver, JibeManeuver)


def test_strong_wind_forbids_the_jibe(follower):
    boat = wp(0.0, 0.0)
    node = follower(route=[boat, wp(-40.0, -40.0)], target_index=1, boat=boat,
                    heading=135.0, wind_direction=0.0, wind_speed=40.0)
    step(node)
    assert isinstance(node.maneuver, TackManeuver)


def test_a_stuck_tack_falls_back_to_a_jibe(follower):
    boat = wp(0.0, 0.0)
    node = follower(route=[boat, wp(40.0, 40.0)], target_index=1, boat=boat,
                    heading=315.0, wind_direction=0.0)
    step(node)
    assert isinstance(node.maneuver, TackManeuver)

    # Hold the heading so the boat never comes through the wind.
    for _ in range(node.config.max_steps_to_cross_wind_when_tacking + 2):
        step(node)
    assert isinstance(node.maneuver, JibeManeuver)
    assert any("stuck" in str(m) for m in node.logged)


def test_the_stuck_threshold_is_the_configured_one(follower):
    """40 steps at 0.2 s is ~8 s; the old 500 was 100 s and never fired."""
    boat = wp(0.0, 0.0)
    node = follower(route=[boat, wp(40.0, 40.0)], target_index=1, boat=boat,
                    heading=315.0, wind_direction=0.0)
    node.config.max_steps_to_cross_wind_when_tacking = 3
    step(node)
    for _ in range(5):
        step(node)
    assert isinstance(node.maneuver, JibeManeuver)


# ── steering through a maneuver ──────────────────────────────────────────────
#
# The incremental ramp was removed. It capped the heading error presented to the
# rudder controller at ~17 deg, which is far too gentle to carry a bow through
# the eye of the wind — the boat lost way and stalled. These pin what replaced
# it, and what had to survive: the turn direction, and the jibe fallback.

def _tack(cur=320.0, tgt=40.0, wind=0.0, **kw):
    kw.setdefault("max_steps_to_cross_wind", 40)
    return TackManeuver(target_heading=tgt, current_heading=cur, heading_step_size=10.0,
                        wind_direction=wind, command_step_size_multiplier=1.5,
                        heading_tolerance=2.0, **kw)


def _jibe(cur=320.0, tgt=40.0, wind=0.0):
    return JibeManeuver(target_heading=tgt, current_heading=cur, heading_step_size=10.0,
                        wind_direction=wind, command_step_size_multiplier=1.5,
                        heading_tolerance=2.0)


def test_a_tack_commands_the_target_immediately():
    """The full error reaches the controller, so the rudder goes hard over."""
    m = _tack(cur=320.0, tgt=40.0)
    assert m.step(320.0) == pytest.approx(40.0)


def test_a_tack_completes_when_the_heading_is_reached():
    m = _tack(cur=320.0, tgt=40.0)
    m.step(320.0)
    m.step(39.0)
    assert m.complete


def test_a_tack_is_not_complete_part_way_round():
    m = _tack(cur=320.0, tgt=40.0)
    m.step(320.0)
    m.step(350.0)
    assert not m.complete


def test_a_jibe_turns_away_from_the_wind_not_the_short_way():
    """320 -> 40 with wind from north: the short way crosses the eye, which is a
    tack. A jibe must go the other way round, through downwind."""
    m = _jibe(cur=320.0, tgt=40.0, wind=0.0)
    commanded = m.step(320.0)
    turn = (commanded - 320.0) % 360
    assert turn > 180, f"jibe should turn anticlockwise, got a {turn:.0f} deg clockwise turn"


def test_a_jibe_reaches_its_target_after_passing_downwind():
    m = _jibe(cur=320.0, tgt=40.0, wind=0.0)
    m.step(320.0)
    for heading in (270.0, 220.0, 180.0, 140.0, 100.0, 60.0, 41.0):
        command = m.step(heading)
    assert m.complete or command == pytest.approx(40.0)


def test_a_jibe_that_needs_no_crossing_still_reaches_its_target():
    m = _jibe(cur=160.0, tgt=200.0, wind=0.0)
    m.step(160.0)
    for heading in (170.0, 180.0, 190.0, 199.0):
        m.step(heading)
    assert m.complete


def test_the_stuck_fallback_survives_and_is_time_based():
    """The tack->jibe fallback used to count ramp steps. With no ramp it must
    still fire, on elapsed ticks."""
    m = _tack(cur=320.0, tgt=40.0, wind=0.0, max_steps_to_cross_wind=5)
    for _ in range(6):
        m.step(320.0)                    # never comes through the wind
    assert m.is_stuck(320.0)


def test_a_tack_that_crosses_the_wind_is_not_stuck():
    m = _tack(cur=320.0, tgt=40.0, wind=0.0, max_steps_to_cross_wind=5)
    for _ in range(10):
        m.step(40.0)                     # already through
    assert not m.is_stuck(40.0)


def test_a_jibe_is_never_reported_stuck():
    m = _jibe()
    for _ in range(50):
        m.step(320.0)
    assert not m.is_stuck(320.0)
