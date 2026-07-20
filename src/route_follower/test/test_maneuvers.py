"""Maneuver selection and the jibe fallback."""

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
