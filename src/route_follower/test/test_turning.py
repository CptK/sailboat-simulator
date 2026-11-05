"""Room to turn: the turn model, and the follower's use of it near land."""

import math

import pytest
from shapely.geometry import box

from route_follower.clearance import Clearance, TurnModel, path_clearance, turn_angle
from route_follower.maneuver_action import JibeManeuver, TackManeuver
from route_follower.node import RouteFollower

from conftest import step, wp


# ── turn_angle ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("current, target, wind, jibe, expected", [
    (45, 315, 0, False, -90),      # tack through the wind, the short way
    (45, 315, 0, True, 270),       # jibe: away from the wind, the long way round
    (135, 225, 0, True, 90),       # jibe whose short way is already downwind
    (135, 225, 180, False, 90),    # tack in a southerly
    (350, 10, 90, False, 20),      # across north
])
def test_turn_angle(current, target, wind, jibe, expected):
    assert turn_angle(current, target, wind, jibe) == pytest.approx(expected)


# ── TurnModel ─────────────────────────────────────────────────────────────────

def test_path_carries_on_before_it_turns():
    model = TurnModel(delay=1.5, rate=60, settle=2.0)
    path = model.path(0, 0, 0, 90, speed=2.0, dt=0.1)
    # 1.5 s straight north at 2 m/s ...
    e, n = path[15]
    assert e == pytest.approx(0, abs=1e-9) and n == pytest.approx(3.0)
    # ... then 1.5 s to swing 90 degrees, then 2 s due east.
    e_end, n_end = path[-1]
    assert e_end > 4.0 and n_end > 3.0
    assert len(path) == 1 + round((1.5 + 1.5 + 2.0) / 0.1)


def test_path_assumes_some_speed_when_stalled():
    path = TurnModel(min_speed=0.5).path(0, 0, 0, 90, speed=0.0)
    assert math.dist(path[0], path[-1]) > 1.0


# ── path_clearance ────────────────────────────────────────────────────────────

POND = box(0, 0, 100, 100)
STRICT, HULL = Clearance(POND, 3.0), Clearance(POND, 1.5)


def test_path_in_open_water_is_clear():
    assert path_clearance([STRICT, HULL], [(50, 50), (55, 55), (60, 50)])


def test_path_close_to_land_is_not():
    assert not path_clearance([STRICT, HULL], [(50, 50), (50, 98)])    # 2 m from the north shore


def test_a_boat_already_close_is_held_to_the_hull_clearance():
    """Starting 2 m from shore, the 3 m standard is unmeetable; use the 1.5 m one."""
    assert path_clearance([STRICT, HULL], [(50, 98), (50, 90)])         # heading away
    assert not path_clearance([STRICT, HULL], [(50, 98), (50, 99)])     # heading in


# ── the follower near land ────────────────────────────────────────────────────

#: A basin whose north shore is at north = 100.
BASIN = box(-200, -200, 200, 100)


def near_land(follower, route, boat, heading, wind, speed=2.0, **kwargs):
    node = follower(route=route, boat=boat, heading=heading, wind_direction=wind, **kwargs)
    node.clearance = Clearance(BASIN, 1.5)
    node._turn_checks = [Clearance(BASIN, 3.0), node.clearance]
    node.turn_model = TurnModel()
    node._waiting_for_room = False
    node._building_speed = False
    rad = math.radians(heading)
    node.velocity = (speed * math.sin(rad), speed * math.cos(rad))
    return node


#: Open water with an islet just north-west of the origin.
ISLET_NW = box(-200, -200, 200, 200).difference(box(-30, 6, -1, 46))


def change_of_tack(follower):
    """Close-hauled on port (heading 45, wind from the north), with the next
    leg on starboard (315). Tacking swings the boat left — into the islet.
    Jibing swings it right, the long way round, away from it."""
    node = near_land(follower, [wp(0, 0), wp(-100, 100)], wp(0, 0), heading=45, wind=0,
                     target_index=1)
    node._turn_checks = [Clearance(ISLET_NW, 3.0), Clearance(ISLET_NW, 1.5)]
    tack = TackManeuver(target_heading=315, current_heading=45, heading_step_size=10,
                        wind_direction=0, command_step_size_multiplier=1.5,
                        heading_tolerance=2, max_steps_to_cross_wind=40)
    return node, tack


def test_tack_without_room_becomes_a_jibe(follower):
    node, tack = change_of_tack(follower)
    assert not RouteFollower._turn_fits(node, node.location, 45, 315, 0, jibe=False)
    assert RouteFollower._turn_fits(node, node.location, 45, 315, 0, jibe=True)
    chosen = RouteFollower._maneuver_with_room(node, tack, node.location, 45, 315, 0, 4.0)
    assert isinstance(chosen, JibeManeuver)


def test_no_jibe_instead_in_a_strong_wind(follower):
    node, tack = change_of_tack(follower)
    strong = node.config.safe_jibe_threshold + 1
    assert RouteFollower._maneuver_with_room(node, tack, node.location, 45, 315, 0, strong) is None


def test_no_room_either_way_holds_course(follower):
    """Running at the shore 1.5 m short of the clearance: no turn fits yet."""
    node = near_land(follower, [wp(0, 0), wp(0, -100)], wp(0, 95.5), heading=0, wind=180,
                     target_index=1)
    commanded = step(node)
    assert node.maneuver is None and node._waiting_for_room
    assert commanded == 0                               # holds course, for now


def test_with_room_the_maneuver_starts_as_before(follower):
    node = near_land(follower, [wp(0, 0), wp(0, -100)], wp(0, 0), heading=0, wind=180,
                     target_index=1)
    step(node)
    assert node.maneuver is not None


def test_turns_early_at_a_corner_beside_land(follower):
    """A tacking corner 4 m off the shore: the turn there would carry the boat
    into the 3 m clearance, so at the last moment it turns early."""
    corner = wp(0, 96, is_soft=True)
    after = wp(-80, 40)
    route = [wp(0, 50), corner, after]
    early = []
    for north in [float(n) for n in range(80, 96)]:
        node = near_land(follower, route, wp(0, north), heading=0, wind=45, target_index=1)
        step(node)
        if node.target_index == 2:
            early.append(north)
    assert early, "never turned early"
    assert early[0] >= 86                               # not from far back


def test_never_turns_early_far_from_the_corner(follower):
    """The corner is 100 m on: whatever the geometry, too far to cut."""
    corner = wp(0, 96, is_soft=True)
    node = near_land(follower, [wp(0, -50), corner, wp(-80, 40)], wp(0, -4), heading=0,
                     wind=45, target_index=1)
    step(node)
    assert node.target_index == 1


# ── speed to tack ─────────────────────────────────────────────────────────────

def beat_to_starboard(follower, speed, heading=45.0):
    """Close-hauled on port in a northerly, with the next waypoint on starboard."""
    return near_land(follower, [wp(0, 0), wp(-100, 100)], wp(0, 0), heading=heading, wind=0,
                     target_index=1, speed=speed)


def test_too_slow_to_tack_builds_speed_first(follower):
    node = beat_to_starboard(follower, speed=0.4)
    commanded = step(node)
    assert node.maneuver is None and node._building_speed
    assert commanded == pytest.approx(65.0)            # close reach, still on port
    assert any("Too slow to tack" in line for line in node.logged)


def test_already_reaching_keeps_its_heading_while_building_speed(follower):
    node = beat_to_starboard(follower, speed=0.4, heading=80.0)
    assert step(node) == pytest.approx(80.0)


def test_fast_enough_tacks(follower):
    node = beat_to_starboard(follower, speed=1.2)
    step(node)
    assert isinstance(node.maneuver, TackManeuver)


def test_tacks_once_up_to_speed(follower):
    node = beat_to_starboard(follower, speed=0.4)
    step(node)
    node.velocity = (0.0, 1.0)
    step(node)
    assert isinstance(node.maneuver, TackManeuver) and not node._building_speed


def test_a_slow_jibe_is_not_held(follower):
    """Running in a northerly (heading 140) onto the other gybe (220): a jibe,
    which needs no speed to get round."""
    node = near_land(follower, [wp(0, 0), wp(-64.28, -76.60)], wp(0, 0),
                     heading=140, wind=0, target_index=1, speed=0.3)
    step(node)
    assert isinstance(node.maneuver, JibeManeuver)
