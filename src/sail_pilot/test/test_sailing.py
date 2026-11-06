"""The no-go zone, points of sail and tacks: pure geometry, no simulation."""

import math

import pytest

from sail_pilot.sailing import (GUARD_DEG, NO_GO_DEG, compass_name, crosses_wind, guard_heading,
                                point_of_sail)


@pytest.mark.parametrize("requested, current, expected", [
    (180.0, 200.0, 227.0),     # dead into the wind (from 180): the side the boat is on
    (160.0, 90.0, 133.0),      # 20° off to the east side -> 47° off on that side
    (200.0, 90.0, 227.0),      # 20° off to the west side
    (230.0, 90.0, None),       # 50° off: sailable, untouched
    (0.0, 90.0, None),         # downwind: untouched
])
def test_guard_heading(requested, current, expected):
    got = guard_heading(requested, 180.0, current)
    assert got == (None if expected is None else pytest.approx(expected))


def test_the_guard_uses_the_no_go_angle():
    assert GUARD_DEG == NO_GO_DEG


@pytest.mark.parametrize("current, target, wind, tack", [
    (130, 230, 180, True), (230, 130, 180, True),      # through the wind, either way
    (90, 0, 180, False), (230, 300, 180, False),        # away from the wind
    (350, 40, 10, True), (40, 340, 10, True),           # wrapping past north
])
def test_crosses_wind(current, target, wind, tack):
    assert crosses_wind(current, target, wind) == tack


@pytest.mark.parametrize("degrees, name", [(0, "north"), (180, "south"), (225, "south-west"),
                                           (350, "north"), (100, "east")])
def test_compass_name(degrees, name):
    assert compass_name(degrees) == name


@pytest.mark.parametrize("off, name", [(20, "too close"), (47, "close-hauled"), (70, "close reach"),
                                       (90, "beam reach"), (130, "broad reach"), (175, "running")])
def test_point_of_sail(off, name):
    assert point_of_sail(off).startswith(name)
    assert point_of_sail(-off).startswith(name)


def test_close_hauled_angle_keeps_the_sail_trimmed_tight():
    """The guard's close-hauled course must stay inside the sail controller's
    tight-trim range; at 50° it eased the sail and the boat lost a third of its
    speed."""
    from controller.sail_controller import SailController

    sail = SailController()
    wind = math.radians(180.0)
    close_hauled = sail.compute_sail_angle(
        wind, wind + math.radians(NO_GO_DEG + 1.5), 1.5)   # + controller slack
    tightest = sail.compute_sail_angle(wind, wind + math.radians(45.0), 1.5)
    assert close_hauled == tightest
