"""Tests for route_follower.clearance: looking ahead for land, and steering clear."""

import pytest
from shapely.geometry import Polygon, box

from route_follower.clearance import Clearance, choose_heading

#: A 200 x 200 m basin, centred on the origin, with a 20 x 20 m island
#: whose south-west corner is at (20, 20).
BASIN = box(-100, -100, 100, 100).difference(box(20, 20, 40, 40))


@pytest.fixture(scope="module")
def basin():
    return Clearance(BASIN, clearance=2.0)


# ── free_distance ─────────────────────────────────────────────────────────────

def test_open_water_is_clear_up_to_the_lookahead(basin):
    assert basin.free_distance(0, 0, 180, 50) == 50


def test_distance_stops_short_of_land_by_the_clearance(basin):
    # Due east along north = 30 hits the island's west face at east = 20.
    assert basin.free_distance(0, 30, 90, 50) == pytest.approx(18.0)
    # Due north from the origin hits the basin's edge at north = 100.
    assert basin.free_distance(0, 0, 0, 500) == pytest.approx(98.0)


def test_inside_the_clearance_it_measures_to_land_itself(basin):
    """A boat that is already too close must still tell a way out from a way in."""
    assert basin.free_distance(19, 30, 90, 50) == pytest.approx(1.0)    # 1 m from the island
    assert basin.free_distance(19, 30, 270, 50) == pytest.approx(50.0)  # away from it


def test_on_land_nothing_is_free(basin):
    assert basin.free_distance(30, 30, 0, 50) == 0.0


def test_distance_to_land(basin):
    assert basin.distance_to_land(0, 30) == pytest.approx(20.0)
    assert basin.distance_to_land(30, 30) == 0.0


def test_works_with_a_multipolygon():
    two_ponds = box(0, 0, 10, 10).union(box(20, 0, 30, 10))
    c = Clearance(two_ponds, clearance=1.0)
    assert c.free_distance(5, 5, 90, 50) == pytest.approx(4.0)
    assert c.free_distance(25, 5, 270, 50) == pytest.approx(4.0)


# ── choose_heading ────────────────────────────────────────────────────────────

WIND_FROM_NORTH = 0.0


def choose(clearance, east, north, desired, current=None, wind=WIND_FROM_NORTH,
           lookahead=15.0, drift=0.0):
    return choose_heading(clearance, east, north, desired, wind_direction=wind,
                          no_go_angle=45.0, lookahead=lookahead,
                          current_heading=desired if current is None else current,
                          drift=drift)


def test_clear_heading_is_kept(basin):
    choice = choose(basin, 0, 0, 120)
    assert not choice.overridden and choice.heading == 120


def test_blocked_heading_is_corrected_on_the_same_tack(basin):
    """Heading east at the island's face: bear away a little, stay on this tack."""
    choice = choose(basin, 10, 30, 90)        # island 10 m ahead, lookahead 15
    assert choice.overridden and choice.kind == "correct"
    assert 90 < choice.heading <= 150          # bears away south, a correction
    assert basin.free_distance(10, 30, choice.heading, 15) == 15


def test_changes_tack_when_this_tack_is_walled_in():
    """A beat up a narrow channel: at the east wall, the only way on is the other tack."""
    channel = box(0, 0, 30, 200)              # 30 m wide, running north
    c = Clearance(channel, clearance=1.0)
    # Close-hauled on port (heading 45, wind from north), 2 m off the east wall.
    # Bearing away on port would mean running back down the channel; tack.
    choice = choose(c, 28, 50, 45, lookahead=10)
    assert choice.overridden and choice.kind == "change tack"
    assert 270 <= choice.heading <= 315        # close-hauled on starboard, not in the no-go zone


def test_never_picks_a_heading_in_the_no_go_zone(basin):
    for desired in range(0, 360, 15):
        choice = choose(basin, 10, 18, desired)
        if choice.overridden:
            off_wind = abs((choice.heading - WIND_FROM_NORTH + 180) % 360 - 180)
            assert off_wind >= 45


def test_drift_is_part_of_the_path_checked(basin):
    """Heading 60 from (5, 25) passes north of the island's corner — unless the
    boat slides 30 degrees to leeward, which carries it into the island."""
    assert not choose(basin, 5, 25, 60, wind=330.0).overridden
    assert choose(basin, 5, 25, 60, wind=330.0, drift=30.0).overridden


def test_boxed_in_heads_for_the_most_open_water():
    pocket = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)])
    c = Clearance(pocket, clearance=0.5)
    choice = choose(c, 3, 1, 180, lookahead=20)   # wall 0.5 m ahead
    assert choice.overridden and choice.kind == "open water"
    assert c.free_distance(3, 1, choice.heading, 20) > c.free_distance(3, 1, 180, 20)
