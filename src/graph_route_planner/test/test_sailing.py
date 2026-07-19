"""Tests for the boat's sailing model."""

import math

import pytest

from graph_route_planner.sailing import SailingModel

NORTH = (0.0, 1.0)
SOUTH = (0.0, -1.0)


@pytest.fixture
def model():
    return SailingModel()


def test_defaults_to_a_northerly(model):
    assert model.wind_from == pytest.approx(NORTH)


def test_wind_vector_is_normalised():
    m = SailingModel(wind_from=(0.0, 7.0))
    assert m.wind_from == pytest.approx(NORTH)


def test_rejects_zero_wind_vector():
    with pytest.raises(ValueError, match="non-zero"):
        SailingModel(wind_from=(0.0, 0.0))


@pytest.mark.parametrize("bad", [0.0, 90.0, -5.0, 120.0])
def test_rejects_impossible_no_go_zone(bad):
    with pytest.raises(ValueError, match="no_go_deg"):
        SailingModel(no_go_deg=bad)


@pytest.mark.parametrize("bearing, expected", [
    (0.0, (0.0, 1.0)),      # from the north
    (90.0, (1.0, 0.0)),     # from the east
    (180.0, (0.0, -1.0)),   # from the south
    (270.0, (-1.0, 0.0)),   # from the west
])
def test_from_bearing_matches_compass_convention(bearing, expected):
    assert SailingModel.from_bearing(bearing).wind_from == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("bearing", [0.0, 45.0, 90.0, 180.0, 275.0, 359.0])
def test_bearing_round_trips(bearing):
    assert SailingModel.from_bearing(bearing).wind_bearing_deg == pytest.approx(bearing, abs=1e-6)


def test_bearing_is_always_below_360():
    """360 is north, and must be reported as 0 rather than wrapping to 360."""
    assert SailingModel.from_bearing(360.0).wind_bearing_deg == pytest.approx(0.0)


def test_blowing_towards_is_opposite_the_wind_source():
    """An easterly blows towards the west."""
    assert SailingModel.from_bearing(90.0).blowing_towards == pytest.approx((-1.0, 0.0), abs=1e-9)


def test_no_go_zone_has_no_speed(model):
    assert model.polar_speed(math.radians(0)) == 0.0
    assert model.polar_speed(math.radians(44.9)) == 0.0


def test_speed_is_positive_once_out_of_the_no_go_zone(model):
    assert model.polar_speed(math.radians(45)) > 0.0


def test_reach_is_faster_than_close_hauled(model):
    close_hauled = model.polar_speed(math.radians(50))
    reach = model.polar_speed(math.radians(115))
    assert reach > close_hauled


def test_optimal_tack_angle_is_outside_the_no_go_zone(model):
    assert math.degrees(model.optimal_tack_angle) >= model.no_go_deg
    assert model.optimal_vmg > 0.0


def test_heading_straight_upwind_is_unsailable(model):
    assert not model.heading_is_sailable((0.0, 0.0), (0.0, 10.0))


def test_heading_downwind_is_sailable(model):
    assert model.heading_is_sailable((0.0, 0.0), (0.0, -10.0))


def test_wind_is_a_parameter_not_a_global():
    """The same leg flips sailability when the wind turns around."""
    leg = ((0.0, 0.0), (0.0, 10.0))
    northerly = SailingModel(wind_from=NORTH)
    southerly = SailingModel(wind_from=SOUTH)

    assert not northerly.heading_is_sailable(*leg)
    assert southerly.heading_is_sailable(*leg)


def test_two_models_coexist():
    """Two winds in one process must not interfere."""
    a = SailingModel.from_bearing(0.0)
    b = SailingModel.from_bearing(180.0)
    assert a.wind_from == pytest.approx(NORTH)
    assert b.wind_from == pytest.approx(SOUTH, abs=1e-9)


def test_zero_length_leg_costs_nothing(model):
    assert model.sailing_time((5.0, 5.0), (5.0, 5.0)) == 0.0


def test_upwind_leg_costs_more_than_downwind(model):
    """Tacking upwind is slower than running the same distance downwind."""
    upwind = model.sailing_time((0.0, 0.0), (0.0, 100.0))
    downwind = model.sailing_time((0.0, 0.0), (0.0, -100.0))
    assert upwind > downwind
    assert math.isfinite(upwind)


def test_sailing_time_scales_with_distance(model):
    short = model.sailing_time((0.0, 0.0), (10.0, 0.0))
    long = model.sailing_time((0.0, 0.0), (20.0, 0.0))
    assert long == pytest.approx(2 * short)
