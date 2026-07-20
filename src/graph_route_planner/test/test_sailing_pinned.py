"""Numeric characterisation of SailingModel.

These values were captured from the scalar implementation before it was
vectorised. They exist so the vectorised version can be proved identical rather
than merely plausible — a polar diagram that drifts by a knot still looks
reasonable in a plot but silently changes every route.

Nothing here asserts the polar is *correct*; it asserts it has not *changed*.
"""

import math

import numpy as np
import pytest

from graph_route_planner.sailing import SailingModel

# (angle_deg, expected_speed) captured from the scalar implementation.
POLAR_45 = [[0, 0.0], [10, 0.0], [44.9, 0.0], [45, 5.0], [45.1, 5.006666666666667], [60, 5.999999999999999], [89.9, 7.993333333333334], [90, 8.0], [90.1, 8.004444444444443], [110, 8.88888888888889], [134.9, 9.995555555555555], [135, 10.0], [135.1, 9.995555555555557], [160, 8.88888888888889], [179.9, 8.004444444444445], [180, 8.0]]
POLAR_50 = [[0, 0.0], [49.9, 0.0], [50, 5.0], [70, 6.5], [90, 8.0], [120, 9.333333333333332], [135, 10.0], [150, 9.333333333333334], [180, 8.0]]
OPTIMAL = {'45': [0.7853981633974483, 3.5355339059327378], '50': [0.8726646259971648, 3.2139380484326967]}

# bearing -> [start, end, angle_off_wind, sailable, sailing_time]
LEGS = {
    0.0: [[[0, 0], [0, 10], 0.0, False, 3.1114476537208247], [[0, 0], [10, 0], 1.5707963267948966, True, 1.25], [[0, 0], [0, -10], 3.141592653589793, True, 1.25], [[0, 0], [7, 7], 0.7853981633974483, False, 2.178013357604577], [[0, 0], [-7, -7], 2.356194490192345, True, 0.9899494936611666], [[0, 0], [3, -9], 2.819842099193151, True, 1.0756862303114052], [[5, 5], [5, 5], 0.0, True, 0.0], [[-10, 60], [-10, -10], 3.141592653589793, True, 8.75], [[60, -10], [60, 60], 0.0, False, 21.78013357604577], [[60, -10], [104, 26], 0.8850668158886104, True, 11.2502213713144], [[104, 26], [60, 60], 0.9129077216126866, True, 10.74936911486125]],
    0.1: [[[0, 0], [0, 10], 0.0017453292519978123, False, 3.111442914716247], [[0, 0], [10, 0], 1.5690509975429023, True, 1.2511729746637472], [[0, 0], [0, -10], 3.1398473243377953, True, 1.2493059411438079], [[0, 0], [7, 7], 0.783652834145454, False, 2.1818113887957056], [[0, 0], [-7, -7], 2.3579398194443395, True, 0.9903896668464317], [[0, 0], [3, -9], 2.8180967699411568, True, 1.075144418284511], [[5, 5], [5, 5], 0.0, True, 0.0], [[-10, 60], [-10, -10], 3.1398473243377953, True, 8.745141588006655], [[60, -10], [60, 60], 0.0017453292519978123, False, 21.78010040301373], [[60, -10], [104, 26], 0.8833214866366161, True, 11.266943546688458], [[104, 26], [60, 60], 0.914653050864681, True, 10.733806654258318]],
    45.0: [[[0, 0], [0, 10], 0.7853981633974483, False, 2.2001257352529677], [[0, 0], [10, 0], 0.7853981633974484, False, 2.2001257352529673], [[0, 0], [0, -10], 2.356194490192345, True, 1.0], [[0, 0], [7, 7], 0.0, False, 3.080176029354155], [[0, 0], [-7, -7], 3.141592653589793, True, 1.2374368670764582], [[0, 0], [3, -9], 2.0344439357957027, True, 1.0333487724328656], [[5, 5], [5, 5], 0.0, True, 0.0], [[-10, 60], [-10, -10], 2.356194490192345, True, 7.0], [[60, -10], [60, 60], 0.7853981633974483, False, 15.400880146770774], [[60, -10], [104, 26], 0.09966865249116297, False, 17.60100588202374], [[104, 26], [60, 60], 1.6983058850101347, True, 6.679610348440474]],
    90.0: [[[0, 0], [0, 10], 1.5707963267948966, True, 1.25], [[0, 0], [10, 0], 0.0, False, 3.1114476537208247], [[0, 0], [0, -10], 1.5707963267948968, True, 1.25], [[0, 0], [7, 7], 0.7853981633974483, False, 2.178013357604577], [[0, 0], [-7, -7], 2.356194490192345, True, 0.9899494936611666], [[0, 0], [3, -9], 1.2490457723982544, True, 1.4336239789725795], [[5, 5], [5, 5], 0.0, True, 0.0], [[-10, 60], [-10, -10], 1.5707963267948968, True, 8.75], [[60, -10], [60, 60], 1.5707963267948966, True, 8.75], [[60, -10], [104, 26], 0.6857295109062863, False, 13.690369676371628], [[104, 26], [60, 60], 2.483704048407583, True, 5.747186912336569]],
    217.3: [[[0, 0], [0, 10], 2.4905848425959083, True, 1.0354348826507134], [[0, 0], [10, 0], 2.221804137788782, True, 1.0354348826507132], [[0, 0], [0, -10], 0.651007810993885, False, 2.4750740956031025], [[0, 0], [7, 7], 3.007202301186231, True, 1.1866736072123738], [[0, 0], [-7, -7], 0.13439035240356242, False, 3.0524026972543203], [[0, 0], [3, -9], 0.9727583653905274, True, 1.7470757461886262], [[5, 5], [5, 5], 0.0, True, 0.0], [[-10, 60], [-10, -10], 0.651007810993885, False, 17.32551866922172], [[60, -10], [60, 60], 2.4905848425959083, True, 7.248044178554993], [[60, -10], [104, 26], 2.907533648695068, True, 6.613600194123302], [[104, 26], [60, 60], 1.5776771209832214, True, 6.935529019465754]],
}


@pytest.mark.parametrize("angle_deg, expected", POLAR_45)
def test_polar_speed_is_unchanged_at_45_no_go(angle_deg, expected):
    model = SailingModel(no_go_deg=45.0)
    assert model.polar_speed(math.radians(angle_deg)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("angle_deg, expected", POLAR_50)
def test_polar_speed_is_unchanged_at_50_no_go(angle_deg, expected):
    model = SailingModel(no_go_deg=50.0)
    assert model.polar_speed(math.radians(angle_deg)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("no_go", ["45", "50"])
def test_optimal_upwind_angle_is_unchanged(no_go):
    model = SailingModel(no_go_deg=float(no_go))
    angle, vmg = OPTIMAL[no_go]
    assert model.optimal_tack_angle == pytest.approx(angle, rel=1e-12)
    assert model.optimal_vmg == pytest.approx(vmg, rel=1e-12)


@pytest.mark.parametrize("bearing", list(LEGS))
def test_leg_geometry_and_cost_are_unchanged(bearing):
    model = SailingModel.from_bearing(bearing, no_go_deg=50.0)
    for u, v, angle, sailable, time in LEGS[bearing]:
        u, v = tuple(u), tuple(v)
        assert model.angle_off_wind(u, v) == pytest.approx(angle, rel=1e-12, abs=1e-12)
        assert model.heading_is_sailable(u, v) is sailable
        assert model.sailing_time(u, v) == pytest.approx(time, rel=1e-12, abs=1e-12)


def test_no_go_branch_of_sailing_time_is_exercised():
    """Guard the goldens: a leg dead upwind must take the tacking branch."""
    model = SailingModel.from_bearing(0.0, no_go_deg=50.0)
    assert not model.heading_is_sailable((0.0, 0.0), (0.0, 10.0))
    assert model.sailing_time((0.0, 0.0), (0.0, 10.0)) > 0.0


# ── array API: must agree with the scalar API element for element ─────────────

def _legs_arrays(legs):
    u = np.array([leg[0] for leg in legs], dtype=float)
    v = np.array([leg[1] for leg in legs], dtype=float)
    return u, v


@pytest.mark.parametrize("bearing", list(LEGS))
def test_array_angle_off_wind_matches_scalar(bearing):
    model = SailingModel.from_bearing(bearing, no_go_deg=50.0)
    u, v = _legs_arrays(LEGS[bearing])
    got = model.angles_off_wind(u, v)
    want = [model.angle_off_wind(tuple(a), tuple(b)) for a, b in zip(u, v)]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bearing", list(LEGS))
def test_array_sailable_matches_scalar(bearing):
    model = SailingModel.from_bearing(bearing, no_go_deg=50.0)
    u, v = _legs_arrays(LEGS[bearing])
    got = model.headings_are_sailable(u, v)
    want = [model.heading_is_sailable(tuple(a), tuple(b)) for a, b in zip(u, v)]
    assert list(got) == want


@pytest.mark.parametrize("bearing", list(LEGS))
def test_array_sailing_time_matches_scalar(bearing):
    model = SailingModel.from_bearing(bearing, no_go_deg=50.0)
    u, v = _legs_arrays(LEGS[bearing])
    got = model.sailing_times(u, v)
    want = [model.sailing_time(tuple(a), tuple(b)) for a, b in zip(u, v)]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_array_polar_speed_matches_scalar_across_the_domain():
    model = SailingModel(no_go_deg=50.0)
    alpha = np.radians(np.linspace(0.0, 180.0, 721))
    got = model.polar_speeds(alpha)
    want = [model.polar_speed(float(a)) for a in alpha]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_array_functions_accept_an_empty_input():
    model = SailingModel(no_go_deg=50.0)
    empty = np.empty((0, 2), dtype=float)
    assert model.sailing_times(empty, empty).shape == (0,)
    assert model.headings_are_sailable(empty, empty).shape == (0,)
