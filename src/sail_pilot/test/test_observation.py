"""What the pilot is shown: the camera image with its marks, and the instruments."""

import math

import numpy as np
import pytest

from sail_pilot.observation import Observation

pytest.importorskip("rclpy", reason="needs the ROS environment (source install/setup.bash)")

from sail_pilot.render import water_distance_rows  # noqa: E402


def bearing_to(east, north, target):
    return math.degrees(math.atan2(target[0] - east, target[1] - north)) % 360


def orange_columns(image):
    r, g, b = (image[..., i].astype(int) for i in range(3))
    mask = (r > 200) & (g > 90) & (g < 180) & (b < 60)
    mask[:30] = False                      # not the scale
    return np.nonzero(mask)[1]


def land_bottom_row(image, column):
    """The lowest row of green land in a column of the onboard image."""
    r, g, b = (image[:, column, i].astype(int) for i in range(3))
    green = (g > r + 30) & (g > b + 30)
    rows = np.nonzero(green)[0]
    return rows.max() if len(rows) else None


# --- the camera image ------------------------------------------------------------

def test_a_buoy_ahead_is_at_zero_and_one_off_the_bow_at_its_bearing(make_run, scenario, orange):
    ahead = bearing_to(*scenario.boat, orange)
    for offset in (0.0, 20.0):
        run = make_run(heading=ahead - offset)
        obs = run.observe(1, [])
        cols = orange_columns(obs.image)
        assert len(cols) > 10
        w, h = run.image_size
        f = (h / 2) / math.tan(math.radians(run.fovy) / 2)
        expected = w / 2 + f * math.tan(math.radians(offset))
        assert abs(np.median(cols) - expected) < 8, (offset, np.median(cols), expected)


def test_the_image_sent_is_a_jpeg_of_the_shown_image(make_run):
    obs = make_run().observe(1, [])
    assert obs.jpeg[:2] == b"\xff\xd8"
    assert obs.image.shape == (480, 640, 3)


def test_half_field_of_view_follows_the_image_shape(make_run):
    assert make_run().half_fov == pytest.approx(43.0, abs=0.5)
    assert make_run(image_size=(640, 320)).half_fov > 50


@pytest.mark.parametrize("start, heading, nearer, farther", [
    ((50.9, 31.0), 312.8, 10, 20),     # land 13.7 m ahead (the run's decision 39)
    ((47.2, 34.2), 301.2, 5, 10),      # land 6.2 m ahead (decision 41)
])
def test_distance_marks_bracket_the_shore_ahead(make_run, start, heading, nearer, farther):
    run = make_run(start=start, heading=heading)
    for _ in range(60):
        run.sim.step()
    rows = water_distance_rows(run.sim, 640, 480)
    assert rows[5] > rows[10] > rows[20] > rows[50]          # nearer = lower in the image
    image = run.sim.render_onboard(640, 480)
    bottom = land_bottom_row(image, 320)
    assert rows[farther] < bottom < rows[nearer]
    s = run.sim.get_status()
    true = run.land_distance(s["x"], s["y"], s["heading"])
    assert nearer < true < farther


def test_marks_are_drawn_on_both_edges(make_run):
    obs = make_run().observe(1, [])
    left, right = obs.image[:, 3], obs.image[:, -4]
    assert (left == 255).all(axis=1).sum() >= 4 and (right == 255).all(axis=1).sum() >= 4


def test_true_land_distance_is_recorded_not_shown(make_run):
    obs = make_run(start=(47.2, 34.2), heading=301.2).observe(1, [])
    assert obs.true_land_ahead == pytest.approx(6.2, abs=1.0)
    assert "6.2" not in obs.instruments_text("")


# --- the instruments ---------------------------------------------------------------

def test_wind_is_given_relative_to_the_bow(make_run):
    obs = make_run(heading=290.0).observe(1, [])
    assert obs.wind_from_rel == pytest.approx(-110.0, abs=1.0)    # from 180°: to port
    assert "110° to port" in obs.instruments_text("")
    obs = make_run(heading=170.0).observe(1, [])
    assert obs.in_no_go and "no-go zone" in obs.instruments_text("")


def test_instruments_give_true_wind_and_the_no_go_zone(make_run):
    text = make_run(heading=200.0).observe(1, []).instruments_text("")
    assert "from 180° (south), that is 20° to port of the bow" in text
    assert "headings from 133° clockwise to 227°" in text
    assert "sails close-hauled on 133° or 227°" in text
    assert "point of sail: too close to the wind: the boat stalls, 20° off the wind" in text


def test_no_stall_warning_on_the_boats_own_close_hauled_course(make_run):
    # The boat settles ~47° off the wind close-hauled: sailable, no warning.
    text = make_run(heading=227.0).observe(1, []).instruments_text("")
    assert "too close" not in text and "point of sail: close-hauled" in text
    text = make_run(heading=220.0).observe(1, []).instruments_text("")
    assert "too close to the wind: the boat stalls, 40° off the wind" in text


def test_no_go_zone_wraps_past_north():
    obs = Observation(
        t=0, index=1, image=None, jpeg=b"", heading=90.0, speed=1.0, wind_from_rel=-80.0,
        wind_speed=4.0, colliding=False, east=0, north=0, instruction="x", wind_from=10.0)
    text = obs.instruments_text("")
    assert "from 10° (north)" in text
    assert "headings from 323° clockwise to 57°" in text
    assert obs.close_hauled == (57.0, 323.0)
