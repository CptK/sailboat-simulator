"""The chart (--chart): boat, track, no-go wedge, scale bar and course line."""

import json
import math

import pytest

pytest.importorskip("rclpy", reason="needs the ROS environment (source install/setup.bash)")

from pilot_helpers import FakePilot, fake_client  # noqa: E402

from sail_pilot.pilots.claude import ClaudePilot  # noqa: E402
from sail_pilot.render import COURSE_LINE_M  # noqa: E402


def test_chart_shows_the_boat_and_its_track(make_run):
    run = make_run(chart=True, heading=310.0)
    first = run.observe(1, [])
    assert first.chart is not None and first.chart.shape == (480, 480, 3)
    assert first.chart_jpeg[:2] == b"\xff\xd8"
    assert 150 < first.chart_width_m < 320        # Landgraben shown whole
    run.run(FakePilot(), mode="paused", period=100.0, max_time=10.0)
    assert len(run.chart.track) > 5               # the track grew as the boat sailed


def test_no_chart_unless_asked(make_run):
    obs = make_run().observe(1, [])
    assert obs.chart is None and obs.chart_jpeg is None


def test_chart_wedge_points_into_the_wind(make_run):
    # Wind from the south: the wedge lies south of the boat, not north of it.
    run = make_run(chart=True, start=(-55.0, 50.0), heading=200.0)
    obs = run.observe(1, [])
    plain = run.chart.render(-55.0, 50.0, 0.0)
    bx, by = (int(v) for v in run.chart.to_pixel(-55.0, 50.0))
    reach = int(0.2 * obs.chart.shape[0])
    south = obs.chart[by + reach, bx].astype(int) - plain[by + reach, bx].astype(int)
    north = obs.chart[by - reach, bx].astype(int) - plain[by - reach, bx].astype(int)
    assert south.sum() > 30          # lightened by the wedge
    assert abs(north.sum()) < 10     # untouched


def test_chart_has_scale_bar_dot_and_arrow_size(make_run):
    run = make_run(chart=True)
    obs = run.observe(1, [])
    w = obs.chart.shape[1]
    corner = obs.chart[-30:, w // 2:]
    assert (corner == 255).all(axis=2).sum() > 30            # the scale bar
    x, y = (int(v) for v in run.chart.to_pixel(*run.start))
    assert (obs.chart[y, x] > 200).all()                     # the white dot
    assert 8 < obs.arrow_length_m < 20
    client, calls = fake_client(json.dumps({"observation": "x", "land_ahead": "none",
                                            "turn_deg": 0, "done": False}))
    ClaudePilot(client=client).decide(obs)
    assert f"about {obs.arrow_length_m:.0f} m long" in calls[0]["system"]
    assert "white dot" in calls[0]["system"]


def test_chart_course_line_runs_along_the_heading(make_run):
    run = make_run(chart=True, start=(37.4, -34.3), heading=321.0)
    obs = run.observe(1, [])

    def yellow_at(metres):
        a = math.radians(obs.heading)
        x, y = (int(v) for v in run.chart.to_pixel(37.4 + metres * math.sin(a),
                                                   -34.3 + metres * math.cos(a)))
        patch = obs.chart[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).astype(int)
        return any(r > 200 and g > 180 and b < 80 for r, g, b in patch)

    assert yellow_at(20.0) and yellow_at(COURSE_LINE_M - 1)
    assert not yellow_at(COURSE_LINE_M + 15)        # ends at its length
    client, calls = fake_client(json.dumps({"observation": "x", "land_ahead": "20-50 m",
                                            "turn_deg": 0, "done": False}))
    ClaudePilot(client=client).decide(obs)
    assert "yellow line from the boat is your current course" in calls[0]["system"]
