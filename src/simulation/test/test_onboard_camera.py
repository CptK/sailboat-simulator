"""
Tests for the camera mounted on the boat.

They check what a policy relies on: the camera looks forward, left and right are
not swapped, and debug overlays never show up in the onboard image.

Rendering needs an OpenGL backend; on a headless machine set MUJOCO_GL=osmesa
(or egl). If no backend is available these tests are skipped.
"""

import mujoco
import numpy as np
import pytest

from simulation.sailboat_simulation import SailboatSimulation

# Magenta does not occur anywhere else in the scene (water, sky, hull, sail).
MAGENTA = (1.0, 0.0, 1.0)
WIDTH, HEIGHT = 160, 120


def make_sim(buoys):
    """A simulation at its initial state: boat at the origin heading East (+x)."""
    sim = SailboatSimulation(buoys=buoys)
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    return sim


def magenta_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 120) & (b > 120) & (g < 80)


@pytest.fixture(autouse=True)
def require_gl():
    sim = make_sim([])
    try:
        sim.render_onboard(WIDTH, HEIGHT)
    except Exception as exc:  # no GL context available
        pytest.skip(f"No OpenGL backend for offscreen rendering: {exc}")
    finally:
        sim.close()


def test_image_shape_and_dtype():
    sim = make_sim([])
    img = sim.render_onboard(WIDTH, HEIGHT)
    assert img.shape == (HEIGHT, WIDTH, 3)
    assert img.dtype == np.uint8
    sim.close()


def test_renderer_is_reused_per_resolution():
    sim = make_sim([])
    sim.render_onboard(WIDTH, HEIGHT)
    sim.render_onboard(WIDTH, HEIGHT)
    sim.render_onboard(64, 48)
    assert len(sim._onboard_renderers) == 2
    sim.close()
    assert len(sim._onboard_renderers) == 0


def test_buoy_ahead_is_visible_near_center():
    sim = make_sim([(20.0, 0.0, *MAGENTA)])
    mask = magenta_mask(sim.render_onboard(WIDTH, HEIGHT))
    assert mask.sum() > 5
    cols = np.nonzero(mask)[1]
    assert abs(cols.mean() - WIDTH / 2) < WIDTH * 0.1
    sim.close()


def test_buoy_behind_is_not_visible():
    sim = make_sim([(-20.0, 0.0, *MAGENTA)])
    assert magenta_mask(sim.render_onboard(WIDTH, HEIGHT)).sum() == 0
    sim.close()


@pytest.mark.parametrize("north, expected_side", [(-8.0, "right"), (8.0, "left")])
def test_starboard_appears_right_and_port_left(north, expected_side):
    # Heading East, so starboard is South (negative north) and port is North.
    sim = make_sim([(20.0, north, *MAGENTA)])
    mask = magenta_mask(sim.render_onboard(WIDTH, HEIGHT))
    assert mask.sum() > 5
    mean_col = np.nonzero(mask)[1].mean()
    if expected_side == "right":
        assert mean_col > WIDTH / 2
    else:
        assert mean_col < WIDTH / 2
    sim.close()


def test_debug_overlays_are_hidden():
    sim = make_sim([])
    wind_arrow = 0  # first mocap body in sailboat.xml

    def render_with_arrow_at(pos):
        sim.data.mocap_pos[wind_arrow] = pos
        mujoco.mj_forward(sim.model, sim.data)
        return sim.render_onboard(WIDTH, HEIGHT)

    in_view = (6.0, 0.0, 1.3)  # right in front of the camera
    far_away = (0.0, 0.0, -100.0)

    # Control: with default visibility the arrow really is in the camera's view.
    sim.data.mocap_pos[wind_arrow] = in_view
    mujoco.mj_forward(sim.model, sim.data)
    with mujoco.Renderer(sim.model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(sim.data, camera="bow_cam")
        with_overlay = renderer.render()
    assert not np.array_equal(with_overlay, render_with_arrow_at(far_away))

    # The onboard image must not change when the overlay moves into view.
    np.testing.assert_array_equal(render_with_arrow_at(in_view),
                                  render_with_arrow_at(far_away))
    sim.close()
