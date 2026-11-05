"""Tests for the picture-in-picture compositing used in recorded videos."""

import numpy as np
import pytest

from simulation.video_overlay import draw_label, inset_size, picture_in_picture


def test_inset_size_keeps_4_3_aspect():
    assert inset_size(1280, 0.25) == (320, 240)


def test_inset_is_drawn_bottom_right_with_border():
    frame = np.zeros((100, 200, 3), np.uint8)
    inset = np.full((20, 40, 3), 7, np.uint8)
    out = picture_in_picture(frame, inset, margin=5, border=2)

    # Inset pixels sit inside a border, which sits `margin` from the corner.
    assert (out[100 - 5 - 2 - 20:100 - 5 - 2, 200 - 5 - 2 - 40:200 - 5 - 2] == 7).all()
    assert (out[100 - 5 - 1, 200 - 5 - 1] == 255).all()   # border corner
    assert (out[100 - 5 + 1, 200 - 5 + 1] == 0).all()     # margin untouched
    assert (out[:50, :100] == 0).all()                    # rest of frame untouched
    assert (frame == 0).all()                             # input not modified


def test_inset_too_large_raises():
    with pytest.raises(ValueError):
        picture_in_picture(np.zeros((50, 50, 3), np.uint8), np.zeros((60, 10, 3), np.uint8))


def test_label_is_drawn_top_right_only():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_label(frame, "Collision detected", margin=10)
    changed = np.any(out != 0, axis=2)
    rows, cols = np.nonzero(changed)
    assert rows.min() >= 10 and cols.max() <= 1280 - 10  # inside the margin
    assert rows.max() < 360 and cols.min() > 640          # in the top-right quarter
    assert (frame == 0).all()                             # input not modified


def test_inset_can_go_bottom_left():
    frame = np.zeros((100, 200, 3), np.uint8)
    inset = np.full((20, 40, 3), 7, np.uint8)
    out = picture_in_picture(frame, inset, margin=5, border=2, corner="bottom-left")
    assert (out[100 - 5 - 2 - 20:100 - 5 - 2, 5 + 2:5 + 2 + 40] == 7).all()
    assert (out[:, 100:] == 0).all()                      # right half untouched


def test_unknown_corner_raises():
    with pytest.raises(ValueError):
        picture_in_picture(np.zeros((50, 50, 3), np.uint8), np.zeros((5, 5, 3), np.uint8),
                           corner="top-middle")
