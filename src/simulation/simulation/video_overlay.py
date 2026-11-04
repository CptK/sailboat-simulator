"""Compositing helpers for recorded simulation videos."""

import numpy as np


def inset_size(frame_width: int, scale: float, aspect: float = 4 / 3) -> tuple[int, int]:
    """Size (width, height) of a picture-in-picture inset.

    Args:
        frame_width: Width of the main video frame in pixels.
        scale: Inset width as a fraction of the frame width.
        aspect: Inset width / height. 4:3 matches the onboard camera's default.
    """
    width = max(2, int(round(frame_width * scale)))
    height = max(2, int(round(width / aspect)))
    return width, height


def picture_in_picture(
    frame: np.ndarray,
    inset: np.ndarray,
    margin: int = 16,
    border: int = 2,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw ``inset`` in the bottom-right corner of ``frame``, video-call style.

    Args:
        frame: Main RGB image, shape (H, W, 3), uint8.
        inset: Small RGB image, shape (h, w, 3), uint8. Must fit inside the frame
            together with margin and border.
        margin: Gap in pixels between the inset's border and the frame edges.
        border: Width in pixels of the frame drawn around the inset.
        border_color: RGB colour of that border.

    Returns:
        A new image; ``frame`` is not modified.
    """
    out = frame.copy()
    fh, fw = frame.shape[:2]
    ih, iw = inset.shape[:2]
    bottom = fh - margin
    right = fw - margin
    top = bottom - ih - 2 * border
    left = right - iw - 2 * border
    if top < 0 or left < 0:
        raise ValueError(
            f"Inset {iw}x{ih} with margin {margin} and border {border} "
            f"does not fit in a {fw}x{fh} frame"
        )
    out[top:bottom, left:right] = border_color
    out[top + border:bottom - border, left + border:right - border] = inset
    return out
