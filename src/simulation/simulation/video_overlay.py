"""Compositing helpers for recorded simulation videos."""

import numpy as np


#: Shown in the top-right corner of the video and viewer while the boat touches land.
COLLISION_TEXT = "Collision detected"


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
    corner: str = "bottom-right",
) -> np.ndarray:
    """Draw ``inset`` in a bottom corner of ``frame``, video-call style.

    Args:
        frame: Main RGB image, shape (H, W, 3), uint8.
        inset: Small RGB image, shape (h, w, 3), uint8. Must fit inside the frame
            together with margin and border.
        margin: Gap in pixels between the inset's border and the frame edges.
        border: Width in pixels of the frame drawn around the inset.
        border_color: RGB colour of that border.
        corner: "bottom-right" or "bottom-left".

    Returns:
        A new image; ``frame`` is not modified.
    """
    out = frame.copy()
    fh, fw = frame.shape[:2]
    ih, iw = inset.shape[:2]
    bottom = fh - margin
    top = bottom - ih - 2 * border
    if corner == "bottom-right":
        right = fw - margin
        left = right - iw - 2 * border
    elif corner == "bottom-left":
        left = margin
        right = left + iw + 2 * border
    else:
        raise ValueError(f"Unknown corner '{corner}'")
    if top < 0 or left < 0 or right > fw:
        raise ValueError(
            f"Inset {iw}x{ih} with margin {margin} and border {border} "
            f"does not fit in a {fw}x{fh} frame"
        )
    out[top:bottom, left:right] = border_color
    out[top + border:bottom - border, left + border:right - border] = inset
    return out


def draw_label(
    frame: np.ndarray,
    text: str,
    font_size: int = 28,
    margin: int = 16,
    padding: int = 10,
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (200, 30, 30),
) -> np.ndarray:
    """Draw ``text`` on a filled box in the top-right corner of ``frame``.

    Args:
        frame: RGB image, shape (H, W, 3), uint8.
        text: The label.
        font_size: Text height in pixels.
        margin: Gap in pixels between the box and the frame edges.
        padding: Gap in pixels between the text and the box edges.
        text_color: RGB colour of the text.
        box_color: RGB colour of the box behind it.

    Returns:
        A new image; ``frame`` is not modified.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    box_w = right - left + 2 * padding
    box_h = bottom - top + 2 * padding
    x0 = frame.shape[1] - margin - box_w
    y0 = margin
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=box_color)
    draw.text((x0 + padding - left, y0 + padding - top), text, font=font, fill=text_color)
    return np.asarray(image)
