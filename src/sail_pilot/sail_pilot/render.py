"""Drawing what the pilot sees: marks on the onboard camera image, and the chart."""

from __future__ import annotations

import io
import math

import mujoco
import numpy as np

from simulation.sailboat_simulation import ONBOARD_CAMERA

from .sailing import NO_GO_DEG

#: Distances (m) marked on the camera image's edges, where the water is that far away.
DISTANCE_MARKS = (5, 10, 20, 50)

#: The chart's course line: how far ahead it reaches, and its tick spacing (m).
COURSE_LINE_M = 40.0
COURSE_TICK_M = 10.0

#: The chart (--chart): side in pixels, and the widest area (m) it shows whole;
#: a larger map is shown as this much around the boat.
CHART_SIZE = 480
CHART_VIEW = 300.0
#: Bearing scale drawn on the image, degrees off the bow.
BEARING_TICKS = range(-40, 41, 10)


def draw_bearing_scale(image: np.ndarray, fovy_deg: float) -> np.ndarray:
    """Draw a scale of bearings off the bow along the top of the onboard image.

    Column x of a pinhole camera looks along atan((x - cx) / f) off its axis,
    with f from the vertical field of view; the camera looks straight ahead
    (its slight downward tilt barely changes horizontal angles).
    """
    from PIL import Image, ImageDraw, ImageFont

    h, w = image.shape[:2]
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=max(10, h // 32))
    band = max(18, h // 14)
    draw.rectangle((0, 0, w, band + 2), fill=(255, 255, 255))
    for bearing in BEARING_TICKS:
        x = w / 2 + f * math.tan(math.radians(bearing))
        if not 0 <= x < w:
            continue
        tick = band // 2 if bearing else band
        draw.line((x, band + 2 - tick, x, band + 2), fill=(0, 0, 0), width=2 if bearing else 3)
        label = f"{bearing:+d}" if bearing else "0"
        tw = draw.textlength(label, font=font)
        draw.text((min(max(x - tw / 2, 0), w - tw), 1), label, fill=(0, 0, 0), font=font)
    # Where the bow points: short marks below the scale and at the bottom edge,
    # leaving the middle clear so they never cover a distant buoy.
    mark = max(8, h // 24)
    draw.line((w / 2, band + 2, w / 2, band + 2 + mark), fill=(255, 255, 255), width=2)
    draw.line((w / 2, h - mark, w / 2, h), fill=(255, 255, 255), width=2)
    return np.asarray(pil)


def draw_wind(chart: np.ndarray, minimap, east: float, north: float,
              wind_from: float) -> np.ndarray:
    """Draw the no-go zone as a light wedge from the boat, and a wind arrow.

    The wedge points from the boat into the wind, NO_GO_DEG either side; its
    edges are the closest sailable headings. Compass angles on a north-up
    chart: x = sin(angle), y = -cos(angle).
    """
    from PIL import Image, ImageDraw, ImageFont

    h, w = chart.shape[:2]
    base = Image.fromarray(chart).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bx, by = minimap.to_pixel(east, north)
    radius = 0.28 * min(w, h)
    points = [(bx, by)]
    for i in range(25):
        a = math.radians(wind_from - NO_GO_DEG + i * 2 * NO_GO_DEG / 24)
        points.append((bx + radius * math.sin(a), by - radius * math.cos(a)))
    draw.polygon(points, fill=(255, 255, 255, 70), outline=(255, 255, 255, 200))

    # Wind arrow in the top-left corner, pointing where the wind blows to.
    size = max(14, h // 12)
    cx, cy = size * 1.6, size * 1.8
    to = math.radians(wind_from + 180.0)
    dx, dy = math.sin(to), -math.cos(to)
    tail = (cx - dx * size, cy - dy * size)
    head = (cx + dx * size, cy + dy * size)
    draw.line([tail, head], fill=(255, 255, 255, 255), width=max(2, w // 120))
    side = (-dy, dx)
    draw.polygon([head,
                  (head[0] - dx * size * 0.45 + side[0] * size * 0.3,
                   head[1] - dy * size * 0.45 + side[1] * size * 0.3),
                  (head[0] - dx * size * 0.45 - side[0] * size * 0.3,
                   head[1] - dy * size * 0.45 - side[1] * size * 0.3)],
                 fill=(255, 255, 255, 255))
    font = ImageFont.load_default(size=max(10, h // 28))
    draw.text((cx, cy + size + 4), "wind", fill=(255, 255, 255, 255), font=font, anchor="mt",
              stroke_width=1, stroke_fill=(0, 0, 0, 255))
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


def water_distance_rows(sim, width: int, height: int,
                        distances=DISTANCE_MARKS) -> dict[float, float]:
    """Image rows where the water surface straight ahead is each distance away.

    Projects points on the water plane, ahead of the onboard camera, through the
    camera as it is now: its height and tilt change as the hull heaves and
    pitches, so the rows move with the image. Distances beyond the horizon, or
    off the image, are left out.
    """
    model, data = sim.model, sim.data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, ONBOARD_CAMERA)
    pos = np.array(data.cam_xpos[cam])
    rot = np.array(data.cam_xmat[cam]).reshape(3, 3)       # columns: camera axes
    f = (height / 2) / math.tan(math.radians(float(model.cam_fovy[cam])) / 2)
    ahead = -rot[:, 2].copy()                                # the camera looks along -z
    ahead[2] = 0.0
    ahead /= np.linalg.norm(ahead)
    water_z = float(model.geom("water").pos[2])
    rows = {}
    for d in distances:
        point = pos + d * ahead
        point[2] = water_z
        v = rot.T @ (point - pos)                            # camera frame: x right, y up
        if v[2] >= 0:
            continue
        y = height / 2 - f * (v[1] / -v[2])
        if 0 <= y < height:
            rows[d] = y
    return rows


def draw_distance_marks(image: np.ndarray, rows: dict[float, float]) -> np.ndarray:
    """Short labelled ticks on the left and right edges at the given rows."""
    from PIL import Image, ImageDraw, ImageFont

    h, w = image.shape[:2]
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=max(9, h // 36))
    tick = max(14, w // 28)
    for d, y in rows.items():
        label = f"{d:g} m"
        for x0, x1, tx, anchor in ((0, tick, tick + 3, "lm"), (w - tick, w, w - tick - 3, "rm")):
            draw.line((x0, y, x1, y), fill=(255, 255, 255), width=2)
            draw.text((tx, y), label, fill=(255, 255, 255), font=font, anchor=anchor,
                      stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(pil)


def draw_course_line(chart: np.ndarray, minimap, east: float, north: float,
                     heading: float) -> np.ndarray:
    """A yellow line from the boat along its heading, COURSE_LINE_M long, with a
    tick every COURSE_TICK_M: where the boat goes if it holds its course."""
    from PIL import Image, ImageDraw

    h, w = chart.shape[:2]
    pil = Image.fromarray(chart)
    draw = ImageDraw.Draw(pil)
    a = math.radians(heading)
    dx, dy = math.sin(a), math.cos(a)               # east, north per metre
    x0, y0 = minimap.to_pixel(east, north)
    x1, y1 = minimap.to_pixel(east + COURSE_LINE_M * dx, north + COURSE_LINE_M * dy)
    color = (255, 230, 0)
    draw.line([(x0, y0), (x1, y1)], fill=color, width=max(2, w // 160))
    tick = max(3, w // 90)
    px, py = -(y1 - y0), (x1 - x0)                   # perpendicular, in pixels
    norm = math.hypot(px, py) or 1.0
    px, py = px / norm * tick, py / norm * tick
    d = COURSE_TICK_M
    while d <= COURSE_LINE_M + 1e-9:
        tx, ty = minimap.to_pixel(east + d * dx, north + d * dy)
        draw.line([(tx - px, ty - py), (tx + px, ty + py)], fill=color, width=max(2, w // 200))
        d += COURSE_TICK_M
    return np.asarray(pil)


def draw_scale_and_position(chart: np.ndarray, minimap, east: float, north: float) -> np.ndarray:
    """The boat's exact position as a white dot, and a scale bar bottom-right.

    The minimap draws its own scale bar (bottom-left) only when it follows the
    boat on a large map; then this one is left out.
    """
    from PIL import Image, ImageDraw, ImageFont

    from simulation.minimap import SCALE_BAR_LENGTHS

    h, w = chart.shape[:2]
    pil = Image.fromarray(chart)
    draw = ImageDraw.Draw(pil)
    x, y = minimap.to_pixel(east, north)
    r = max(2, w // 160)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(0, 0, 0))
    if not minimap.follows_boat:
        view_m = w / minimap._scale
        length = min(SCALE_BAR_LENGTHS, key=lambda m: abs(m - view_m / 4))
        bar = length * minimap._scale
        margin = max(4, w // 30)
        tick = max(3, h // 40)
        x1, yb = w - margin, h - margin
        x0 = x1 - bar
        draw.line([(x0, yb - tick), (x0, yb), (x1, yb), (x1, yb - tick)],
                  fill=(255, 255, 255), width=max(1, w // 140))
        font = ImageFont.load_default(size=max(9, h // 30))
        draw.text(((x0 + x1) / 2, yb - tick), f"{length} m", font=font, fill=(255, 255, 255),
                  anchor="mb", stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(pil)


def draw_north(chart: np.ndarray) -> np.ndarray:
    """Mark north on the chart: an 'N' with an arrow in the top-right corner."""
    from PIL import Image, ImageDraw, ImageFont

    h, w = chart.shape[:2]
    pil = Image.fromarray(chart)
    draw = ImageDraw.Draw(pil)
    size = max(10, h // 24)
    x, y = w - 2 * size, size // 2
    draw.polygon([(x, y), (x - size // 3, y + size), (x + size // 3, y + size)],
                 fill=(255, 255, 255))
    font = ImageFont.load_default(size=size)
    draw.text((x, y + size + 2), "N", fill=(255, 255, 255), font=font, anchor="mt",
              stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(pil)


def jpeg_bytes(image: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
