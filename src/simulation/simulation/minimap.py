"""Top-down minimap of the course, for a picture-in-picture inset in videos.

North is up. Shows land and water, the planned route, the buoys, the track
the boat has sailed and the boat itself as an arrow pointing along its heading.

When the course is wider than the minimap's view, it zooms in and follows the
boat, which stays in the centre; a scale bar shows how far the view reaches.
A course that fits is shown whole. Drawn with PIL rather than MuJoCo: the map
is rasterised once, and each frame only crops it and draws a few lines on top.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WATER_COLOR = (40, 90, 150)
LAND_COLOR = (95, 130, 70)
ROUTE_COLOR = (255, 220, 60)
TRACK_COLOR = (255, 255, 255)
BOAT_COLOR = (230, 40, 40)
DEFAULT_BUOY_COLOR = (255, 140, 0)
SCALE_BAR_COLOR = (255, 255, 255)

#: Extra space around a course shown whole, as a fraction of its extent.
PADDING = 0.05
#: Smallest extent (m) of open water shown whole, so it is not over-zoomed.
MIN_OPEN_WATER_EXTENT = 100.0
#: The track gets a new point only after the boat moved this far (m).
TRACK_STEP = 1.0
#: Scale bar lengths (m) to choose from: the one nearest a quarter of the view.
SCALE_BAR_LENGTHS = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000)
#: Largest side (px) of the pre-rendered map; a bigger map is drawn coarser.
MAX_BACKGROUND_SIDE = 8000


class Minimap:
    """Renders the course from above into a fixed-size RGB image.

    Args:
        width, height: Size of the rendered image in pixels.
        water: The map's navigable water (shapely geometry), or None for open
            water. Everything outside the water is drawn as land.
        boat: The boat's start (east, north) in metres.
        buoys: (east, north, r, g, b) per buoy, colour channels in [0, 1].
        view_size: Width of the area shown (m) when following the boat. If the
            course is no wider than this — or view_size is None or <= 0 — the
            whole course is shown instead and the view does not move.
    """

    def __init__(self, width: int, height: int, water=None,
                 boat: tuple[float, float] = (0.0, 0.0), buoys=(),
                 view_size: float | None = 300.0):
        self.width = width
        self.height = height
        self.water = water if water is not None and not water.is_empty else None
        self.buoys = list(buoys)
        self.route: list[tuple[float, float]] = []
        self.track: list[tuple[float, float]] = [tuple(boat)]
        self.view_size = view_size if view_size and view_size > 0 else None

        if self.water is not None:
            self._bounds = self.water.bounds
        else:
            self._bounds = (boat[0], boat[1], boat[0], boat[1])
            for b in self.buoys:
                self._include(b[0], b[1])
        self._center = tuple(boat)
        self._fit()

    # --- framing -----------------------------------------------------------

    @property
    def follows_boat(self) -> bool:
        """True if the view is zoomed in and centred on the boat."""
        return self._follow

    def _include(self, east: float, north: float) -> bool:
        """Grow the course bounds to hold a point. Returns True if they grew."""
        x0, y0, x1, y1 = self._bounds
        nb = (min(x0, east), min(y0, north), max(x1, east), max(y1, north))
        grew = nb != self._bounds
        self._bounds = nb
        return grew

    def _fit(self):
        """Choose between showing the whole course and following the boat."""
        x0, y0, x1, y1 = self._bounds
        w, h = x1 - x0, y1 - y0
        if self.water is None:
            w = h = max(w, h, MIN_OPEN_WATER_EXTENT)
        w, h = max(w, 1.0), max(h, 1.0)
        usable = 1 - 2 * PADDING
        whole_scale = min(self.width * usable / w, self.height * usable / h)
        follow_scale = self.width / self.view_size if self.view_size else 0.0

        self._follow = follow_scale > whole_scale
        if self._follow:
            self._scale = follow_scale
        else:
            self._scale = whole_scale
            self._center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self._draw_background()

    def to_pixel(self, east: float, north: float) -> tuple[float, float]:
        """Image coordinates (x right, y down) of a point in local metres."""
        cx, cy = self._center
        return (self.width / 2 + (east - cx) * self._scale,
                self.height / 2 - (north - cy) * self._scale)

    def _draw_background(self):
        """Rasterise the map's land and water once, at the minimap's scale.

        The raster covers the map's bounds; anything beyond them is land (or
        water for open water), so a view near the edge is padded with that.
        """
        self._outside = WATER_COLOR if self.water is None else LAND_COLOR
        self._background = None
        if self.water is None:
            return
        x0, y0, x1, y1 = self.water.bounds
        scale = min(self._scale, MAX_BACKGROUND_SIDE / max(x1 - x0, y1 - y0, 1.0))
        bw = max(1, math.ceil((x1 - x0) * scale))
        bh = max(1, math.ceil((y1 - y0) * scale))

        def px(p):
            return ((p[0] - x0) * scale, (y1 - p[1]) * scale)

        image = Image.new("RGB", (bw, bh), LAND_COLOR)
        draw = ImageDraw.Draw(image)
        for poly in getattr(self.water, "geoms", [self.water]):
            draw.polygon([px(p) for p in poly.exterior.coords], fill=WATER_COLOR)
            for hole in poly.interiors:
                draw.polygon([px(p) for p in hole.coords], fill=LAND_COLOR)
        if scale != self._scale:  # map too big to draw at full detail
            image = image.resize((max(1, round((x1 - x0) * self._scale)),
                                  max(1, round((y1 - y0) * self._scale))), Image.NEAREST)
        self._background = np.asarray(image)
        self._background_origin = (x0, y1)  # (east, north) of its top-left pixel

    def _view_background(self) -> np.ndarray:
        """The part of the pre-rendered map under the current view."""
        view = np.empty((self.height, self.width, 3), np.uint8)
        view[:] = self._outside
        if self._background is None:
            return view
        # Top-left of the background, in view pixels.
        left, top = (round(v) for v in self.to_pixel(*self._background_origin))
        bh, bw = self._background.shape[:2]
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + bw, self.width), min(top + bh, self.height)
        if x0 < x1 and y0 < y1:
            view[y0:y1, x0:x1] = self._background[y0 - top:y1 - top, x0 - left:x1 - left]
        return view

    # --- state -------------------------------------------------------------

    def set_route(self, waypoints: list[tuple[float, float]]):
        """The planned route as (east, north) waypoints; empty clears it."""
        self.route = [tuple(wp) for wp in waypoints]
        if self.water is None and any([self._include(*wp) for wp in self.route]):
            self._fit()

    def update_boat(self, east: float, north: float):
        """Record the boat's position in its track, and follow it if zoomed in."""
        last = self.track[-1]
        if math.hypot(east - last[0], north - last[1]) >= TRACK_STEP:
            self.track.append((east, north))
            if self.water is None and self._include(east, north):
                self._fit()
        if self._follow:
            self._center = (east, north)

    # --- drawing -----------------------------------------------------------

    def render(self, east: float, north: float, yaw: float) -> np.ndarray:
        """The minimap with the boat at (east, north), heading `yaw`.

        Args:
            east, north: Boat position in local metres.
            yaw: Boat heading in radians, mathematical convention
                (0 = east, counter-clockwise positive).

        Returns:
            RGB image, shape (height, width, 3), uint8.
        """
        self.update_boat(east, north)
        image = Image.fromarray(self._view_background())
        draw = ImageDraw.Draw(image)
        line_w = max(1, self.width // 160)

        if len(self.route) >= 2:
            draw.line([self.to_pixel(*p) for p in self.route], fill=ROUTE_COLOR, width=line_w)
        track = [self.to_pixel(*p) for p in self.track] + [self.to_pixel(east, north)]
        if len(track) >= 2:
            draw.line(track, fill=TRACK_COLOR, width=line_w)

        r = max(2, self.width // 70)
        for b in self.buoys:
            x, y = self.to_pixel(b[0], b[1])
            color = (tuple(int(round(255 * c)) for c in b[2:5])
                     if len(b) >= 5 else DEFAULT_BUOY_COLOR)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0))

        self._draw_boat(draw, east, north, yaw)
        if self._follow:
            self._draw_scale_bar(draw)
        return np.asarray(image)

    def _draw_boat(self, draw: ImageDraw.ImageDraw, east: float, north: float, yaw: float):
        """A dart pointing along the heading, at a fixed pixel size."""
        x, y = self.to_pixel(east, north)
        size = max(6, self.width // 22)
        # Image y points down, so the north component flips sign.
        fx, fy = math.cos(yaw), -math.sin(yaw)   # forward
        sx, sy = -fy, fx                         # sideways

        def at(forward: float, side: float) -> tuple[float, float]:
            return (x + (fx * forward + sx * side) * size,
                    y + (fy * forward + sy * side) * size)

        dart = [at(1.0, 0), at(-0.5, 0.45), at(-0.25, 0), at(-0.5, -0.45)]
        draw.polygon(dart, fill=BOAT_COLOR, outline=(255, 255, 255))

    def _draw_scale_bar(self, draw: ImageDraw.ImageDraw):
        """A labelled bar in the bottom-left corner, about a quarter of the view wide."""
        view_m = self.width / self._scale
        length = min(SCALE_BAR_LENGTHS, key=lambda m: abs(m - view_m / 4))
        bar = length * self._scale
        margin = max(4, self.width // 30)
        tick = max(3, self.height // 40)
        x0, y = margin, self.height - margin
        line_w = max(1, self.width // 140)
        draw.line([(x0, y - tick), (x0, y), (x0 + bar, y), (x0 + bar, y - tick)],
                  fill=SCALE_BAR_COLOR, width=line_w)
        font = ImageFont.load_default(size=max(9, self.height // 14))
        draw.text((x0 + bar / 2, y - tick), f"{length} m", font=font,
                  fill=SCALE_BAR_COLOR, anchor="mb", stroke_width=1, stroke_fill=(0, 0, 0))
