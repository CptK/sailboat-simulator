"""Tests for the top-down minimap inset."""

import math

import numpy as np
from shapely.geometry import Polygon

from simulation.minimap import BOAT_COLOR, LAND_COLOR, WATER_COLOR, Minimap


def _boat_tip(minimap, image, east, north):
    """Offset (dx, dy) in pixels of the boat pixel farthest from its position."""
    ys, xs = np.nonzero(np.all(image == BOAT_COLOR, axis=2))
    cx, cy = minimap.to_pixel(east, north)
    i = np.hypot(xs - cx, ys - cy).argmax()
    return xs[i] - cx, ys[i] - cy


def test_north_is_up_and_east_is_right():
    m = Minimap(200, 150, boat=(0, 0))
    x0, y0 = m.to_pixel(0, 0)
    x1, y1 = m.to_pixel(10, 10)
    assert x1 > x0 and y1 < y0


def test_boat_points_along_heading():
    m = Minimap(300, 300, boat=(0, 0))
    dx, dy = _boat_tip(m, m.render(0, 0, 0.0), 0, 0)               # east
    assert dx > 0 and abs(dy) < dx / 3
    dx, dy = _boat_tip(m, m.render(0, 0, math.pi / 2), 0, 0)       # north
    assert dy < 0 and abs(dx) < -dy / 3


def test_map_frames_water_with_land_around_it():
    water = Polygon([(-100, -50), (100, -50), (100, 50), (-100, 50)],
                    holes=[[(-10, -10), (10, -10), (10, 10), (-10, 10)]])  # an island
    m = Minimap(160, 120, water=water, boat=(-50, 0))
    image = m.render(-50, 0, 0.0)
    assert image.shape == (120, 160, 3)
    assert tuple(image[2, 2]) == LAND_COLOR                    # outside the map
    assert tuple(image[tuple(reversed([int(v) for v in m.to_pixel(50, 30)]))]) == WATER_COLOR
    assert tuple(image[tuple(reversed([int(v) for v in m.to_pixel(0, 0)]))]) == LAND_COLOR
    # The whole water area is in view.
    for corner in [(-100, -50), (100, 50)]:
        x, y = m.to_pixel(*corner)
        assert 0 <= x < 160 and 0 <= y < 120


def test_open_water_grows_to_keep_route_in_view():
    m = Minimap(100, 100, boat=(0, 0), view_size=None)
    m.set_route([(0, 0), (500, -300)])
    x, y = m.to_pixel(500, -300)
    assert 0 <= x < 100 and 0 <= y < 100


def test_large_map_zooms_in_and_follows_the_boat():
    water = Polygon([(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)])
    m = Minimap(200, 150, water=water, boat=(0, 0), view_size=100)
    assert m.follows_boat
    for east, north in [(0, 0), (300, -200), (990, 990)]:
        m.render(east, north, 0.0)
        assert m.to_pixel(east, north) == (100, 75)              # boat stays centred
    # 100 m across 200 px: a point 50 m east of the boat is at the right edge.
    assert m.to_pixel(990 + 50, 990)[0] == 200


def test_follow_view_shows_land_beyond_the_map_edge():
    water = Polygon([(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)])
    m = Minimap(200, 150, water=water, boat=(0, 0), view_size=100)
    image = m.render(990, 0, 0.0)                    # 10 m from the east edge
    assert tuple(image[40, 60]) == WATER_COLOR       # 20 m west, off the track
    assert tuple(image[40, 140]) == LAND_COLOR       # 30 m east: off the map


def test_small_map_is_shown_whole_even_with_a_view_size():
    water = Polygon([(-40, -40), (40, -40), (40, 40), (-40, 40)])
    m = Minimap(200, 150, water=water, boat=(0, 0), view_size=300)
    assert not m.follows_boat
    before = m.to_pixel(0, 0)
    m.render(20, 20, 0.0)
    assert m.to_pixel(0, 0) == before                # the view does not move
