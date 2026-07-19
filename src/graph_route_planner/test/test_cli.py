"""Tests for the non-interactive entry point's own logic."""

import math

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from graph_route_planner import MAPS_DIR
from graph_route_planner.cli import farthest_pair
from graph_route_planner.map_loader import load_kml
from graph_route_planner.planner import NoWaterError


def test_farthest_pair_picks_opposite_corners():
    pair = farthest_pair(box(0, 0, 100, 50))
    assert math.dist(*pair) == pytest.approx(math.hypot(100, 50))


def test_farthest_pair_returns_two_coordinates():
    a, b = farthest_pair(box(0, 0, 10, 10))
    assert len(a) == 2 and len(b) == 2


def test_farthest_pair_points_lie_on_the_area():
    """Endpoints must be sailable, or planning fails before it starts."""
    area = load_kml(MAPS_DIR / "Landgraben.kml").components[0]
    a, b = farthest_pair(area)
    assert area.covers(Point(a))
    assert area.covers(Point(b))
    assert a != b


def test_farthest_pair_spans_multiple_pieces():
    """Eroding can split a body; endpoints may come from different pieces."""
    split = MultiPolygon([box(0, 0, 10, 10), box(90, 0, 100, 10)])
    a, b = farthest_pair(split)
    assert math.dist(a, b) == pytest.approx(math.hypot(100, 10))


def test_farthest_pair_rejects_empty_area():
    """A margin that erodes the whole map must not raise a bare IndexError."""
    with pytest.raises(NoWaterError, match="no navigable water left"):
        farthest_pair(Polygon())


def test_farthest_pair_rejects_over_eroded_map():
    area = load_kml(MAPS_DIR / "Test.kml").components[0]
    with pytest.raises(NoWaterError, match="reduce the margin"):
        farthest_pair(area.buffer(-500.0))
