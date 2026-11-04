"""Tests for building a simulated world around a map of any size."""

import mujoco
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from simulation.build_env import (
    WATER_TILE_M, WORLD_HALF_SIZE, WORLD_MAP_MARGIN, _triangulate, land_from_water,
    world_half_size_for)
from simulation.sailboat_simulation import SailboatSimulation


def comb(teeth: int = 12) -> Polygon:
    """A deeply concave coastline: a strip of land with long narrow teeth."""
    pts = [(0, 0), (teeth * 4, 0)]
    for i in reversed(range(teeth)):
        x = i * 4
        pts += [(x + 3, 20), (x + 2, 20), (x + 2, 2), (x + 1, 2), (x + 1, 20), (x, 20)]
    return Polygon(pts)


def test_triangulation_covers_land_exactly():
    land = comb()
    tris = unary_union(_triangulate(land))
    assert tris.symmetric_difference(land).area == pytest.approx(0, abs=1e-6)


def test_small_maps_keep_the_default_world():
    assert world_half_size_for(box(-40, -40, 40, 40)) == WORLD_HALF_SIZE


def test_large_maps_get_a_world_that_holds_them():
    half = world_half_size_for(box(-390, -330, 380, 330))
    assert half >= 390 + WORLD_MAP_MARGIN


def test_land_is_the_world_minus_the_water():
    water = box(-300, -300, 300, 300).difference(box(-10, -10, 10, 10))
    half = world_half_size_for(water)
    land = unary_union(land_from_water(water, half))
    assert land.area == pytest.approx((2 * half) ** 2 - water.area)
    assert land.contains(box(-5, -5, 5, 5).centroid)       # the island
    assert land.contains(box(320, 0, 321, 1).centroid)     # beyond the map


def test_world_size_reaches_the_water_plane_and_keeps_tile_size():
    sim = SailboatSimulation(world_half_size=500.0)
    water = sim.model.geom("water")
    assert water.size[0] == pytest.approx(500.0)
    mat = sim.model.mat("water_mat")
    assert mat.texrepeat[0] == pytest.approx(2 * 500.0 / WATER_TILE_M)


def test_default_world_is_unchanged():
    sim = SailboatSimulation()
    assert sim.model.geom("water").size[0] == pytest.approx(WORLD_HALF_SIZE)
    assert sim.model.mat("water_mat").texrepeat[0] == pytest.approx(50.0)


def test_no_land_triangle_reaches_into_the_water_on_a_real_coastline():
    # An unconstrained Delaunay filtered by centroid spills about 4 m² of land
    # into the water on this map: invisible rocks once they become colliders.
    pytest.importorskip("graph_route_planner")
    from graph_route_planner.map_loader import load_map

    water = load_map("kornaten").water
    land = land_from_water(water, world_half_size_for(water))
    tris = unary_union([t for piece in land for t in _triangulate(piece)])
    assert tris.intersection(water).area < 1e-6
