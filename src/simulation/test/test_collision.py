"""
Tests for running aground: the boat must not pass through land, and the
simulation must report the contact while it lasts.

These run physics only (no rendering), with no wind so the only motion is the
initial push.
"""

import numpy as np
from shapely.geometry import box

from simulation.sailboat_simulation import SailboatSimulation

# Distance from the boat origin to the tip of its collider (capsule end + radius).
BOW_REACH = 0.95 + 0.3


def push(sim, speed, heading_rad):
    """Point the boat along heading_rad (0 = East) and give it that velocity."""
    sim.data.qpos[sim.qadr_yaw] = heading_rad
    sim.data.qvel[sim.vadr_x] = speed * np.cos(heading_rad)
    sim.data.qvel[sim.vadr_y] = speed * np.sin(heading_rad)


def run(sim, seconds):
    """Step for `seconds`; return (max east reached, whether any contact occurred)."""
    max_x, touched = -np.inf, False
    for _ in range(int(seconds / sim.model.opt.timestep)):
        sim.step()
        max_x = max(max_x, sim.data.qpos[sim.qadr_x])
        touched |= sim.in_collision()
    return max_x, touched


def make_sim(obstacles):
    sim = SailboatSimulation(obstacles=obstacles, wind_speed=0.0)
    sim.reset()
    return sim


def test_without_land_the_boat_passes_freely():
    sim = make_sim([])
    push(sim, 3.0, 0.0)
    max_x, touched = run(sim, 5.0)
    assert max_x > 8.0
    assert not touched


def test_boat_cannot_sail_through_land():
    wall_x = 5.0
    sim = make_sim([box(wall_x, -10, wall_x + 10, 10)])
    push(sim, 3.0, 0.0)
    max_x, touched = run(sim, 5.0)
    assert touched
    # The bow may sink a few centimetres into the bank, never through it.
    assert max_x + BOW_REACH < wall_x + 0.05
    # And it is not still driving into the wall afterwards.
    assert sim.data.qvel[sim.vadr_x] < 0.1


def test_collision_is_reported_only_while_touching():
    sim = make_sim([box(5, -10, 15, 10)])
    assert not sim.in_collision()
    assert not sim.get_status()['collision']

    push(sim, 3.0, 0.0)
    for _ in range(2000):
        sim.step()
        if sim.in_collision():
            break
    assert sim.in_collision()
    assert sim.get_status()['collision']

    # Pull the boat well clear: the contact must end with it.
    sim.set_boat_position(-20.0, 0.0)
    sim.step()
    assert not sim.in_collision()


def test_oblique_grounding_stops_the_boat_at_the_shore():
    # Shore to the north (y >= 5), boat heading 40° off it.
    sim = make_sim([box(-50, 5, 80, 12)])
    push(sim, 3.0, np.radians(40))
    _, touched = run(sim, 4.0)
    assert touched
    y = sim.data.qpos[sim.qadr_y]
    assert y + BOW_REACH * np.sin(np.radians(40)) < 5.0 + 0.3


def test_buoys_are_not_obstacles():
    # Only land is solid; a buoy straight ahead is sailed through as before.
    sim = SailboatSimulation(buoys=[(4.0, 0.0)], wind_speed=0.0)
    sim.reset()
    push(sim, 3.0, 0.0)
    max_x, touched = run(sim, 5.0)
    assert max_x > 8.0
    assert not touched
