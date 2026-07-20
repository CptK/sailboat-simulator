"""Shared fixtures for the route_follower tests.

`_step` and `_on_target_route` are pure logic over the route, the boat and the
wind — no ROS spinning needed — so these drive them on a stand-in carrying the
attributes they read. That keeps the steering under test without a live graph.
"""

import types

import pytest

from route_follower.node import RouteFollower
from route_follower.waypoint import WayPoint


@pytest.fixture
def config():
    return types.SimpleNamespace(
        step_interval=0.2,
        dist_threshold=5.0,
        maneuver_threshold=70.0,
        safe_jibe_threshold=15.0,
        maneuver_heading_step_size=10.0,
        command_step_size_multiplier=1.5,
        heading_tolerance=2.0,
        max_steps_to_cross_wind_when_tacking=40,
    )


@pytest.fixture
def follower(config):
    """Build a stand-in follower. `logged` collects what it would have logged."""

    def build(route=(), target_index=0, boat=None, heading=0.0,
              wind_direction=0.1, wind_speed=4.0, maneuver=None):
        node = types.SimpleNamespace(
            route=list(route), target_index=target_index, maneuver=maneuver,
            completed=False, config=config, location=boat, heading=heading,
            wind_direction=wind_direction, wind_speed=wind_speed,
        )
        node.logged = []
        node.published_completed = []
        node.published_route = []
        node.get_logger = lambda: types.SimpleNamespace(
            info=node.logged.append, warn=node.logged.append, error=node.logged.append
        )
        node._publisher_route_completed = types.SimpleNamespace(
            publish=lambda m: node.published_completed.append(m.data)
        )
        node._publisher_current_route = types.SimpleNamespace(
            publish=node.published_route.append
        )
        # Bind the real methods so the object behaves like the node.
        for name in ("_select_maneuver", "_prospective_target",
                     "_skip_reached_soft_waypoints"):
            setattr(node, name,
                    (lambda n=name: lambda *a, **k: getattr(RouteFollower, n)(node, *a, **k))())
        return node

    return build


def step(node):
    """Run one steering tick and return the commanded heading."""
    return RouteFollower._step(node, node.location, node.heading,
                               node.wind_direction, node.wind_speed)


class RouteMsgStub:
    """Stands in for boat_msgs Route without needing the generated type."""

    def __init__(self, points):
        self.waypoints = [
            types.SimpleNamespace(east=e, north=n, is_soft=soft)
            if len(p) == 3 else types.SimpleNamespace(east=p[0], north=p[1], is_soft=False)
            for p in points
            for e, n, soft in [(p[0], p[1], p[2] if len(p) == 3 else False)]
        ]


def wp(east, north, is_soft=False):
    return WayPoint(east, north, is_soft=is_soft)
