"""Turning a body of water plus two points into a sailed route."""

import logging
import math
import time
from dataclasses import dataclass

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from graph_route_planner.geometry import bbox_diagonal, component_containing
from graph_route_planner.graph import (
    build_graph,
    dijkstra,
    extract_nodes,
    merge_nearby_nodes,
    navigable_area,
    prune_graph,
)
from graph_route_planner.sailing import SailingModel

log = logging.getLogger(__name__)

Position = tuple[float, float]
#: Adjacency: node -> {neighbour: cost in time units}.
Graph = dict[Position, dict[Position, float]]


class NoWaterError(Exception):
    """Raised when start and goal cannot be connected by water at all.

    This is a statement about the map, not a planner failure: the points may
    be on land, on different bodies of water, or too close to shore for the
    configured margin.
    """


@dataclass(frozen=True)
class PlannerConfig:
    """Tuning for a planning run.

    Args:
        margin: Clearance kept from shore and islands, in metres.
        grid_spacing: Spacing of the open-water waypoint grid, in metres.
        merge_threshold: Nodes closer than this are merged; 0 disables.
        enable_pruning: Drop edges that lead away from the goal.
        max_leg_distance: Longest leg considered, in metres, or None to derive
            it from the map — the navigable area's bounding-box diagonal, which
            is the smallest limit that never truncates a legal leg.

            Skipping long pairs is an optimisation: it avoids O(n^2) visibility
            checks. But a fixed limit is a scale assumption, and setting it
            below the map's longest crossing silently changes the answer — a
            leg the boat could sail in one go is never offered, so the route
            detours through a waypoint it does not need. Prefer None unless you
            are deliberately trading accuracy for speed on a large map.
        min_leg_distance: Shortest leg considered, in metres. 0 disables.
            Suppresses fiddly hops the boat cannot usefully sail. Raising it
            can disconnect the graph and leave no route at all — including
            between a start and goal closer together than this.

    Raises:
        ValueError: If the leg bounds are negative or leave no legal range.
    """

    margin: float = 5.0
    grid_spacing: float = 15.0
    merge_threshold: float = 3.0
    enable_pruning: bool = True
    max_leg_distance: float | None = None
    min_leg_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.min_leg_distance < 0.0:
            raise ValueError(f"min_leg_distance must be >= 0, got {self.min_leg_distance}")
        # A derived maximum can only be checked once the map is known; see
        # resolve_max_leg.
        if (self.max_leg_distance is not None
                and self.min_leg_distance >= self.max_leg_distance):
            raise ValueError(
                f"min_leg_distance ({self.min_leg_distance}) must be below "
                f"max_leg_distance ({self.max_leg_distance}), or no leg is legal"
            )


def resolve_max_leg(config: PlannerConfig, navigable) -> float:
    """Settle the longest-leg limit for one planning run.

    An explicit ``max_leg_distance`` always wins. Otherwise it is derived from
    the navigable area, so the limit scales with the map instead of assuming
    one.

    Args:
        config: The run's tuning.
        navigable: The area being planned over.

    Returns:
        The limit in metres.

    Raises:
        ValueError: If the derived limit leaves no legal leg range.
    """
    if config.max_leg_distance is not None:
        return config.max_leg_distance

    derived = bbox_diagonal(navigable)
    if config.min_leg_distance >= derived:
        raise ValueError(
            f"min_leg_distance ({config.min_leg_distance}) exceeds the map's own "
            f"span ({derived:.1f} m), so no leg is legal"
        )
    return derived


@dataclass(frozen=True)
class Route:
    """The outcome of a planning run.

    "No route" is an empty path, not None. A failed search still carries the
    ``navigable`` area and the ``graph`` it searched — the viewer draws both
    either way — so returning None instead would throw that away. Keeping every
    field concrete also spares callers from narrowing an Optional just to take
    a length.

    Args:
        waypoints: The sailed path. Empty if no route exists.
        duration: Total time in the model's units. Infinite if no route exists.
        navigable: The eroded water the search actually ran on.
        graph: The visibility graph that was searched.
    """

    waypoints: list[Position]
    duration: float
    navigable: BaseGeometry
    graph: Graph

    @property
    def found(self) -> bool:
        """Whether a route was found."""
        return bool(self.waypoints)


def select_navigable(water, start: Position, goal: Position, margin: float):
    """Reduce a whole map to the single piece of water the boat can use.

    Two steps, either of which can legitimately fail:

    1. Pick the water body containing ``start``. Anything else on the map —
       another lake, or a lake on an island — is unreachable without a
       portage, so the goal must lie in the same body.
    2. Erode that body by ``margin``. Eroding can split it, when a channel
       narrower than twice the margin closes, or push start/goal into the
       excluded strip near the shore, so both are re-checked afterwards.

    Args:
        water: The map's navigable geometry.
        start: Start position, ``(east, north)`` in metres.
        goal: Goal position, ``(east, north)`` in metres.
        margin: Shore clearance in metres.

    Returns:
        The single eroded polygon containing both start and goal.

    Raises:
        NoWaterError: If no such polygon exists.
    """
    body = component_containing(water, start)
    if body is None:
        raise NoWaterError(f"start {start} is not on water")
    if not body.covers(Point(goal)):
        other = component_containing(water, goal)
        raise NoWaterError(
            f"goal {goal} is on a different body of water than start {start}"
            if other is not None else f"goal {goal} is not on water"
        )

    eroded = navigable_area(body, margin)
    if eroded.is_empty:
        raise NoWaterError(f"margin {margin} leaves no navigable water at all")

    navigable = component_containing(eroded, start)
    if navigable is None:
        raise NoWaterError(f"start {start} is within {margin} of shore — reduce the margin")
    if not navigable.covers(Point(goal)):
        raise NoWaterError(
            f"goal {goal} is unreachable with margin {margin} "
            f"(shore clearance closes the route) — reduce the margin"
        )
    return navigable


def plan_route(water, start: Position, goal: Position,
               model: SailingModel | None = None,
               config: PlannerConfig | None = None) -> Route:
    """Plan a sailing route between two points on a map.

    Args:
        water: Navigable geometry, e.g. ``SailMap.water`` or ``Scenario.water``.
        start: Start position, ``(east, north)`` in metres.
        goal: Goal position, ``(east, north)`` in metres.
        model: The boat's sailing model. Defaults to a northerly wind.
        config: Tuning. Defaults to ``PlannerConfig()``, whose leg limit is
            derived from the map.

    Returns:
        A Route; check ``.found`` before using ``.waypoints``.

    Raises:
        NoWaterError: If start and goal cannot be connected by water.
        ValueError: If the leg bounds leave no legal range for this map.
    """
    model = model or SailingModel()
    config = config or PlannerConfig()

    navigable = select_navigable(water, start, goal, config.margin)
    max_leg = resolve_max_leg(config, navigable)
    nodes = extract_nodes(start, goal, navigable, config.grid_spacing)

    t0 = time.perf_counter()
    if config.merge_threshold > 0:
        nodes = merge_nearby_nodes(nodes, navigable, config.merge_threshold)
    t_merge = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph = build_graph(nodes, navigable, model, max_leg, config.min_leg_distance)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    if config.enable_pruning:
        graph = prune_graph(graph, start, goal)
    t_prune = time.perf_counter() - t0

    t0 = time.perf_counter()
    found_path, cost = dijkstra(graph, start, goal)
    t_search = time.perf_counter() - t0
    waypoints = found_path or []
    duration = cost if cost is not None else math.inf

    log.debug("merge %.3fs | build %.3fs | prune %.3fs | search %.3fs | total %.3fs",
              t_merge, t_build, t_prune, t_search,
              t_merge + t_build + t_prune + t_search)
    log.info("%d nodes, %d edges -> %s",
             len(graph), sum(len(v) for v in graph.values()) // 2,
             f"{len(waypoints)} waypoints" if waypoints else "no route")

    return Route(waypoints=waypoints, duration=duration, navigable=navigable, graph=graph)
