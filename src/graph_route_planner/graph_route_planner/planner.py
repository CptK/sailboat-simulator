"""Turning a body of water plus two points into a sailed route."""

import logging
import math
import time
from dataclasses import dataclass

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from graph_route_planner.geometry import bbox_diagonal, component_containing
from graph_route_planner.graph import (
    build_graph,
    cost_pairs,
    dijkstra,
    extract_grid_nodes,
    extract_nodes,
    merge_nearby_nodes,
    navigable_area,
    prune_graph,
    visible_from,
    visible_pairs,
)
from graph_route_planner.sailing import SailingModel

log = logging.getLogger(__name__)

Position = tuple[float, float]
#: Adjacency: node -> {neighbour: cost in time units}.
Graph = dict[Position, dict[Position, float]]


def true_duration(path, model: SailingModel) -> float:
    """The boat's own sailing time along `path`, with no shaping cost in it.

    Dijkstra's total includes `segment_cost`, which is a search bias rather than
    anything the boat experiences. Callers compare durations against each other
    and against the clock, so the reported figure has to be the real one.

    Args:
        path: The waypoints, or None/empty if no route was found.
        model: The boat's sailing model.

    Returns:
        Total time in the model's units; infinite if there is no route.
    """
    if not path:
        return math.inf
    return sum(model.sailing_time(u, v) for u, v in zip(path, path[1:]))


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
        segment_cost: A fixed charge, in time units, added to every leg.

            Time alone prices an 8 m repositioning hop at 8 m of sailing and
            nothing else, so the search takes one to gain a fraction of a
            percent — producing routes that reach a mark via a stub and a sharp
            turn. Every leg boundary is a turn, and a turn is where the boat
            stops making progress while it settles, which time-only costing has
            no way to express. This charge is that missing term.

            It is a shaping cost, not a physical one: it steers the search but
            never appears in `Route.duration`. It cannot make an unsailable
            heading sailable, so a beat still tacks however high it is set. Set
            it to 0 to recover pure time-optimal routing.

    Raises:
        ValueError: If the leg bounds are negative or leave no legal range, or
            if segment_cost is negative.
    """

    margin: float = 5.0
    grid_spacing: float = 15.0
    merge_threshold: float = 3.0
    enable_pruning: bool = True
    max_leg_distance: float | None = None
    min_leg_distance: float = 0.0
    segment_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.min_leg_distance < 0.0:
            raise ValueError(f"min_leg_distance must be >= 0, got {self.min_leg_distance}")
        if self.segment_cost < 0.0:
            raise ValueError(f"segment_cost must be >= 0, got {self.segment_cost}")
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


class CachedPlanner:
    """`plan_route` with the wind-independent geometry kept between calls.

    Planning the same water twice rebuilds the same visibility graph twice, and
    that graph is ~96% of the work. It depends on the map, the margin, the grid
    spacing and the leg bounds — not on the wind, and not on where the boat is.
    So it is built once and then:

      * a wind change re-costs the cached pairs, which is arithmetic, not
        geometry. The last few winds are memoised, so a steady wind costs
        nothing at all after the first plan.
      * a new start or goal is attached with one O(n) visibility pass each,
        against the O(n^2) pass that built the cache.

    The cache is rebuilt only when it cannot serve the query: a different body
    of water, or a start or goal the eroded area does not cover. Configuration
    is fixed for the planner's lifetime, so in practice that means once.

    Args:
        water: Navigable geometry, e.g. ``SailMap.water``.
        config: Tuning. Defaults to ``PlannerConfig()``.
    """

    #: How many winds to keep costed graphs for. Small on purpose — the graph is
    #: large, and a wind that drifts back and forth only needs the last few.
    MEMO_SIZE = 4

    def __init__(self, water, config: PlannerConfig | None = None) -> None:
        self.water = water
        self.config = config or PlannerConfig()
        self._navigable = None
        self._inside = None
        self._nodes: list[Position] = []
        self._pairs: list[tuple[int, int]] = []
        self._max_leg = 0.0
        self._costed: dict = {}
        self.builds = 0          # how many times the geometry was rebuilt

    def _usable(self, start: Position, goal: Position) -> bool:
        """Whether the cached area can serve this query."""
        if self._navigable is None:
            return False
        return (self._inside.covers(Point(start))
                and self._inside.covers(Point(goal)))

    def _rebuild(self, start: Position, goal: Position) -> None:
        """Build the wind-independent half for the water containing `start`."""
        cfg = self.config
        self._navigable = select_navigable(self.water, start, goal, cfg.margin)
        self._inside = prep(self._navigable)
        self._max_leg = resolve_max_leg(cfg, self._navigable)

        nodes = extract_grid_nodes(self._navigable, cfg.grid_spacing)
        if cfg.merge_threshold > 0:
            # No start or goal in this set, so nothing is protected from merging.
            nodes = merge_nearby_nodes(nodes, self._navigable, cfg.merge_threshold,
                                       protected=0)
        self._nodes = nodes
        self._pairs = visible_pairs(nodes, self._navigable, self._max_leg,
                                    cfg.min_leg_distance)
        self._costed = {}
        self.builds += 1
        log.info("visibility graph built: %d nodes, %d pairs (build #%d)",
                 len(self._nodes), len(self._pairs), self.builds)

    def _base_graph(self, model: SailingModel) -> dict:
        """The cached graph costed for `model`, memoised per wind."""
        key = (model.wind_from, model.no_go_deg, self.config.segment_cost)
        graph = self._costed.get(key)
        if graph is None:
            graph = cost_pairs(self._nodes, self._pairs, model, self.config.segment_cost)
            if len(self._costed) >= self.MEMO_SIZE:
                self._costed.pop(next(iter(self._costed)))
            self._costed[key] = graph
        return graph

    def _attach(self, graph: dict, point: Position, model: SailingModel) -> None:
        """Add one query point and its edges to `graph`, in place."""
        charge = self.config.segment_cost
        graph.setdefault(point, {})
        for i in visible_from(point, self._nodes, self._navigable, self._max_leg,
                              self.config.min_leg_distance):
            node = self._nodes[i]
            # The same charge as the cached edges. Penalising only those would
            # leave the route a free stub at either end, which is the failure
            # this whole term exists to remove.
            if model.heading_is_sailable(point, node):
                graph[point][node] = model.sailing_time(point, node) + charge
            if model.heading_is_sailable(node, point):
                graph.setdefault(node, {})[point] = model.sailing_time(node, point) + charge

    def plan(self, start: Position, goal: Position,
             model: SailingModel | None = None) -> Route:
        """Plan a route, reusing the cached geometry wherever possible.

        Args:
            start: Start position, ``(east, north)`` in metres.
            goal: Goal position, ``(east, north)`` in metres.
            model: The boat's sailing model. Defaults to a northerly wind.

        Returns:
            A Route; check ``.found`` before using ``.waypoints``.

        Raises:
            NoWaterError: If start and goal cannot be connected by water.
            ValueError: If the leg bounds leave no legal range for this map.
        """
        model = model or SailingModel()
        if not self._usable(start, goal):
            self._rebuild(start, goal)

        t0 = time.perf_counter()
        base = self._base_graph(model)
        t_cost = time.perf_counter() - t0

        # Copy before splicing: the cached graph is shared with the next plan,
        # and attaching endpoints writes into its neighbours' adjacency.
        t0 = time.perf_counter()
        graph = {node: dict(edges) for node, edges in base.items()}
        self._attach(graph, start, model)
        self._attach(graph, goal, model)

        # The direct leg. Neither endpoint is in the cached node set, so this
        # pair is in neither attachment pass — and it is very often the answer.
        leg = math.dist(start, goal)
        if (self.config.min_leg_distance <= leg <= self._max_leg
                and self._inside.covers(LineString([start, goal]))):
            if model.heading_is_sailable(start, goal):
                graph[start][goal] = model.sailing_time(start, goal) + self.config.segment_cost
            if model.heading_is_sailable(goal, start):
                graph[goal][start] = model.sailing_time(goal, start) + self.config.segment_cost
        t_attach = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.config.enable_pruning:
            graph = prune_graph(graph, start, goal)
        found_path, cost = dijkstra(graph, start, goal)
        t_search = time.perf_counter() - t0

        log.debug("cost %.3fs | attach %.3fs | prune+search %.3fs | total %.3fs",
                  t_cost, t_attach, t_search, t_cost + t_attach + t_search)
        return Route(waypoints=found_path or [],
                     duration=true_duration(found_path, model),
                     navigable=self._navigable, graph=graph)


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
    graph = build_graph(nodes, navigable, model, max_leg, config.min_leg_distance,
                        config.segment_cost)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    if config.enable_pruning:
        graph = prune_graph(graph, start, goal)
    t_prune = time.perf_counter() - t0

    t0 = time.perf_counter()
    found_path, _ = dijkstra(graph, start, goal)
    t_search = time.perf_counter() - t0
    waypoints = found_path or []
    duration = true_duration(found_path, model)

    log.debug("merge %.3fs | build %.3fs | prune %.3fs | search %.3fs | total %.3fs",
              t_merge, t_build, t_prune, t_search,
              t_merge + t_build + t_prune + t_search)
    log.info("%d nodes, %d edges -> %s",
             len(graph), sum(len(v) for v in graph.values()) // 2,
             f"{len(waypoints)} waypoints" if waypoints else "no route")

    return Route(waypoints=waypoints, duration=duration, navigable=navigable, graph=graph)
