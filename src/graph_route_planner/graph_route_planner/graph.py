"""Building and searching the visibility graph over navigable water."""

import heapq
import logging
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from graph_route_planner.geometry import polygons_of
from graph_route_planner.sailing import SailingModel

log = logging.getLogger(__name__)


def navigable_area(water, margin: float):
    """
    Shrink the water by `margin` to keep the boat off everything solid.

    A negative buffer erodes each water body away from its shoreline *and*
    away from every island hole in one operation, which is why obstacles no
    longer need inflating separately. Eroding can split a body in two (a
    narrow channel closes up) or erase it entirely (a pond narrower than the
    margin) — both are correct answers, not errors.
    """
    return water.buffer(-margin)


def extract_nodes(start, goal, navigable, grid_spacing: float = 15.0) -> list:
    """
    Graph nodes:
      - start and goal
      - every vertex of the navigable area  (shoreline and island corners)
      - a regular grid of open-water points (for tacking waypoints)

    Grid points outside the navigable area are dropped. The grid is sized to
    the area itself, so it adapts to whatever map was loaded.
    """
    nodes = [start, goal]

    for poly in polygons_of(navigable):
        nodes.extend(tuple(c) for c in list(poly.exterior.coords)[:-1])
        for hole in poly.interiors:
            nodes.extend(tuple(c) for c in list(hole.coords)[:-1])

    if navigable.is_empty:
        return nodes

    inside = prep(navigable)
    x0, y0, x1, y1 = navigable.bounds
    x = x0
    while x <= x1:
        y = y0
        while y <= y1:
            if inside.covers(Point(x, y)):
                nodes.append((x, y))
            y += grid_spacing
        x += grid_spacing

    return nodes


def segment_is_clear(p1, p2, navigable) -> bool:
    """True iff the whole leg stays inside the navigable area."""
    return navigable.covers(LineString([p1, p2]))


def merge_nearby_nodes(nodes: list, navigable, threshold: float = 3.0,
                       protected: int = 2) -> list:
    """
    Merge nodes that are closer than `threshold` units apart.

    Strategy:
      - Build clusters of mutually-close nodes
      - Replace each cluster with a single representative node
      - Choose the representative as the cluster centroid, but verify it's
        still inside the navigable area (if not, use the closest valid node)

    This reduces graph size significantly when grid nodes cluster near the
    shoreline or when a hand-drawn outline has many closely-spaced vertices.
    """
    if threshold <= 0:
        return nodes

    inside = prep(navigable)

    # The first `protected` nodes are never merged. With the default of 2 that
    # is start and goal; a cached node set has neither, and passes 0.
    keep = nodes[:protected]
    other_nodes = nodes[protected:]
    
    # Build clusters using simple greedy approach
    used = [False] * len(other_nodes)
    clusters = []
    
    for i, node in enumerate(other_nodes):
        if used[i]:
            continue
        cluster = [node]
        used[i] = True
        
        # Find all nodes within threshold of any node in cluster
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(other_nodes):
                if used[j]:
                    continue
                if any(math.dist(other, c) < threshold for c in cluster):
                    cluster.append(other)
                    used[j] = True
                    changed = True
        
        clusters.append(cluster)
    
    # Replace each cluster with a single representative
    merged = list(keep)
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        
        # Try centroid first
        cx = sum(n[0] for n in cluster) / len(cluster)
        cy = sum(n[1] for n in cluster) / len(cluster)
        centroid = (cx, cy)

        if inside.covers(Point(centroid)):
            merged.append(centroid)
        else:
            # Centroid drifted onto land (e.g. across a narrow inlet) — fall
            # back to the first cluster member that is still navigable.
            for node in cluster:
                if inside.covers(Point(node)):
                    merged.append(node)
                    break
    
    log.debug("merged %d nodes -> %d nodes (threshold = %s)",
              len(nodes), len(merged), threshold)
    return merged


def extract_grid_nodes(navigable, grid_spacing: float = 15.0) -> list:
    """The map's own nodes: shoreline and island corners, plus an open-water grid.

    Identical to `extract_nodes` minus the start and goal, so the result depends
    only on the map and the spacing — which is what makes it worth caching.

    Args:
        navigable: The area to cover.
        grid_spacing: Spacing of the open-water grid, in metres.

    Returns:
        Nodes as ``(east, north)`` tuples.
    """
    nodes: list = []

    for poly in polygons_of(navigable):
        nodes.extend(tuple(c) for c in list(poly.exterior.coords)[:-1])
        for hole in poly.interiors:
            nodes.extend(tuple(c) for c in list(hole.coords)[:-1])

    if navigable.is_empty:
        return nodes

    inside = prep(navigable)
    x0, y0, x1, y1 = navigable.bounds
    x = x0
    while x <= x1:
        y = y0
        while y <= y1:
            if inside.covers(Point(x, y)):
                nodes.append((x, y))
            y += grid_spacing
        x += grid_spacing

    return nodes


#: Candidate pairs tested per block. Bounds peak memory: the whole upper
#: triangle of a 6000-node graph is 18M pairs, which is worth chunking.
_PAIR_BLOCK = 2_000_000


def visible_pairs(nodes: list, navigable, max_leg_distance: float,
                  min_leg_distance: float = 0.0, inside=None) -> list:
    """Which node pairs are within the leg bounds and clear of land.

    This is the expensive half of building a graph — O(n^2) `covers` calls — and
    it depends only on the map and the leg bounds. Not on the wind, and not on
    where the boat happens to be. Everything else is cheap arithmetic over the
    pairs this returns, which is why this is the half worth keeping.

    Args:
        nodes: Candidate waypoints.
        navigable: The area legs must stay inside.
        max_leg_distance: Longest leg to consider, in metres.
        min_leg_distance: Shortest leg to consider, in metres. 0 disables.
        inside: A prepared `navigable`, if the caller already has one.

    Returns:
        Index pairs ``(i, j)`` with ``i < j``. Indices rather than coordinates,
        because on a fine grid there are hundreds of thousands of them.
    """
    arr = np.asarray(nodes, dtype=float)
    count = len(arr)
    if count < 2:
        return []

    shapely.prepare(navigable)
    kept_i, kept_j = [], []
    rows = max(1, _PAIR_BLOCK // count)

    for lo in range(0, count, rows):
        hi = min(lo + rows, count)
        i, j = np.meshgrid(np.arange(lo, hi), np.arange(count), indexing="ij")
        upper = j > i                       # each pair once, and never a self-pair
        i, j = i[upper], j[upper]
        if i.size == 0:
            continue

        span = np.hypot(arr[j, 0] - arr[i, 0], arr[j, 1] - arr[i, 1])
        within = (span <= max_leg_distance) & (span >= min_leg_distance)
        i, j = i[within], j[within]
        if i.size == 0:
            continue

        clear = shapely.covers(navigable, _segments(arr[i], arr[j]))
        kept_i.append(i[clear])
        kept_j.append(j[clear])

    if not kept_i:
        return []
    i = np.concatenate(kept_i)
    j = np.concatenate(kept_j)
    log.debug("visibility: %d pairs over %d nodes", i.size, count)
    return list(zip(i.tolist(), j.tolist()))


def _segments(starts, ends):
    """Build one LineString per row of `starts`/`ends`, in one call.

    Args:
        starts: Array of shape (n, 2).
        ends: Array of shape (n, 2).

    Returns:
        An array of n LineStrings.
    """
    coords = np.empty((len(starts) * 2, 2), dtype=float)
    coords[0::2] = starts
    coords[1::2] = ends
    return shapely.linestrings(coords, indices=np.repeat(np.arange(len(starts)), 2))


def visible_from(point, nodes: list, navigable, max_leg_distance: float,
                 min_leg_distance: float = 0.0, inside=None) -> list:
    """Which nodes one extra point can see. O(n), not O(n^2).

    Attaching the boat and its goal to a cached graph costs one pass each,
    against the n^2 pass that produced the cache.

    Args:
        point: The point to attach.
        nodes: The cached node set.
        navigable: The area legs must stay inside.
        max_leg_distance: Longest leg to consider, in metres.
        min_leg_distance: Shortest leg to consider, in metres.
        inside: A prepared `navigable`, if the caller already has one.

    Returns:
        Indices into `nodes` that `point` can reach directly.
    """
    arr = np.asarray(nodes, dtype=float)
    if len(arr) == 0:
        return []

    origin = np.asarray(point, dtype=float)
    span = np.hypot(arr[:, 0] - origin[0], arr[:, 1] - origin[1])
    index = np.flatnonzero((span >= min_leg_distance) & (span <= max_leg_distance))
    if index.size == 0:
        return []

    shapely.prepare(navigable)
    clear = shapely.covers(
        navigable, _segments(np.broadcast_to(origin, (index.size, 2)), arr[index])
    )
    return index[clear].tolist()


def cost_pairs(nodes: list, pairs: list, model: SailingModel,
               segment_cost: float = 0.0) -> dict:
    """Turn visible pairs into a directed adjacency costed for one wind.

    A pair is undirected but an edge is not: a leg may be sailable one way and
    inside the no-go zone the other, so each direction is tested separately.

    Args:
        nodes: The node set the pair indices refer to.
        pairs: Index pairs from `visible_pairs`.
        model: The boat's sailing model for the wind in question.
        segment_cost: A fixed charge added to every edge, so a route pays it
            once per leg. See `PlannerConfig.segment_cost`.

    Returns:
        ``{node: {neighbour: cost}}`` in time units plus the per-segment charge.
        The charge shapes the search only — `Route.duration` reports the boat's
        true sailing time, with no penalty in it.
    """
    graph: dict = {n: {} for n in nodes}
    if not pairs:
        return graph

    arr = np.asarray(nodes, dtype=float)
    index = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    starts, ends = arr[index[:, 0]], arr[index[:, 1]]

    # Every edge's angle, sailability and cost in four array passes rather than
    # four Python calls per pair. The two directions are separate edges: a leg
    # can be sailable one way and inside the no-go zone the other.
    out_ok = model.headings_are_sailable(starts, ends)
    out_cost = model.sailing_times(starts, ends) + segment_cost
    back_ok = model.headings_are_sailable(ends, starts)
    back_cost = model.sailing_times(ends, starts) + segment_cost

    # Only the dict assembly stays in Python; the adjacency has to be built.
    for k, (i, j) in enumerate(pairs):
        u, v = nodes[i], nodes[j]
        if out_ok[k]:
            graph[u][v] = out_cost[k]
        if back_ok[k]:
            graph[v][u] = back_cost[k]
    return graph


def build_graph(nodes: list, navigable, model: SailingModel,
                max_leg_distance: float = 60.0,
                min_leg_distance: float = 0.0,
                segment_cost: float = 0.0) -> dict:
    """Build the directed visibility graph over the navigable area.

    An edge u->v exists only if its length is within the leg bounds, the leg
    stays inside the navigable area, and its heading is outside the no-go
    zone. The area is prepared once, since covers() dominates this loop.

    Args:
        nodes: Candidate waypoints as ``(east, north)`` tuples.
        navigable: The area legs must stay inside.
        model: The boat's sailing model, supplying no-go zone and leg costs.
        max_leg_distance: Longest leg to consider, in metres. Skipping long
            pairs cuts the O(n^2) checks sharply, but a leg the boat could
            sail directly is then never offered — see PlannerConfig.
        min_leg_distance: Shortest leg to consider, in metres. 0 disables.

    Returns:
        ``{node: {neighbour: cost}}`` with cost in time units.
    """
    pairs = visible_pairs(nodes, navigable, max_leg_distance, min_leg_distance)
    log.debug("visibility: %d pairs kept (leg bounds %s..%s)",
              len(pairs), min_leg_distance, max_leg_distance)
    return cost_pairs(nodes, pairs, model, segment_cost)


def prune_topology(graph: dict, start: tuple, goal: tuple) -> dict:
    """
    Remove nodes based on graph topology:
      1. Dead-end nodes (degree 1): only one edge total → always a detour
      2. Self-cycle nodes: all edges to/from a single neighbor → A→B→A detour
    
    Iterate until no more nodes can be removed (some removals expose new dead ends).
    Never remove start or goal.
    """
    nodes_removed = 0
    changed = True
    
    while changed:
        changed = False
        to_remove = []
        
        for node, neighbors in graph.items():
            if node in (start, goal):
                continue
            
            # Count total degree (in + out)
            out_degree = len(neighbors)
            in_degree = sum(1 for n in graph if node in graph[n])
            total_degree = out_degree + in_degree
            
            # Rule 1: Dead end (degree 1)
            if total_degree == 1:
                to_remove.append(node)
                changed = True
                continue
            
            # Rule 2: All edges to/from a single neighbor
            # Collect all neighbors (both directions)
            all_neighbors = set(neighbors.keys())
            for n in graph:
                if node in graph[n]:
                    all_neighbors.add(n)
            
            if len(all_neighbors) == 1:
                # Only connects to one other node → A↔B cycle, remove B
                to_remove.append(node)
                changed = True
        
        # Remove nodes and their edges
        for node in to_remove:
            nodes_removed += 1
            # Remove outgoing edges
            del graph[node]
            # Remove incoming edges
            for n in graph:
                graph[n].pop(node, None)
    
    log.debug("topological pruning: removed %d dead-end/cycle nodes", nodes_removed)
    return graph


def prune_graph(graph: dict, start: tuple, goal: tuple) -> dict:
    """
    Prune edges that are unlikely to be on the optimal path:
      1. Edges that move away from the goal (negative progress).
      2. Edges from nodes that are farther from goal than start is.
      3. Edges to nodes behind the start line (perpendicular to start→goal).

    This reduces graph size significantly for large search spaces while
    preserving the optimal path (or a near-optimal one).
    """
    gx, gy = goal[0] - start[0], goal[1] - start[1]
    goal_dist = math.hypot(gx, gy)
    if goal_dist < 1e-9:
        return graph
    
    # Goal direction unit vector
    gdx, gdy = gx / goal_dist, gy / goal_dist
    
    # Distance from start along the goal direction (signed)
    def progress(node):
        nx, ny = node[0] - start[0], node[1] - start[1]
        return nx * gdx + ny * gdy
    
    dist_to_goal = {n: math.dist(n, goal) for n in graph}
    
    pruned = {n: {} for n in graph}
    edges_removed = 0
    
    for u, neighbors in graph.items():
        u_dist = dist_to_goal[u]
        u_prog = progress(u)
        
        # Skip nodes that are farther from goal than start is
        # (unless they're the start itself)
        if u != start and u_dist > goal_dist * 1.2:
            edges_removed += len(neighbors)
            continue
        
        for v, cost in neighbors.items():
            v_dist = dist_to_goal[v]
            v_prog = progress(v)
            
            # Rule 1: Edge moves toward goal (reduces distance)
            moves_toward_goal = v_dist < u_dist
            
            # Rule 2: Edge makes forward progress along start→goal axis
            edge_progress = v_prog - u_prog
            makes_progress = edge_progress > -goal_dist * 0.1  # allow small backtrack
            
            # Rule 3: Destination is not behind the start line
            behind_start = v_prog < -goal_dist * 0.15
            
            if moves_toward_goal or (makes_progress and not behind_start):
                pruned[u][v] = cost
            else:
                edges_removed += 1
    
    log.debug("pruned %d edges", edges_removed)
    return pruned


# ── Dijkstra ───────────────────────────────────────────────────────────────────

def dijkstra(
    graph: dict, start: tuple[float, float], goal: tuple[float, float]
) -> tuple[list[tuple[float, float]] | None, float | None]:
    inf  = float("inf")
    dist = {n: inf for n in graph}
    prev: dict[tuple[float, float], tuple[float, float] | None] = {n: None for n in graph}
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist[u]:
            continue
        for v, w in graph[u].items():
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    if dist[goal] == inf:
        return None, None

    path, node = [], goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path)), dist[goal]