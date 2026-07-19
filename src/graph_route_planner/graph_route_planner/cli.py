"""Non-interactive entry point: plan once, print the legs, save the plots.

    python -m graph_route_planner.cli                    # built-in synthetic scenario
    python -m graph_route_planner.cli maps/Landgraben.kml
"""

import argparse
import logging
import math

import matplotlib.pyplot as plt

from graph_route_planner import DEFAULT_MAP
from graph_route_planner.geometry import polygons_of
from graph_route_planner.map_loader import load_kml
from graph_route_planner.planner import NoWaterError, PlannerConfig, Position, plan_route
from graph_route_planner.plotting import plot_environment, plot_graph, plot_polar
from graph_route_planner.sailing import SailingModel
from graph_route_planner.scenario import synthetic

log = logging.getLogger(__name__)


def farthest_pair(navigable) -> tuple[Position, Position]:
    """Pick two far-apart navigable points, for demoing a map with no endpoints.

    A KML has no start or goal, so the CLI invents them. Uses the eroded
    area's own vertices, which are guaranteed to be on water.

    Args:
        navigable: The area to pick endpoints from.

    Returns:
        The two most distant vertices, as ``(east, north)`` tuples.

    Raises:
        NoWaterError: If the area is empty, e.g. a margin that erodes the whole
            map. Without this the caller gets a bare IndexError.
    """
    pts: list[Position] = [(c[0], c[1]) for p in polygons_of(navigable)
                           for c in p.exterior.coords[:-1]]
    if not pts:
        raise NoWaterError("no navigable water left to pick endpoints from — "
                           "reduce the margin")
    best, pair = -1.0, (pts[0], pts[0])
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            d = math.dist(a, b)
            if d > best:
                best, pair = d, (a, b)
    return pair


def describe(route, model: SailingModel) -> None:
    """Print each leg's heading off the wind and resulting speed."""
    print(f"\nPath found: {len(route.waypoints)} waypoints, "
          f"total time = {route.duration:.2f} units\n")
    print("Waypoint  |  Position         |  Heading vs wind  |  Speed")
    print("-" * 60)
    for i, wp in enumerate(route.waypoints):
        if i == 0:
            print(f"  [{i}]      ({wp[0]:5.1f}, {wp[1]:5.1f})   —  start")
            continue
        alpha = math.degrees(model.angle_off_wind(route.waypoints[i - 1], wp))
        spd = model.polar_speed(math.radians(alpha))
        print(f"  [{i}]      ({wp[0]:5.1f}, {wp[1]:5.1f})   {alpha:5.1f}° from wind   {spd:.1f} kn")


def main() -> None:
    """Entry point: ``python -m graph_route_planner.cli [map.kml]``."""
    ap = argparse.ArgumentParser(description="Graph-based sailing route planner")
    ap.add_argument("kml", nargs="?",
                    help=f"KML map to plan on (default: the synthetic scenario; "
                         f"try {DEFAULT_MAP.name})")
    ap.add_argument("--wind", type=float, default=0.0,
                    help="bearing the wind blows FROM, degrees (0 = north)")
    ap.add_argument("--margin", type=float, default=5.0, help="shore clearance, metres")
    ap.add_argument("--grid", type=float, default=12.0, help="waypoint grid spacing, metres")
    ap.add_argument("--merge", type=float, default=3.0, help="node merge threshold, metres")
    ap.add_argument("--min-leg", type=float, default=0.0,
                    help="shortest leg to consider, metres (0 disables)")
    ap.add_argument("--max-leg", type=float, default=None,
                    help="longest leg to consider, metres (default: the map's span)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log planner internals")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    model = SailingModel.from_bearing(args.wind)
    config = PlannerConfig(margin=args.margin, grid_spacing=args.grid,
                           merge_threshold=args.merge,
                           min_leg_distance=args.min_leg,
                           max_leg_distance=args.max_leg)

    try:
        if args.kml:
            sail_map = load_kml(args.kml)
            water = sail_map.water
            print(f"Loaded '{sail_map.name}': {len(sail_map.features)} features, "
                  f"{len(sail_map.components)} navigable body(ies)")
            # Nudge the endpoints inside the shore margin so they stay on water.
            start, goal = farthest_pair(sail_map.components[0].buffer(-args.margin * 1.5))
        else:
            scene = synthetic()
            water, start, goal = scene.water, scene.start, scene.goal
            print(f"Scenario: {scene.name}")

        print(f"Planning from {start} to {goal} …")
        route = plan_route(water, start, goal, model=model, config=config)
    except NoWaterError as e:
        print(f"✗  {e}")
        return

    if not route.found:
        print("No route found. Try reducing the margin.")
        return

    unsailable = [i for i in range(len(route.waypoints) - 1)
                  if not model.heading_is_sailable(route.waypoints[i], route.waypoints[i + 1])]
    if unsailable:
        print(f"⚠  WARNING: {len(unsailable)} unsailable leg(s) — check build_graph!")
    else:
        print("✓  All path edges verified sailable.")

    describe(route, model)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [3, 1]})
    plot_environment(axes[0], water, route.navigable, route.waypoints, start, goal, model)
    plot_polar(axes[1], model)
    plt.tight_layout()
    plt.savefig("sail_path_wind.png", dpi=150)
    print("\nSaved → sail_path_wind.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_graph(ax, route.graph)
    ax.set_title("Full Graph (for debugging)", fontsize=10)
    plt.savefig("full_graph.png", dpi=150)
    print("Saved → full_graph.png")
    plt.show()


if __name__ == "__main__":
    main()
