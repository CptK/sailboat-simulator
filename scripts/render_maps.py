#!/usr/bin/env python3
"""Render maps to PNG for visual inspection.

For each map this draws what the simulation and planner will actually use
(after crop, scale and simplification) in simulation metres:

    light blue   navigable water
    pale blue    water too close to land for the planner (within --margin),
                 so narrow channels that are closed for the boat stand out
    green        land

and, with --overview, the raw source file in lon/lat next to it, with the
crop box drawn in red — the view to use when choosing a crop.

    python scripts/render_maps.py                       # every map in the catalog
    python scripts/render_maps.py kornaten --overview   # one map, plus its source
    python scripts/render_maps.py maps/Some.kml         # any KML file
    python scripts/render_maps.py --scenarios           # overlay the courses from
                                                        # scenarios/*.yaml, with planned routes

Images go to --out (default: map_previews/).
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "graph_route_planner"))

from graph_route_planner import MAPS_DIR  # noqa: E402
from graph_route_planner.geometry import polygons_of  # noqa: E402
from graph_route_planner.graph import navigable_area  # noqa: E402
from graph_route_planner.map_loader import (  # noqa: E402
    _parse_coords, load_catalog, load_kml, resolve_map)

WATER = "#a9d6f5"
TOO_NARROW = "#dcecf7"
LAND = "#8fbf6a"
COAST = "#4d7a35"
BOAT_LENGTH_M = 2.7


def draw_polygon(ax, poly, **style):
    """Fill a shapely polygon, holes included."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        pts = list(ring.coords)
        verts += pts
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), **style))


def scenarios_for(map_name: str, kml: Path) -> list:
    """Scenarios (scenarios/*.yaml) that sail on this map."""
    sys.path.insert(0, str(REPO / "scripts"))
    from scenario import ScenarioError, list_scenarios, load_scenario

    found = []
    for name in list_scenarios():
        try:
            sc = load_scenario(name)
        except ScenarioError as err:
            print(f"  (skipping scenario {name}: {err})")
            continue
        if sc.map.lower() in (map_name.lower(), kml.name.lower(), kml.stem.lower()):
            found.append(sc)
    return found


def plan_course(sail_map, scenario, margin: float) -> list:
    """The route the graph planner would sail for a scenario, leg by leg.

    Uses the scenario's own planner settings and wind, so what is drawn is
    what the planner node will publish at the start of a run.
    """
    from graph_route_planner.planner import CachedPlanner, PlannerConfig
    from graph_route_planner.sailing import SailingModel

    config = PlannerConfig(margin=margin, grid_spacing=scenario.grid_spacing,
                           merge_threshold=1.5, segment_cost=1.0,
                           max_leg_distance=scenario.max_leg_distance or None)
    planner = CachedPlanner(sail_map.water, config)
    model = SailingModel.from_bearing(scenario.wind_direction, no_go_deg=50)
    pts = [scenario.boat, *scenario.waypoints]
    route = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        leg = planner.plan(a, b, model)
        if not leg.found:
            raise ValueError(f"no route from {a} to {b}")
        route += list(leg.waypoints)[1:]
    return route


def draw_scenario(ax, scenario, sail_map, margin: float):
    bx, by = scenario.boat
    ax.plot(bx, by, "k^", ms=9, zorder=6, label=f"start ({scenario.name})")
    xs, ys = zip(*[scenario.boat, *scenario.waypoints])
    ax.plot(xs, ys, "k--", lw=0.6, alpha=0.5, zorder=5)
    try:
        route = plan_course(sail_map, scenario, margin)
        rx, ry = zip(*route)
        ax.plot(rx, ry, "-", color="#c0392b", lw=1.4, zorder=6, label="planned route")
        length = sum(math.dist(p, q) for p, q in zip(route, route[1:]))
        print(f"  scenario '{scenario.name}': planned route {length:.0f} m")
    except Exception as err:
        print(f"  scenario '{scenario.name}': could not plan ({err})")
    wind_from = math.radians(scenario.wind_direction)
    x0, y0, x1, y1 = sail_map.bounds
    span = max(x1 - x0, y1 - y0)
    # Arrow pointing where the wind blows to, in the top-left corner.
    ax.annotate("", xy=(x0 + 0.12 * span - 0.05 * span * math.sin(wind_from),
                        y1 - 0.08 * span - 0.05 * span * math.cos(wind_from)),
                xytext=(x0 + 0.12 * span, y1 - 0.08 * span), zorder=9,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.text(x0 + 0.12 * span, y1 - 0.03 * span, f"wind {scenario.wind_speed:g} m/s",
            ha="center", fontsize=8, zorder=9)
    for b in scenario.buoys:
        ax.plot(b.east, b.north, "o", color=b.rgb, mec="black", ms=7, zorder=7)
    # Via point numbers sit beside the point, so a buoy on it does not hide them.
    for i, (x, y) in enumerate(scenario.waypoints, 1):
        ax.plot(x, y, "o", mfc="none", mec="black", ms=11, mew=0.8, zorder=8)
        ax.annotate(str(i), (x, y), xytext=(9, 9), textcoords="offset points",
                    fontsize=8, ha="center", va="center", zorder=9,
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=0.6))


def render_map(ax, sail_map, margin: float, title: str):
    """The map as the boat sees it, in simulation metres."""
    water = sail_map.water
    navigable = navigable_area(water, margin) if margin > 0 else water
    for f in sail_map.features:
        if not f.is_water:
            draw_polygon(ax, f.polygon, fc=LAND, ec=COAST, lw=0.4, zorder=1)
    for p in polygons_of(water):
        draw_polygon(ax, p, fc=TOO_NARROW, ec="none", zorder=2)
    for p in polygons_of(navigable):
        draw_polygon(ax, p, fc=WATER, ec="none", zorder=3)

    x0, y0, x1, y1 = sail_map.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor(LAND)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(title, fontsize=10)

    # Scale bar with the boat for reference.
    span = max(x1 - x0, y1 - y0)
    magnitude = 10 ** math.floor(math.log10(span / 5))
    bar = max(m * magnitude for m in (1, 2, 5) if m * magnitude <= span / 4)
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.04 * (y1 - y0)
    ax.plot([bx, bx + bar], [by, by], "k-", lw=3, zorder=8)
    ax.text(bx, by + 0.015 * span, f"{bar:g} m  (boat: {BOAT_LENGTH_M} m)",
            ha="left", fontsize=8, zorder=8)


def render_overview(ax, kml: Path, crop):
    """The raw source file in lon/lat, with the crop box."""
    root = ET.parse(kml).getroot()
    for poly in root.iterfind(".//{*}Polygon"):
        outer = _parse_coords(poly.findtext(".//{*}outerBoundaryIs//{*}coordinates") or "")
        if len(outer) >= 3:
            xs, ys = zip(*outer)
            ax.fill(xs, ys, fc=LAND, ec=COAST, lw=0.3)
    for line in root.iterfind(".//{*}LineString"):
        pts = _parse_coords(line.findtext("{*}coordinates") or "")
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COAST, lw=0.6)
    if crop:
        lo0, la0, lo1, la1 = crop
        ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], "r-", lw=1.5,
                label="crop")
        ax.legend(loc="upper right", fontsize=8)
    lat_mid = sum(ax.get_ylim()) / 2
    ax.set_aspect(1 / math.cos(math.radians(lat_mid)))
    ax.set_facecolor(WATER)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"source: {kml.name}", fontsize=10)
    ax.grid(True, lw=0.3)


def describe(sail_map, opts: dict) -> str:
    x0, y0, x1, y1 = sail_map.bounds
    land = [f for f in sail_map.features if not f.is_water]
    verts = sum(len(f.polygon.exterior.coords) +
                sum(len(r.coords) for r in f.polygon.interiors) for f in sail_map.features)
    scale = opts.get("scale", 1.0)
    ratio = f"1:{1 / scale:.4g}" if scale != 1 else "1:1"
    return (f"{sail_map.name}  —  scale {ratio}, {x1 - x0:.0f} × {y1 - y0:.0f} m, "
            f"{len(land)} land pieces, {verts} vertices")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("maps", nargs="*",
                        help="catalog names, file names in maps/, or KML paths "
                             "(default: every catalog entry)")
    parser.add_argument("--out", default="map_previews", help="output directory")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="planner shore clearance in metres (default 2.0)")
    parser.add_argument("--overview", action="store_true",
                        help="also draw the raw source file with the crop box")
    parser.add_argument("--scenarios", action="store_true",
                        help="overlay the scenarios (scenarios/*.yaml) on each map")
    parser.add_argument("--show", action="store_true", help="open a window as well")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = args.maps or sorted(load_catalog(MAPS_DIR))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    failed = False
    for spec in specs:
        kml, opts = resolve_map(spec)
        name = spec if not Path(spec).suffix else kml.stem
        ncols = 2 if args.overview else 1
        fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 9), squeeze=False)
        try:
            sail_map = load_kml(kml, name=name, **opts)
        except ValueError as err:
            print(f"{name}: {err}")
            failed = True
            sail_map = None
        if sail_map is not None:
            title = describe(sail_map, opts)
            render_map(axes[0][0], sail_map, args.margin, title)
            if args.scenarios:
                for scenario in scenarios_for(name, kml):
                    draw_scenario(axes[0][0], scenario, sail_map, args.margin)
            print(title)
        else:
            axes[0][0].set_title(f"{name}: could not load (see console)", fontsize=10)
        if args.overview:
            render_overview(axes[0][-1], kml, opts.get("crop"))
        fig.tight_layout()
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110)
        print(f"  → {path}")
        if not args.show:
            plt.close(fig)
    if args.show:
        plt.show()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
