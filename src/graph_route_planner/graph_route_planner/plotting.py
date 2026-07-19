"""Drawing maps, routes and polar diagrams with matplotlib."""

import math
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.ops import unary_union

from graph_route_planner.geometry import polygons_of
from graph_route_planner.map_loader import immediate_children
from graph_route_planner.sailing import SailingModel

if TYPE_CHECKING:
    from graph_route_planner.map_loader import SailMap

LAND_COLOR  = "#ddd6c6"
WATER_COLOR = "#bfe3f5"


def _patch(poly, **kwargs) -> PathPatch:
    """
    A matplotlib patch for a shapely polygon, holes included.

    Filling only the exterior would paint over the islands, so rings are
    stitched into one compound path and matplotlib's even-odd rule cuts the
    holes out.
    """
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        coords = list(ring.coords)
        verts.extend(coords)
        codes.extend([MplPath.MOVETO]
                     + [MplPath.LINETO] * (len(coords) - 2)
                     + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(verts, codes), **kwargs)


def draw_water(ax: Axes, water, facecolor: str = WATER_COLOR, zorder: int = 0):
    """
    Paint navigable water over a land-coloured background.

    Everything that is not water is land, so the axes background carries the
    land and each water body is punched on top of it. This renders correctly
    at any nesting depth without caring how deep it goes.
    """
    ax.set_facecolor(LAND_COLOR)
    for poly in polygons_of(water):
        ax.add_patch(_patch(poly, facecolor=facecolor, edgecolor="steelblue",
                            linewidth=1.2, zorder=zorder))


def plot_map(ax: Axes, sail_map: "SailMap", labels: bool = True):
    """Draw a loaded SailMap. Axes are metres east/north of the map origin."""
    draw_water(ax, sail_map.water)

    if labels:
        for f in sail_map.features:
            # Label inside the feature's own surface, not on top of whatever
            # is nested within it.
            kids = [g.polygon for g in immediate_children(sail_map.features, f)]
            surface = f.polygon.difference(unary_union(kids)) if kids else f.polygon
            if surface.is_empty:
                continue
            pt = surface.representative_point()
            ax.text(pt.x, pt.y, f.name, fontsize=7, ha="center", va="center",
                    color="black", zorder=6)

    x0, y0, x1, y1 = sail_map.bounds
    pad = 0.05 * max(x1 - x0, y1 - y0, 1.0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("metres east of origin", fontsize=8)
    ax.set_ylabel("metres north of origin", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_title(f"Map: {sail_map.name}  "
                 f"({len(sail_map.components)} navigable body(ies))", fontsize=10)


def plot_polar(ax: Axes, model: SailingModel):
    """
    Classic polar plot: radius = speed, angle = heading relative to wind.
    Plotted symmetrically for port and starboard tacks.
    """
    alphas = [math.radians(a) for a in range(0, 181)]
    speeds = [model.polar_speed(a) for a in alphas]

    # Mirror for port tack
    all_a = alphas + [-a for a in reversed(alphas)]
    all_s = speeds + list(reversed(speeds))

    ax.plot([s * math.sin(a) for a, s in zip(all_a, all_s)],
            [s * math.cos(a) for a, s in zip(all_a, all_s)],
            color="royalblue", linewidth=2)

    # No-go zone shading
    ngo = [math.radians(a) for a in range(-int(model.no_go_deg), int(model.no_go_deg) + 1)]
    r   = 12
    ax.fill([r * math.sin(a) for a in ngo] + [0],
            [r * math.cos(a) for a in ngo] + [0],
            color="red", alpha=0.15, label="No-go zone")

    # Wind arrow
    ax.annotate("", xy=(0, 10), xytext=(0, 14),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(0, 14.5, f"wind\nfrom {model.wind_bearing_deg:.0f}°",
            ha="center", va="bottom", fontsize=8)

    # Labels
    for a_deg, label in [(45, "45°"), (90, "90°"), (135, "135°"), (180, "180°")]:
        a = math.radians(a_deg)
        s = model.polar_speed(math.radians(a_deg)) + 1.5
        ax.text(s * math.sin(a), s * math.cos(a), label, fontsize=7,
                ha="center", color="gray")

    ax.set_aspect("equal")
    ax.set_title("Polar Diagram", fontsize=10)
    ax.set_xlim(-13, 13)
    ax.set_ylim(-13, 16)
    ax.axis("off")
    ax.legend(fontsize=8, loc="lower center")


def draw_navigable(ax: Axes, navigable):
    """Outline the navigable area: water minus the shore clearance margin."""
    for poly in polygons_of(navigable):
        ax.add_patch(_patch(poly, facecolor="none", edgecolor="steelblue",
                            linewidth=1, linestyle="--", alpha=0.6, zorder=3))


def draw_wind(ax: Axes, bounds, span: float, model: SailingModel) -> float:
    """Draw a row of wind arrows above the map.

    The arrows follow the model rather than assuming a northerly, so the
    picture cannot disagree with what was planned.

    Args:
        ax: Axes to draw on.
        bounds: ``(min_x, min_y, max_x, max_y)`` of the map, in metres.
        span: Map extent in metres, used for scaling.
        model: Supplies the wind direction and bearing.

    Returns:
        The y coordinate the arrows sit at.
    """
    x0, _, x1, y1 = bounds
    top = y1 + 0.04 * span
    bx, by = model.blowing_towards
    length = 0.10 * span

    for i in range(6):
        x = x0 + (i + 0.5) * (x1 - x0) / 6
        tail = (x - bx * length / 2, top - by * length / 2)
        head = (x + bx * length / 2, top + by * length / 2)
        ax.annotate("", xy=head, xytext=tail,
                    arrowprops=dict(arrowstyle="->", color="deepskyblue", lw=1.5, alpha=0.7))

    ax.text(x0, top + 0.11 * span, f"wind from {model.wind_bearing_deg:.0f}°",
            fontsize=8, color="deepskyblue")
    return top


def draw_path(ax: Axes, path: list, span: float, model: SailingModel | None = None):
    """Draw the sailed path.

    Args:
        ax: Axes to draw on.
        path: Waypoints as ``(east, north)`` tuples.
        span: Map extent in metres, used to scale label offsets.
        model: If given, each leg is labelled with its heading off the wind and
            the resulting boat speed. Omit to draw the bare path.
    """
    if not path:
        return
    xs, ys = zip(*path)
    ax.plot(xs, ys, color="orange", linewidth=2.5, zorder=5, label="Sailed path")
    ax.plot(xs, ys, "o", color="orange", markersize=7, zorder=6)

    if model is None:
        return
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        alpha_deg = math.degrees(model.angle_off_wind(u, v))
        spd = model.polar_speed(math.radians(alpha_deg))
        mx, my = (u[0] + v[0]) / 2, (u[1] + v[1]) / 2
        ax.text(mx + 0.01 * span, my, f"{alpha_deg:.0f}°\n{spd:.1f}kn",
                fontsize=7, color="darkorange", zorder=7)


def plot_environment(ax: Axes, water, navigable, path: list, start: tuple, goal: tuple,
                     model: SailingModel):
    """Draw the water, the eroded navigable area, and the sailed path."""
    draw_water(ax, water)
    draw_navigable(ax, navigable)

    x0, y0, x1, y1 = water.bounds
    span = max(x1 - x0, y1 - y0)
    top = draw_wind(ax, water.bounds, span, model)
    draw_path(ax, path, span, model)

    ax.plot(*start, "go", markersize=10, zorder=8, label="Start")
    ax.plot(*goal,  "ro", markersize=10, zorder=8, label="Goal")

    pad = 0.05 * span
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, top + 0.16 * span)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Sail Path  (only directly sailable edges)", fontsize=10)


def plot_graph(ax: Axes, graph: dict, color: str = "lightgray",
               linewidth: float = 0.5, zorder: int = 2):
    """
    Draw the search graph as a single LineCollection.

    One ax.plot() per edge creates an artist per segment — tens of thousands
    on a fine grid — which costs seconds per redraw and makes the interactive
    viewer unusable. Edges are also deduplicated: the graph is directed, so
    u→v and v→u would otherwise be drawn on top of each other.
    """
    seen = set()
    segments = []
    for u, neighbors in graph.items():
        for v in neighbors:
            key = (u, v) if u <= v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            segments.append((u, v))

    ax.add_collection(LineCollection(segments, colors=color,
                                     linewidths=linewidth, zorder=zorder))
    ax.autoscale_view()
