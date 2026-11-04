"""Scenario files: a map plus everything placed on it, as YAML.

A scenario is one file in scenarios/ (or anywhere, given by path):

    description: A lap around the island.          # optional, free text
    map: landgraben                                 # name from maps/catalog.yaml
    boat: {east: 0, north: -38}                     # start, metres
    wind: {direction: 180, speed: 4}                # degrees it blows FROM, m/s
    buoys:                                          # optional
      - {east: -10, north: -30, color: orange}      # named colour ...
      - {east: 22, north: -22, color: [1, 0, 0]}    # ... or RGB in [0, 1]
    waypoints:                                      # via points, rounded in order
      - [27, -12]
      - {east: 27, north: 8}
    planner:                                        # optional
      grid_spacing: 6          # tacking-grid spacing, metres (default 10)
      max_leg_distance: 120    # longest leg, metres; 0 = from the map (default)
      planning_delay: 15       # seconds to wait for the first route (default 5)

Positions are local (east, north) metres of the map, as scripts/render_maps.py
shows them. `load_scenario` checks the file's structure; `check_positions`
checks that the boat, buoys and via points are on water.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO / "scenarios"

#: Buoy colours by name. Named colours keep scenario files readable, and give
#: language tasks ("go to the green buoy") a name to refer to.
COLORS = {
    "orange": (1.0, 0.5, 0.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.8, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "black": (0.1, 0.1, 0.1),
}
DEFAULT_COLOR = "orange"

DEFAULT_GRID_SPACING = 10.0
DEFAULT_MAX_LEG_DISTANCE = 0.0
DEFAULT_PLANNING_DELAY = 5.0

#: The planner's shore clearance (graph_route_planner parameters.yaml margin).
#: A via point closer to land than this cannot be reached exactly.
PLANNER_MARGIN = 2.0

#: How far onto land (metres) a position may lie before it counts as on land.
#: A buoy placed exactly on a coastline vertex is on the boundary, which
#: floating point decides either way; that is not a mistake worth refusing.
LAND_TOLERANCE = 0.5

_TOP_KEYS = {"description", "map", "boat", "wind", "buoys", "waypoints", "planner"}
_PLANNER_KEYS = {"grid_spacing", "max_leg_distance", "planning_delay"}


class ScenarioError(ValueError):
    """A scenario file that cannot be used, with a message saying why."""


@dataclass
class Buoy:
    east: float
    north: float
    rgb: tuple[float, float, float]
    color: str | None = None      # the colour's name, if it was given by name


@dataclass
class Scenario:
    name: str
    map: str
    boat: tuple[float, float]
    wind_direction: float         # degrees the wind blows FROM, 0 = north
    wind_speed: float             # m/s
    waypoints: list[tuple[float, float]]
    buoys: list[Buoy] = field(default_factory=list)
    description: str = ""
    grid_spacing: float = DEFAULT_GRID_SPACING
    max_leg_distance: float = DEFAULT_MAX_LEG_DISTANCE
    planning_delay: float = DEFAULT_PLANNING_DELAY
    path: Path | None = None


def list_scenarios(directory: Path = SCENARIOS_DIR) -> list[str]:
    """Names of the scenario files in a directory."""
    return sorted(p.stem for p in Path(directory).glob("*.yaml"))


def resolve_scenario(spec: str, directory: Path = SCENARIOS_DIR) -> Path:
    """The file a scenario name or path refers to."""
    path = Path(spec)
    if path.is_file():
        return path
    candidate = Path(directory) / f"{spec}.yaml"
    if candidate.is_file():
        return candidate
    known = ", ".join(list_scenarios(directory)) or "none"
    raise ScenarioError(f"No scenario '{spec}' (known: {known}), nor a YAML file of that name.")


def _number(value, where: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScenarioError(f"{where}: expected a number, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ScenarioError(f"{where}: {value} is outside [{lo}, {hi}]")
    return float(value)


def _position(value, where: str) -> tuple[float, float]:
    """[east, north] or {east: .., north: ..}."""
    if isinstance(value, dict):
        extra = set(value) - {"east", "north"}
        if extra or not {"east", "north"} <= set(value):
            raise ScenarioError(f"{where}: expected {{east, north}}, got keys {sorted(value)}")
        east, north = value["east"], value["north"]
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        east, north = value
    else:
        raise ScenarioError(f"{where}: expected [east, north] or {{east, north}}, got {value!r}")
    return _number(east, f"{where} east"), _number(north, f"{where} north")


def _buoy(value, where: str) -> Buoy:
    if not isinstance(value, dict):
        raise ScenarioError(f"{where}: expected {{east, north, color}}, got {value!r}")
    extra = set(value) - {"east", "north", "color"}
    if extra:
        raise ScenarioError(f"{where}: unknown key(s) {sorted(extra)}")
    east, north = _position({k: value.get(k) for k in ("east", "north")}, where)
    color = value.get("color", DEFAULT_COLOR)
    if isinstance(color, str):
        if color.lower() not in COLORS:
            raise ScenarioError(f"{where}: unknown colour '{color}' "
                                f"(known: {', '.join(COLORS)}, or [r, g, b])")
        return Buoy(east, north, COLORS[color.lower()], color.lower())
    if isinstance(color, (list, tuple)) and len(color) == 3:
        rgb = tuple(_number(c, f"{where} color", 0.0, 1.0) for c in color)
        name = next((n for n, c in COLORS.items() if c == rgb), None)
        return Buoy(east, north, rgb, name)
    raise ScenarioError(f"{where}: color must be a name or [r, g, b], got {color!r}")


def load_scenario(spec: str, directory: Path = SCENARIOS_DIR) -> Scenario:
    """Read and validate a scenario by name (a file in scenarios/) or path.

    Raises:
        ScenarioError: With a message naming the file and the problem.
    """
    path = resolve_scenario(spec, directory)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as err:
        raise ScenarioError(f"{path}: not valid YAML: {err}") from err
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: expected a mapping of scenario fields")

    def fail(message: str):
        raise ScenarioError(f"{path}: {message}")

    try:
        extra = set(data) - _TOP_KEYS
        if extra:
            fail(f"unknown key(s) {sorted(extra)} (allowed: {', '.join(sorted(_TOP_KEYS))})")
        missing = {"map", "boat", "wind", "waypoints"} - set(data)
        if missing:
            fail(f"missing required key(s) {sorted(missing)}")

        if not isinstance(data["map"], str) or not data["map"]:
            fail(f"map: expected a map name, got {data['map']!r}")

        wind = data["wind"]
        if not isinstance(wind, dict) or set(wind) != {"direction", "speed"}:
            fail(f"wind: expected {{direction, speed}}, got {wind!r}")
        direction = _number(wind["direction"], "wind direction", 0.0, 360.0) % 360.0
        speed = _number(wind["speed"], "wind speed", 0.0, 25.0)

        waypoints = data["waypoints"]
        if not isinstance(waypoints, list) or not waypoints:
            fail("waypoints: expected a non-empty list of [east, north]")

        buoys = data.get("buoys") or []
        if not isinstance(buoys, list):
            fail("buoys: expected a list")

        planner = data.get("planner") or {}
        if not isinstance(planner, dict):
            fail("planner: expected a mapping")
        extra = set(planner) - _PLANNER_KEYS
        if extra:
            fail(f"planner: unknown key(s) {sorted(extra)} "
                 f"(allowed: {', '.join(sorted(_PLANNER_KEYS))})")

        return Scenario(
            name=path.stem,
            map=data["map"],
            boat=_position(data["boat"], "boat"),
            wind_direction=direction,
            wind_speed=speed,
            waypoints=[_position(w, f"waypoint {i + 1}") for i, w in enumerate(waypoints)],
            buoys=[_buoy(b, f"buoy {i + 1}") for i, b in enumerate(buoys)],
            description=str(data.get("description") or "").strip(),
            grid_spacing=_number(planner.get("grid_spacing", DEFAULT_GRID_SPACING),
                                 "planner grid_spacing", 0.5),
            max_leg_distance=_number(planner.get("max_leg_distance", DEFAULT_MAX_LEG_DISTANCE),
                                     "planner max_leg_distance", 0.0),
            planning_delay=_number(planner.get("planning_delay", DEFAULT_PLANNING_DELAY),
                                   "planner planning_delay", 0.0),
            path=path,
        )
    except ScenarioError as err:
        if str(err).startswith(str(path)):
            raise
        raise ScenarioError(f"{path}: {err}") from None


def check_positions(scenario: Scenario, sail_map=None) -> tuple[list[str], list[str]]:
    """Check the boat, via points and buoys against the scenario's map.

    Args:
        scenario: A loaded scenario.
        sail_map: The map, if already loaded; otherwise it is loaded by name.

    Returns:
        ``(errors, warnings)``: anything more than LAND_TOLERANCE onto land is
        an error; a via point closer to land than the planner's margin is a
        warning, since the planner can only get near it.
    """
    from shapely.geometry import Point

    if sail_map is None:
        sail_map = load_scenario_map(scenario)
    water = sail_map.water
    errors, warnings = [], []

    def where(x, y):
        return f"({x:g}, {y:g})"

    def on_land(x, y) -> float:
        """How far onto land a point is (0 on water); errors past the tolerance."""
        return 0.0 if water.contains(Point(x, y)) else water.distance(Point(x, y))

    def report(what: str, x: float, y: float) -> None:
        depth = on_land(x, y)
        if depth > LAND_TOLERANCE:
            errors.append(f"{what} {where(x, y)} is on land, {depth:.1f} m from water")

    report("boat start", *scenario.boat)
    for i, (x, y) in enumerate(scenario.waypoints, 1):
        report(f"waypoint {i}", x, y)
        if on_land(x, y) <= LAND_TOLERANCE and water.boundary.distance(Point(x, y)) < PLANNER_MARGIN:
            warnings.append(f"waypoint {i} {where(x, y)} is within {PLANNER_MARGIN:g} m "
                            f"of land; the planner can only get close to it")
    for i, b in enumerate(scenario.buoys, 1):
        report(f"buoy {i}", b.east, b.north)
    return errors, warnings


def load_scenario_map(scenario: Scenario):
    """The scenario's map, loaded exactly as the simulation and planner load it."""
    try:
        from graph_route_planner.map_loader import load_map
    except ImportError:     # not sourced: use the source tree
        sys.path.insert(0, str(REPO / "src" / "graph_route_planner"))
        from graph_route_planner.map_loader import load_map
    return load_map(scenario.map)


def _fmt(value: float) -> str:
    """A coordinate rounded to 0.1 m, without trailing zeros."""
    return f"{round(value, 1):g}"


def scenario_to_yaml(scenario: Scenario) -> str:
    """The scenario as YAML, in the same style as the hand-written files.

    Inline comments of a file that was loaded are not kept; the description is.
    """
    lines = []
    if scenario.description:
        lines.append("description: >")
        words, line = scenario.description.split(), "  "
        for word in words:
            if len(line) + len(word) > 78 and line.strip():
                lines.append(line.rstrip())
                line = "  "
            line += word + " "
        lines.append(line.rstrip())
    lines.append(f"map: {scenario.map}")
    lines.append(f"boat: {{east: {_fmt(scenario.boat[0])}, north: {_fmt(scenario.boat[1])}}}")
    lines.append(f"wind: {{direction: {round(scenario.wind_direction, 1):g}, "
                 f"speed: {round(scenario.wind_speed, 1):g}}}")
    if scenario.buoys:
        lines.append("buoys:")
        for b in scenario.buoys:
            color = b.color or "[" + ", ".join(f"{c:g}" for c in b.rgb) + "]"
            lines.append(f"  - {{east: {_fmt(b.east)}, north: {_fmt(b.north)}, color: {color}}}")
    lines.append("waypoints:")
    for x, y in scenario.waypoints:
        lines.append(f"  - [{_fmt(x)}, {_fmt(y)}]")
    planner = []
    if scenario.grid_spacing != DEFAULT_GRID_SPACING:
        planner.append(f"  grid_spacing: {scenario.grid_spacing:g}")
    if scenario.max_leg_distance != DEFAULT_MAX_LEG_DISTANCE:
        planner.append(f"  max_leg_distance: {scenario.max_leg_distance:g}")
    if scenario.planning_delay != DEFAULT_PLANNING_DELAY:
        planner.append(f"  planning_delay: {scenario.planning_delay:g}")
    if planner:
        lines += ["planner:", *planner]
    return "\n".join(lines) + "\n"


def save_scenario(scenario: Scenario, path: Path) -> Path:
    """Write a scenario file; it loads back to the same scenario (to 0.1 m)."""
    path = Path(path)
    if not scenario.waypoints:
        raise ScenarioError("a scenario needs at least one waypoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scenario_to_yaml(scenario))
    return path
