#!/usr/bin/env python3
"""Change a map's scale, and move every scenario on that map along with it.

    python scripts/rescale_map.py kornaten 1:30              # or a factor: 0.0333
    python scripts/rescale_map.py kornaten 1:30 --dry-run    # show, change nothing

Scenario positions are in the map's scaled metres, so a new scale would put
every via point and buoy somewhere else. Scaling stretches the map about its
origin, which makes the fix exact: every position is multiplied by the ratio of
the scales. This does that, and keeps the rest consistent:

  maps/catalog.yaml   scale; simplify (x ratio) and min_area (x ratio²), so the
                      map keeps exactly the same islands and coastline detail
  scenarios/*.yaml    boat, buoys and waypoints of every scenario on the map;
                      planner grid_spacing and max_leg_distance (x ratio), so
                      planning stays as fast as before

Files are edited line by line, so comments survive. Everything is checked
against the rescaled map before any file is written. Sizes or distances written
in a description are not touched; the script lists the descriptions to reread.
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src" / "graph_route_planner"))

from scenario import (  # noqa: E402
    SCENARIOS_DIR, check_positions, list_scenarios, load_scenario)

NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def parse_scale(text: str) -> float:
    """'1:30' or '0.0333' → 0.0333…"""
    if ":" in text:
        a, b = text.split(":", 1)
        value = float(a) / float(b)
    else:
        value = float(text)
    if value <= 0:
        raise ValueError(f"scale must be positive, got {text}")
    return value


def ratio_name(scale: float) -> str:
    return "1:1" if scale == 1 else f"1:{1 / scale:.4g}"


def fmt(value: float) -> str:
    return f"{round(value, 1):g}"


def fmt_option(value: float) -> str:
    # Enough digits that 1/30 reads back as 1:30, not 1:30.003.
    return f"{value:.6g}"


# ── catalog ───────────────────────────────────────────────────────────────────

def entry_span(lines: list[str], name: str) -> tuple[int, int]:
    """First and one-past-last line of a top-level catalog entry."""
    start = next((i for i, line in enumerate(lines) if re.match(rf"^{re.escape(name)}:\s*$", line)),
                 None)
    if start is None:
        raise SystemExit(f"No entry '{name}' in maps/catalog.yaml")
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " #"):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return start, end


def rescale_catalog(text: str, name: str, new_scale: float) -> tuple[str, float]:
    """The catalog text with the entry rescaled, and the old scale."""
    lines = text.splitlines()
    start, end = entry_span(lines, name)
    entry = lines[start:end]

    def get(key):
        for line in entry:
            m = re.match(rf"^\s+{key}:\s*({NUMBER})", line)
            if m:
                return float(m.group(1))
        return None

    old_scale = get("scale") or 1.0
    ratio = new_scale / old_scale
    new = {"scale": new_scale}
    if get("simplify") is not None:
        new["simplify"] = get("simplify") * ratio
    if get("min_area") is not None:
        new["min_area"] = get("min_area") * ratio * ratio

    out = []
    for line in entry:
        m = re.match(rf"^(\s+)(scale|simplify|min_area):\s*{NUMBER}(.*)$", line)
        if m:
            line = f"{m.group(1)}{m.group(2)}: {fmt_option(new[m.group(2)])}{m.group(3)}"
        elif re.match(r"^\s+description:", line):
            line = re.sub(r"\bat 1:\d+(?:\.\d+)?", f"at {ratio_name(new_scale)}", line)
        out.append(line)
    if get("scale") is None:
        out.append(f"  scale: {fmt_option(new_scale)}")
    lines[start:end] = out
    return "\n".join(lines) + "\n", old_scale


# ── scenarios ─────────────────────────────────────────────────────────────────

def rescale_scenario_text(text: str, ratio: float) -> str:
    """Multiply every position and the planner's distances in a scenario file."""
    out, in_description = [], False
    for line in text.splitlines():
        if re.match(r"^description:", line):
            in_description = True
            out.append(line)
            continue
        if in_description and (not line.strip() or line.startswith(" ")):
            out.append(line)
            continue
        in_description = False

        code, hash_, comment = line.partition("#")
        code = re.sub(rf"\b(east|north):(\s*)({NUMBER})",
                      lambda m: f"{m.group(1)}:{m.group(2)}{fmt(float(m.group(3)) * ratio)}",
                      code)
        # [east, north] pairs: a waypoint item, or `boat: [x, y]`.
        code = re.sub(rf"^(\s*(?:-|boat:)\s*\[\s*)({NUMBER})(\s*,\s*)({NUMBER})(\s*\])",
                      lambda m: (f"{m.group(1)}{fmt(float(m.group(2)) * ratio)}{m.group(3)}"
                                 f"{fmt(float(m.group(4)) * ratio)}{m.group(5)}"), code)
        code = re.sub(rf"^(\s+(?:grid_spacing|max_leg_distance):\s*)({NUMBER})",
                      lambda m: f"{m.group(1)}{fmt(float(m.group(2)) * ratio)}", code)
        out.append(code + hash_ + comment)
    return "\n".join(out) + "\n"


def expected(scenario, ratio: float):
    """What a rescaled scenario must load as."""
    def pt(p):
        return (round(p[0] * ratio, 1), round(p[1] * ratio, 1))
    return (pt(scenario.boat), [pt(w) for w in scenario.waypoints],
            [pt((b.east, b.north)) for b in scenario.buoys],
            round(scenario.grid_spacing * ratio, 1),
            round(scenario.max_leg_distance * ratio, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog=__doc__.split("\n\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", help="map name in maps/catalog.yaml")
    parser.add_argument("scale", help="new scale, as 1:N or a factor")
    parser.add_argument("--dry-run", action="store_true", help="show the changes only")
    args = parser.parse_args()

    from graph_route_planner import MAPS_DIR
    from graph_route_planner.map_loader import CATALOG_FILE, load_kml, resolve_map

    new_scale = parse_scale(args.scale)
    catalog_path = Path(MAPS_DIR) / CATALOG_FILE
    if catalog_path.resolve() != (REPO / "maps" / CATALOG_FILE).resolve():
        raise SystemExit(f"Refusing to edit {catalog_path}: edit the catalog in the "
                         f"repository's maps/ (run from a source checkout).")
    new_catalog, old_scale = rescale_catalog(catalog_path.read_text(), args.map, new_scale)
    ratio = new_scale / old_scale
    if abs(ratio - 1) < 1e-9:
        print(f"'{args.map}' is already at {ratio_name(new_scale)}; nothing to do.")
        return

    # The rescaled map, built from the new catalog text without writing it.
    import yaml
    entry = yaml.safe_load(new_catalog)[args.map]
    kml, _ = resolve_map(args.map)
    options = {k: (tuple(entry[k]) if k == "crop" else entry[k])
               for k in ("scale", "crop", "simplify", "min_area") if k in entry}
    new_map = load_kml(kml, name=args.map, **options)
    x0, y0, x1, y1 = new_map.bounds
    print(f"{args.map}: {ratio_name(old_scale)} → {ratio_name(new_scale)} "
          f"(distances x{ratio:.3g}); the map becomes {x1 - x0:.0f} x {y1 - y0:.0f} m")

    # Rescale and check every scenario on the map before writing anything.
    changes, problems, descriptions = [], [], []
    for name in list_scenarios():
        sc = load_scenario(name)
        if sc.map.lower() != args.map.lower():
            continue
        new_text = rescale_scenario_text(sc.path.read_text(), ratio)
        tmp = SCENARIOS_DIR / f".{name}.rescale-check.yaml"
        try:
            tmp.write_text(new_text)
            got = load_scenario(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)
        want = expected(sc, ratio)
        have = (got.boat, got.waypoints, [(b.east, b.north) for b in got.buoys],
                got.grid_spacing, got.max_leg_distance)
        if have != want:
            problems.append(f"{name}: the rewritten file does not load as expected "
                            f"(unusual formatting?) — rescale it by hand")
            continue
        errors, _ = check_positions(got, new_map)
        problems += [f"{name}: {e}" for e in errors]
        changes.append((sc.path, new_text))
        if re.search(r"\d+\s*(?:x\s*\d+\s*)?m\b|1:\d+", sc.description):
            descriptions.append(name)
        print(f"  scenario {name}: {len(sc.waypoints)} waypoints, {len(sc.buoys)} buoys, "
              f"grid {sc.grid_spacing:g} → {got.grid_spacing:g} m, "
              f"max leg {sc.max_leg_distance:g} → {got.max_leg_distance:g} m")

    if problems:
        print("Not changing anything:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    catalog_path.write_text(new_catalog)
    for path, text in changes:
        path.write_text(text)
    print(f"Wrote maps/{CATALOG_FILE} and {len(changes)} scenario file(s).")
    if descriptions:
        print("These descriptions mention sizes or a scale; reread them: "
              + ", ".join(descriptions))
    print("Installed copies (colcon build) are refreshed by: colcon build --packages-select sail_maps")


if __name__ == "__main__":
    main()
