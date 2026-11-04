"""Graph-based sailing route planner.

Loads maps from KML, projects them to a local (east, north) metric frame,
and plans a wind-aware route across the water.

    from graph_route_planner import MAPS_DIR
    from graph_route_planner.map_loader import load_kml
    from graph_route_planner.planner import plan_route, PlannerConfig
    from graph_route_planner.sailing import SailingModel

    sail_map = load_kml(MAPS_DIR / "Landgraben.kml")
    route = plan_route(sail_map.water, start, goal,
                       model=SailingModel.from_bearing(0.0),
                       config=PlannerConfig(margin=2.0))

The library is plain Python and imports no ROS, so it stays testable without
a running graph; `node.py` is the only ROS entry point.

This package is self-contained: it does not import from the ROS
`route_planner` package, and is intended to replace it once complete.
"""

import os
from pathlib import Path


#: Environment variable that, when set, overrides where maps are looked up.
MAPS_DIR_ENV = "SAIL_MAPS_DIR"

#: The file every maps directory must contain.
_CATALOG = "catalog.yaml"


def _maps_dir_candidates() -> list[Path]:
    """Where the shared maps (the sail_maps package) may be, best first."""
    here = Path(__file__).resolve()
    candidates = []
    if os.environ.get(MAPS_DIR_ENV):
        candidates.append(Path(os.environ[MAPS_DIR_ENV]))
    # Source checkout or --symlink-install: <repo>/src/graph_route_planner/
    # graph_route_planner/__init__.py, maps at <repo>/maps.
    candidates.append(here.parents[3] / "maps")
    # The installed sail_maps package, if the workspace was sourced after it
    # was first built.
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("sail_maps")))
    except Exception:      # not on the ament path, or ament unavailable
        pass
    # Installed without sourcing: <ws>/install/graph_route_planner/lib/
    # pythonX.Y/site-packages/graph_route_planner/__init__.py. A shell sourced
    # before sail_maps existed does not know the package, but it sits right
    # next to this one (isolated layout) or in the same prefix (merged layout),
    # and the workspace itself holds the maps/ it was built from.
    for parent in here.parents:
        if parent.name == "install":
            candidates += [parent / "sail_maps" / "share" / "sail_maps",
                           parent / "share" / "sail_maps",
                           parent.parent / "maps"]
            break
    return candidates


def _maps_dir() -> Path:
    """Locate the shared maps: the repository's maps/ directory.

    Every map, and the catalog naming them (catalog.yaml), lives there so the
    simulation and the planners agree on what a map name means. It is the
    sail_maps ROS package; see _maps_dir_candidates for where it is looked
    for. Set SAIL_MAPS_DIR to use another directory.
    """
    candidates = _maps_dir_candidates()
    for path in candidates:
        if (path / _CATALOG).is_file():
            return path
    return candidates[0]


#: Where every map and the map catalog live.
MAPS_DIR = _maps_dir()

#: The map used when a CLI is given no argument.
DEFAULT_MAP = MAPS_DIR / "Test.kml"

__all__ = ["MAPS_DIR", "DEFAULT_MAP"]
