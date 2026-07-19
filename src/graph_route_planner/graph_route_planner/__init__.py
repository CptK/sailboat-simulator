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

from pathlib import Path


def _assets_dir() -> Path:
    """Locate the bundled KML maps.

    colcon installs the Python package and the assets into separate trees, so
    the source layout cannot simply be assumed. The source tree is preferred
    when present — covering both a plain checkout and a --symlink-install
    workspace — otherwise the package's share directory is used.
    """
    source = Path(__file__).resolve().parent.parent / "assets"
    if source.is_dir():
        return source

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("graph_route_planner")) / "assets"
    except Exception:      # not built yet, or ament unavailable
        return source


#: Bundled example maps.
MAPS_DIR = _assets_dir()

#: The map used when a CLI is given no argument.
DEFAULT_MAP = MAPS_DIR / "Test.kml"

__all__ = ["MAPS_DIR", "DEFAULT_MAP"]
