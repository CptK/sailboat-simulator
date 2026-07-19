"""The full stack with a clean planner/steering split.

launch.py uses route_planner, which both plans and steers. This launch splits
those two jobs across two nodes:

    /planning/mission ──> graph_route_planner ──> /planning/target_route ──> route_follower ──> /planning/desired_heading

  * graph_route_planner  plans the route: it expands the mission's via points
    into a detailed path over the water, tacking upwind where needed.
  * route_follower        steers along that route exactly as given, without
    re-planning it, executing tacks and jibes as legs cross the wind.

So route_planner is not used here at all — route_follower replaces its steering
role, and graph_route_planner replaces its planning role. The simulation and
controller are the same as in launch.py.
"""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch.actions import IncludeLaunchDescription

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('simulation'),
                'launch',
                'simulation.launch.py',
            ])
        )
    )
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('controller'),
                'launch',
                'controller.launch.py',
            ])
        )
    )
    route_follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('route_follower'),
                'launch',
                'route_follower.launch.py',
            ])
        )
    )

    graph_parameters_file = PathJoinSubstitution([
        FindPackageShare('graph_route_planner'),
        'resource',
        'parameters.yaml',
    ])
    open_water_map = PathJoinSubstitution([
        FindPackageShare('graph_route_planner'),
        'assets',
        'OpenWater.kml',
    ])

    graph_route_planner_node = Node(
        package='graph_route_planner',
        executable='graph_route_planner_node',
        name='graph_route_planner_node',
        output='screen',
        parameters=[
            graph_parameters_file,
            {
                # The simulation's world is open water, not the bundled pond.
                'map_path': open_water_map,
                # Coarser than the pond default: this map is 160x160 m, and the
                # grid is laid over the whole of it, so cost grows with the area.
                'grid_spacing': 8.0,
            },
        ],
    )

    return LaunchDescription([
        simulation_launch,
        controller_launch,
        route_follower_launch,
        graph_route_planner_node,
    ])
