"""The full stack, but with the graph planner expanding the mission.

Same as launch.py — simulation, controller, route_planner — plus
graph_route_planner sitting in front of route_planner:

    /planning/mission ──> graph_route_planner ──> /planning/target_route ──> route_planner

launch.py has whoever sets the course publish /planning/target_route directly,
so route_planner sails straight between the via points. Here the via points go
to /planning/mission instead, and the graph planner works out the water between
them — tacking upwind where a direct heading is unsailable.
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
    route_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('route_planner'),
                'launch',
                'route_planner.launch.py',
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
        route_planner_launch,
        graph_route_planner_node,
    ])
