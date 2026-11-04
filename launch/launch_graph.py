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

Launch arguments:
    map           A map name from maps/catalog.yaml (e.g. landgraben, kornaten),
                  a KML file in maps/, or a path. Default openwater.
    grid_spacing  The planner's tacking-grid spacing in metres. Default 10.0,
                  sized for the 240 m open-water map; small maps like the
                  Landgraben pond want something finer (around 6), large
                  archipelagos coarser (around 20).
    max_leg_distance  Longest leg the planner considers, in metres; 0 (the
                  default) derives it from the map. Large maps need a limit
                  (100-150) to build their visibility graph in seconds.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch.actions import IncludeLaunchDescription

    # One map, two consumers: the planner routes around its land, and the
    # simulation draws that same land on the water. Passing the same path to
    # both is what keeps the picture honest — if the two ever disagree about
    # the frame, the boat will visibly sail over an obstacle instead of it
    # going unnoticed.
    #
    # Both nodes resolve the name the same way (map_loader.load_map), crop and
    # scale included, so it is passed through as given.
    map_arg = DeclareLaunchArgument(
        'map', default_value='openwater',
        description='Map: a name from maps/catalog.yaml, a KML file in maps/, or a path',
    )
    grid_spacing_arg = DeclareLaunchArgument(
        'grid_spacing', default_value='10.0',
        description="Planner tacking-grid spacing in metres",
    )
    max_leg_arg = DeclareLaunchArgument(
        'max_leg_distance', default_value='0.0',
        description="Planner's longest leg in metres; 0 derives it from the map",
    )
    map_path = LaunchConfiguration('map')

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('simulation'),
                'launch',
                'simulation.launch.py',
            ])
        ),
        launch_arguments={'map_path': map_path}.items(),
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

    graph_route_planner_node = Node(
        package='graph_route_planner',
        executable='graph_route_planner_node',
        name='graph_route_planner_node',
        output='screen',
        parameters=[
            graph_parameters_file,
            {
                # The same map the simulation draws.
                'map_path': map_path,
                # Planning cost grows with the square of map size / spacing, so
                # large maps want a coarser grid than the pond.
                'grid_spacing': ParameterValue(
                    LaunchConfiguration('grid_spacing'), value_type=float),
                'max_leg_distance': ParameterValue(
                    LaunchConfiguration('max_leg_distance'), value_type=float),
            },
        ],
    )

    return LaunchDescription([
        map_arg,
        grid_spacing_arg,
        max_leg_arg,
        simulation_launch,
        controller_launch,
        route_follower_launch,
        graph_route_planner_node,
    ])
