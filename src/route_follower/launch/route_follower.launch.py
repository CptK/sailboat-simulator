from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    parameters_file = PathJoinSubstitution([
        FindPackageShare('route_follower'),
        'resource',
        'parameters.yaml',
    ])
    # The map whose land to keep clear of; empty (the default) disables the
    # check, which is what launch.py's open-water setup wants.
    map_path_arg = DeclareLaunchArgument(
        'map_path', default_value='',
        description="Map to keep clear of: catalog name, file in maps/, or path",
    )

    # True when the simulation publishes /clock (launch_graph.py): timers then
    # tick in simulated time.
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    route_follower_node = Node(
        package='route_follower',
        executable='route_follower_node',
        name='route_follower_node',
        output='screen',
        parameters=[parameters_file, {
            'map_path': LaunchConfiguration('map_path'),
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        }],
    )

    return LaunchDescription([
        map_path_arg,
        use_sim_time_arg,
        route_follower_node,
    ])
