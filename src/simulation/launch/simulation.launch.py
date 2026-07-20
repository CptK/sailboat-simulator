from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # The KML whose land is drawn on the water. Empty by default: launch.py has
    # no map, and only launch_graph.py passes one through.
    map_path = DeclareLaunchArgument('map_path', default_value='')

    simulation_node = Node(
        package='simulation',
        executable='simulation_node',
        name='simulation_node',
        output='screen',
        parameters=[{'map_path': LaunchConfiguration('map_path')}],
    )

    return LaunchDescription([
        map_path,
        simulation_node,
    ])
