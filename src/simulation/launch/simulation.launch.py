from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    simulation_node = Node(
        package='simulation',
        executable='simulation_node',
        name='simulation_node',
        output='screen',
    )

    return LaunchDescription([
        simulation_node,
    ])
