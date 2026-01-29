from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    controller_node = Node(
        package='controller',
        executable='controller_node',
        name='controller_node',
        output='screen',
    )

    return LaunchDescription([
        controller_node,
    ])
