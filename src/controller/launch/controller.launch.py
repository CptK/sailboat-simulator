from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution



def generate_launch_description():
    parameters_file = PathJoinSubstitution([
        FindPackageShare('controller'),
        'resource',
        'parameters.yaml',
    ])

    controller_node = Node(
        package='controller',
        executable='controller_node',
        name='controller_node',
        output='screen',
        parameters=[parameters_file],
    )

    return LaunchDescription([
        controller_node,
    ])
