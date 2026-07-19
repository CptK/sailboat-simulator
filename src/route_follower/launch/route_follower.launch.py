from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    parameters_file = PathJoinSubstitution([
        FindPackageShare('route_follower'),
        'resource',
        'parameters.yaml',
    ])

    route_follower_node = Node(
        package='route_follower',
        executable='route_follower_node',
        name='route_follower_node',
        output='screen',
        parameters=[parameters_file],
    )

    return LaunchDescription([
        route_follower_node,
    ])
