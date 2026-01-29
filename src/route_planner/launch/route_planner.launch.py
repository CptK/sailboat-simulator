from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    parameters_file = PathJoinSubstitution([
        FindPackageShare('route_planner'),
        'resource',
        'parameters.yaml',
    ])

    route_planner_node = Node(
        package='route_planner',
        executable='route_planner_node',
        name='route_planner_node',
        output='screen',
        parameters=[parameters_file],
    )

    return LaunchDescription([
        route_planner_node,
    ])
