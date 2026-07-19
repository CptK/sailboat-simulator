from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    parameters_file = PathJoinSubstitution([
        FindPackageShare('graph_route_planner'),
        'resource',
        'parameters.yaml',
    ])

    graph_route_planner_node = Node(
        package='graph_route_planner',
        executable='graph_route_planner_node',
        name='graph_route_planner_node',
        output='screen',
        parameters=[parameters_file],
    )

    return LaunchDescription([
        graph_route_planner_node,
    ])
