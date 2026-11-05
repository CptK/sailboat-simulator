from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution



def generate_launch_description():
    parameters_file = PathJoinSubstitution([
        FindPackageShare('controller'),
        'resource',
        'parameters.yaml',
    ])

    # True when the simulation publishes /clock (launch_graph.py): timers then
    # tick in simulated time.
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    controller_node = Node(
        package='controller',
        executable='controller_node',
        name='controller_node',
        output='screen',
        parameters=[parameters_file, {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        controller_node,
    ])
