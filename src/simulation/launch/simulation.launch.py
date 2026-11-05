from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # The KML whose land is drawn on the water. Empty by default: launch.py has
    # no map, and only launch_graph.py passes one through.
    map_path = DeclareLaunchArgument('map_path', default_value='')
    # Publish simulated time on /clock and record frame by frame; the other
    # nodes must then run with use_sim_time (see launch/launch_graph.py).
    publish_clock = DeclareLaunchArgument('publish_clock', default_value='false')
    # Simulated seconds per second of video, with publish_clock.
    video_speedup = DeclareLaunchArgument('video_speedup', default_value='1.0')
    # "video" renders while running; "record" saves the frames' state for
    # render_video to render afterwards; "none" / "viewer" as the node says.
    render_mode = DeclareLaunchArgument('render_mode', default_value='video')
    # Simulated seconds per wall-clock second, when not rendering live.
    realtime_factor = DeclareLaunchArgument('realtime_factor', default_value='1.0')
    # Where the recording (render_mode record) or the video (render_mode video)
    # is written; relative paths are relative to the working directory.
    recording_output = DeclareLaunchArgument(
        'recording_output', default_value='simulation_recording.npz')
    video_output = DeclareLaunchArgument('video_output', default_value='simulation.mp4')

    simulation_node = Node(
        package='simulation',
        executable='simulation_node',
        name='simulation_node',
        output='screen',
        parameters=[{
            'map_path': LaunchConfiguration('map_path'),
            'publish_clock': ParameterValue(
                LaunchConfiguration('publish_clock'), value_type=bool),
            'video_speedup': ParameterValue(
                LaunchConfiguration('video_speedup'), value_type=float),
            'render_mode': LaunchConfiguration('render_mode'),
            'realtime_factor': ParameterValue(
                LaunchConfiguration('realtime_factor'), value_type=float),
            'recording_output': LaunchConfiguration('recording_output'),
            'video_output': LaunchConfiguration('video_output'),
        }],
    )

    return LaunchDescription([
        map_path,
        publish_clock,
        video_speedup,
        render_mode,
        realtime_factor,
        recording_output,
        video_output,
        simulation_node,
    ])
