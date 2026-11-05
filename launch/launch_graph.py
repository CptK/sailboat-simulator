"""The full stack with a clean planner/steering split.

launch.py uses route_planner, which both plans and steers. This launch splits
those two jobs across two nodes:

    /planning/mission ──> graph_route_planner ──> /planning/target_route ──> route_follower ──> /planning/desired_heading

  * graph_route_planner  plans the route: it expands the mission's via points
    into a detailed path over the water, tacking upwind where needed.
  * route_follower        steers along that route exactly as given, without
    re-planning it, executing tacks and jibes as legs cross the wind.

So route_planner is not used here at all — route_follower replaces its steering
role, and graph_route_planner replaces its planning role. The simulation and
controller are the same as in launch.py.

Launch arguments:
    map           A map name from maps/catalog.yaml (e.g. landgraben, kornaten),
                  a KML file in maps/, or a path. Default openwater.
    grid_spacing  The planner's tacking-grid spacing in metres. Default 10.0,
                  sized for the 240 m open-water map; small maps want
                  something finer (around 6), large
                  archipelagos coarser (around 20).
    max_leg_distance  Longest leg the planner considers, in metres; 0 (the
                  default) derives it from the map. Large maps need a limit
                  (100-150) to build their visibility graph in seconds.
    render_mode   "record" (default): render nothing while running; the
                  simulation saves simulation_recording.npz on shutdown, and
                  render_video renders the video from it afterwards, in
                  parallel (run_simulation_graph.py does both). "video":
                  render while running (slow without a GPU, see Time).
                  "none": no video.
    realtime_factor  Simulated seconds per wall-clock second, unless rendering
                  live. Default 1.0.
    video_speedup Only for render_mode video: simulated seconds per second of
                  video. Default 1.0 (real speed).
    recording_output, video_output  Where the recording / live video is
                  written. Defaults simulation_recording.npz / simulation.mp4
                  in the working directory; run_simulation_graph.py points
                  them into the run's directory under runs/.

Time: the simulation is the ROS clock. It publishes simulated time on /clock
and every other node runs with use_sim_time, so their timers tick in simulated
time; the simulation itself keeps the wall clock. Without live rendering it
runs at realtime_factor times the wall clock. With render_mode video, it
advances one frame's worth of simulated time per rendered frame, however long
rendering takes. That keeps the video at a constant speed: on a CPU-only
machine rendering is slower than real time, so with a wall clock the video sped
up and slowed down with the render load.

use_sim_time is passed to each node rather than set for all of them with
SetParameter: launch_ros then drops the nodes' own launch parameters, such as
map_path.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch.actions import IncludeLaunchDescription

    # One map, two consumers: the planner routes around its land, and the
    # simulation draws that same land on the water. Passing the same path to
    # both is what keeps the picture honest — if the two ever disagree about
    # the frame, the boat will visibly sail over an obstacle instead of it
    # going unnoticed.
    #
    # Both nodes resolve the name the same way (map_loader.load_map), crop and
    # scale included, so it is passed through as given.
    map_arg = DeclareLaunchArgument(
        'map', default_value='openwater',
        description='Map: a name from maps/catalog.yaml, a KML file in maps/, or a path',
    )
    grid_spacing_arg = DeclareLaunchArgument(
        'grid_spacing', default_value='10.0',
        description="Planner tacking-grid spacing in metres",
    )
    max_leg_arg = DeclareLaunchArgument(
        'max_leg_distance', default_value='0.0',
        description="Planner's longest leg in metres; 0 derives it from the map",
    )
    video_speedup_arg = DeclareLaunchArgument(
        'video_speedup', default_value='1.0',
        description='render_mode video: simulated seconds per second of video',
    )
    render_mode_arg = DeclareLaunchArgument(
        'render_mode', default_value='record',
        description='record (render afterwards with render_video), video, or none',
    )
    realtime_factor_arg = DeclareLaunchArgument(
        'realtime_factor', default_value='1.0',
        description='Simulated seconds per wall-clock second, unless rendering live',
    )
    recording_output_arg = DeclareLaunchArgument(
        'recording_output', default_value='simulation_recording.npz',
        description='render_mode record: where the recording is saved',
    )
    video_output_arg = DeclareLaunchArgument(
        'video_output', default_value='simulation.mp4',
        description='render_mode video: where the video is saved',
    )
    map_path = LaunchConfiguration('map')

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('simulation'),
                'launch',
                'simulation.launch.py',
            ])
        ),
        launch_arguments={
            'map_path': map_path,
            # The simulation is the clock; see "Time" above.
            'publish_clock': 'true',
            'video_speedup': LaunchConfiguration('video_speedup'),
            'render_mode': LaunchConfiguration('render_mode'),
            'realtime_factor': LaunchConfiguration('realtime_factor'),
            'recording_output': LaunchConfiguration('recording_output'),
            'video_output': LaunchConfiguration('video_output'),
        }.items(),
    )
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('controller'),
                'launch',
                'controller.launch.py',
            ])
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )
    route_follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('route_follower'),
                'launch',
                'route_follower.launch.py',
            ])
        ),
        # The follower keeps clear of the same land the planner routes around.
        launch_arguments={'map_path': map_path, 'use_sim_time': 'true'}.items(),
    )

    graph_parameters_file = PathJoinSubstitution([
        FindPackageShare('graph_route_planner'),
        'resource',
        'parameters.yaml',
    ])

    graph_route_planner_node = Node(
        package='graph_route_planner',
        executable='graph_route_planner_node',
        name='graph_route_planner_node',
        output='screen',
        parameters=[
            graph_parameters_file,
            {
                # The same map the simulation draws.
                'map_path': map_path,
                # Planning cost grows with the square of map size / spacing, so
                # large maps want a coarser grid than the pond.
                'grid_spacing': ParameterValue(
                    LaunchConfiguration('grid_spacing'), value_type=float),
                'max_leg_distance': ParameterValue(
                    LaunchConfiguration('max_leg_distance'), value_type=float),
                # Timers tick in simulated time; see "Time" above.
                'use_sim_time': True,
            },
        ],
    )

    return LaunchDescription([
        map_arg,
        grid_spacing_arg,
        max_leg_arg,
        video_speedup_arg,
        render_mode_arg,
        realtime_factor_arg,
        recording_output_arg,
        video_output_arg,
        simulation_launch,
        controller_launch,
        route_follower_launch,
        graph_route_planner_node,
    ])
