import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
import numpy as np

from simulation.sailboat_simulation import SailboatSimulation
from simulation.minimap import Minimap
from simulation.recording import Recording, Scene
from simulation.video_overlay import COLLISION_TEXT, draw_label, inset_size, picture_in_picture
from boat_msgs.msg import Wind, RudderAngle, SailAngle, Heading, BoatInfo, EnvironmentCreation, Route
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool

class SimulationNode(Node):
    """
    ROS2 node that runs the MuJoCo sailboat simulation.

    All published values use DEGREES for angles (converted from internal radians).
    All subscribed commands expect DEGREES.

    Subscribes to:
        /control/rudder (RudderAngle): Rudder command in degrees [-45, 45]
        /control/sail (SailAngle): Sail angle command in degrees [0, 90]
        /env/wind (Wind): External wind override in degrees [0, 360)
        /control/reset_simulation (Bool): Reset simulation to initial state

    Publishes:
        /sense/boat_info (BoatInfo): Full boat state (degrees, m, m/s)
        /sense/wind (Wind): Current wind conditions in degrees [0, 360)
        /sense/heading (Heading): Current heading in degrees [0, 360)
        /sense/collision (Bool): True while the boat is touching land
        /clock (Clock): Simulated time, only when publish_clock is true

    Time:
        With publish_clock false, physics follows the wall clock (real time,
        or realtime_factor times it).

        With publish_clock true, this node is the ROS clock: it publishes the
        simulated time on /clock, and the other nodes are expected to run with
        use_sim_time. While recording video it then steps physics in lockstep
        with the frames: every frame advances video_speedup / video_fps
        seconds of simulated time, however long that frame takes to render.
        The video therefore plays at a constant speed (real speed at
        video_speedup 1), and the run simply takes as long as rendering needs.
        Without video it follows the wall clock, as with publish_clock false.

    Recording (render_mode "record"):
        Nothing is rendered live. Every 1/record_rate simulated seconds the
        node records what a frame would show (see simulation.recording), and
        saves it to recording_output on shutdown. render_video then renders
        the video from it, in parallel and at any speedup. Physics follows the
        wall clock, so the control loop runs at its full rate.

    Parameters:
        render_mode (str): "none", "viewer" (live window), "video" (render to a
            file while running) or "record" (record, render afterwards)
        recording_output (str): Where render_mode "record" saves the recording
            (default: simulation_recording.npz)
        record_rate (float): Frames recorded per simulated second (default: 30)
        realtime_factor (float): Simulated seconds per wall-clock second when
            physics follows the wall clock (default: 1.0)
        video_output (str): Output path for video recording (only used when render_mode="video")
        video_fps (int): Frames per second for video recording (default: 30)
        video_width (int): Video width in pixels (default: 1280)
        video_height (int): Video height in pixels (default: 720)
        onboard_inset_scale (float): Width of the onboard-camera inset in the
            bottom-right corner of the video, as a fraction of the video width
            (default: 0.22). 0 disables the inset.
        minimap_inset_scale (float): Width of the top-down minimap inset in the
            bottom-left corner of the video, as a fraction of the video width
            (default: 0.22). 0 disables it.
        minimap_view_size (float): Width in metres of the area the minimap
            shows around the boat, which it follows (default: 300). A map no
            wider than this is shown whole. 0 always shows the whole map.
        publish_clock (bool): Publish simulated time on /clock and, while
            recording, step physics frame by frame (default: false). See Time.
        video_speedup (float): Simulated seconds per second of video when
            publish_clock is true (default: 1.0, real speed). 2.0 makes a
            constant 2x timelapse that also renders half as many frames.
    """

    def __init__(self):
        super().__init__('simulation_node')

        # Declare parameters for rendering
        self.declare_parameter('render_mode', 'video')  # "none", "viewer", "video" or "record"
        self.declare_parameter('recording_output', 'simulation_recording.npz')
        self.declare_parameter('record_rate', 30.0)
        self.declare_parameter('realtime_factor', 1.0)
        self.declare_parameter('video_output', 'simulation.mp4')
        self.declare_parameter('video_fps', 30)
        self.declare_parameter('video_width', 1280)
        self.declare_parameter('video_height', 720)
        # Picture-in-picture of the boat's own camera, like a video call self-view
        self.declare_parameter('onboard_inset_scale', 0.22)
        # Top-down minimap of the course (map, route, buoys, track, boat)
        self.declare_parameter('minimap_inset_scale', 0.22)
        self.declare_parameter('minimap_view_size', 300.0)
        # Map whose land is drawn on the water and collided with: a name from
        # maps/catalog.yaml, a KML file in maps/, or a path. Empty means open
        # water, which is what launch.py wants — only the graph planner has a map.
        self.declare_parameter('map_path', '')
        # Own the ROS clock and record frame by frame (see the class docstring)
        self.declare_parameter('publish_clock', False)
        self.declare_parameter('video_speedup', 1.0)

        self.render_mode = self.get_parameter('render_mode').value
        self.video_output = self.get_parameter('video_output').value
        self.video_fps = self.get_parameter('video_fps').value
        self.video_width = self.get_parameter('video_width').value
        self.video_height = self.get_parameter('video_height').value
        self.onboard_inset_scale = float(self.get_parameter('onboard_inset_scale').value)
        self.minimap_inset_scale = float(self.get_parameter('minimap_inset_scale').value)
        self.minimap_view_size = float(self.get_parameter('minimap_view_size').value)
        self.map_water = None
        self.minimap = None
        self.map_path = self.get_parameter('map_path').value
        self.publish_clock = bool(self.get_parameter('publish_clock').value)
        self.video_speedup = float(self.get_parameter('video_speedup').value)
        if self.video_speedup <= 0:
            raise ValueError(f'video_speedup must be positive, got {self.video_speedup}')
        self.realtime_factor = float(self.get_parameter('realtime_factor').value)
        if self.realtime_factor <= 0:
            raise ValueError(f'realtime_factor must be positive, got {self.realtime_factor}')
        self.recording_output = self.get_parameter('recording_output').value
        self.record_rate = float(self.get_parameter('record_rate').value)
        if self.record_rate <= 0:
            raise ValueError(f'record_rate must be positive, got {self.record_rate}')
        self.obstacles = self._load_obstacles()

        self.sim = SailboatSimulation(wind_speed=6.0)

        # Initialize video writer
        self.video_writer = None

        # Subscribe to control topics with only one buffered message
        self.create_subscription(RudderAngle, '/control/rudder', self._on_rudder_control, 1)
        self.create_subscription(SailAngle, '/control/sail', self._on_sail_control, 1)
        self.create_subscription(Wind, '/env/wind', self._on_wind_update, 1)
        self.create_subscription(Bool, '/env/reset_simulation', self._on_reset_simulation, 1)
        self.create_subscription(EnvironmentCreation, '/env/create', self._on_create_environment, 1)
        self.create_subscription(Heading, '/planning/desired_heading', self._on_desired_heading, 1)
        self.create_subscription(Route, '/planning/current_route', self._on_current_route, 1)
        self.create_subscription(Bool, '/planning/route_completed', self._on_route_completed, 1)

        # Publish boat information
        self.heading_publisher = self.create_publisher(Heading, '/sense/heading', 1)
        self.wind_publisher = self.create_publisher(Wind, '/sense/wind', 1)
        self.boat_info_publisher = self.create_publisher(BoatInfo, '/sense/boat_info', 1)
        # True while the boat is touching land (it cannot sail through it)
        self.collision_publisher = self.create_publisher(Bool, '/sense/collision', 1)
        self.was_colliding = False

        # Track time for physics stepping. The wall clock is read with
        # time.monotonic(), not this node's ROS clock: with publish_clock this
        # node *is* the ROS clock, and must not wait on itself.
        self.last_wall_time = time.monotonic()
        self.physics_dt = self.sim.model.opt.timestep
        # Time owed to physics that is less than one timestep, carried over to
        # the next tick so that no simulated time is lost to rounding.
        self.pending_time = 0.0
        # Simulated time since start, published on /clock. Kept here rather
        # than read from the MuJoCo model, which is rebuilt (and its time
        # reset to 0) by /env/create; the ROS clock must never run backwards.
        self.sim_time = 0.0
        self.clock_publisher = (self.create_publisher(Clock, '/clock', 10)
                                if self.publish_clock else None)
        self._publish_clock()

        # Recording (render_mode "record"): segments start with /env/create
        self.recording = None
        self.next_record_time = 0.0
        if self.render_mode == 'record':
            self.recording = Recording(
                map_path=self.map_path,
                rate=self.record_rate,
                render={
                    'fps': self.video_fps,
                    'width': self.video_width,
                    'height': self.video_height,
                    'onboard_inset_scale': self.onboard_inset_scale,
                    'minimap_inset_scale': self.minimap_inset_scale,
                    'minimap_view_size': self.minimap_view_size,
                },
            )
            self.get_logger().info(
                f'Recording at {self.record_rate:g} frames per simulated second '
                f'to {self.recording_output}; render it with render_video'
            )

        # Rendering state
        self._init_rendering()

        # Timer at 60 Hz (suitable for rendering later)
        self.timer = self.create_timer(1.0 / 60.0, self.timer_callback)

    def _load_obstacles(self) -> list:
        """Land polygons from the configured map, for drawing on the water.

        Loads the map exactly as the graph planner does (by catalog name, file
        name in maps/, or path — with the catalog's crop and scale), so what is
        rendered is exactly what the planner refuses to cross. Land is
        everything in the simulated world that is not navigable water, and the
        world is sized to hold the map (self.world_half_size).

        Returns:
            Shapely Polygons in local (east, north) metres. Empty if no map is
            configured, or if the map could not be read — a missing map costs
            the picture, not the simulation, so it warns rather than raising.
        """
        self.world_half_size = None
        if not self.map_path:
            return []

        try:
            from simulation.build_env import load_map_world

            map_name, self.map_water, self.world_half_size, land = load_map_world(
                self.map_path)
        except Exception as err:
            self.get_logger().warn(
                f"Could not load map '{self.map_path}': {err}. "
                f"The simulation runs without it."
            )
            self.world_half_size = None
            self.map_water = None
            return []

        self.get_logger().info(
            f"Loaded map '{map_name}': {len(land)} land polygon(s), "
            f"world {2 * self.world_half_size:.0f} m across"
        )
        return land

    def _init_rendering(self, clear_frames: bool = True):
        """Initialize rendering based on render_mode parameter.

        Args:
            clear_frames: If True, closes existing video writer. Set to False when
                          reinitializing after environment change to keep recording.
        """
        # Close existing viewer if any
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        self.renderer = None
        self.camera = None

        # Close video writer if clearing frames
        if clear_frames and hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        self.last_render_time = None
        self.render_interval = 1.0 / self.video_fps if self.video_fps > 0 else 0.033

        if self.render_mode == 'viewer':
            self.get_logger().info('Initializing MuJoCo viewer (live rendering)')
            self.viewer = mujoco.viewer.launch_passive(
                self.sim.model,
                self.sim.data
            )
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 15
            self.viewer.cam.lookat[:] = [0, 0, 0]

        elif self.render_mode == 'video':
            self.get_logger().info(
                f'Initializing video recording: {self.video_width}x{self.video_height} @ {self.video_fps}fps'
            )
            self.renderer = mujoco.Renderer(
                self.sim.model,
                width=self.video_width,
                height=self.video_height
            )
            # Use the chase_cam attached to the boat (follows automatically)
            self.camera = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "chase_cam")

            # Initialize video writer for streaming to disk
            if not hasattr(self, 'video_writer') or self.video_writer is None:
                try:
                    import imageio
                    self.video_writer = imageio.get_writer(
                        self.video_output,
                        fps=self.video_fps,
                        codec='libx264',
                        quality=8,
                        pixelformat='yuv420p',
                        macro_block_size=None
                    )
                    self.get_logger().info(f'Video writer initialized: {self.video_output}')
                except ImportError:
                    self.get_logger().error('imageio not installed. Cannot record video.')
                    self.video_writer = None
                except Exception as e:
                    self.get_logger().error(f'Failed to initialize video writer: {e}')
                    self.video_writer = None

    def _render(self):
        """Render a frame based on current render_mode."""
        if self.render_mode == 'none':
            return

        # Get boat position for camera tracking
        boat_x = self.sim.data.qpos[self.sim.qadr_x]
        boat_y = self.sim.data.qpos[self.sim.qadr_y]

        if self.render_mode == 'viewer':
            if self.viewer is None or not self.viewer.is_running():
                return
            # Camera follows boat
            self.viewer.cam.lookat[0] = boat_x
            self.viewer.cam.lookat[1] = boat_y
            if self.sim.in_collision():
                self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                       mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                       COLLISION_TEXT, ""))
            else:
                self.viewer.clear_texts()
            self.viewer.sync()

        elif self.render_mode == 'video':
            if self.video_writer is None:
                return

            # In lockstep every tick is exactly one frame. Otherwise physics
            # follows the wall clock, so rate limit frame capture to it.
            if not self._lockstep():
                now = time.monotonic()
                if self.last_render_time is not None:
                    if now - self.last_render_time < self.render_interval:
                        return
                self.last_render_time = now

            # Render using chase_cam (attached to boat, follows automatically)
            self.renderer.update_scene(self.sim.data, self.camera)
            frame = self.renderer.render()
            if self.onboard_inset_scale > 0:
                inset_w, inset_h = inset_size(self.video_width, self.onboard_inset_scale)
                inset = self.sim.render_onboard(inset_w, inset_h)
                frame = picture_in_picture(frame, inset)
            if self.minimap is not None:
                minimap = self.minimap.render(
                    boat_x, boat_y, self.sim.data.qpos[self.sim.qadr_yaw])
                frame = picture_in_picture(frame, minimap, corner="bottom-left")
            if self.sim.in_collision():
                frame = draw_label(frame, COLLISION_TEXT)
            # Write frame directly to disk instead of buffering in memory
            try:
                self.video_writer.append_data(frame)
            except Exception as e:
                self.get_logger().error(f'Failed to write video frame: {e}')

    def _close_video(self):
        """Close video writer and finalize video file."""
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            try:
                self.get_logger().info(f'Closing video file: {self.video_output}')
                self.video_writer.close()
                self.video_writer = None
                self.get_logger().info(f'Video saved: {self.video_output}')
            except Exception as e:
                self.get_logger().error(f'Failed to close video writer: {e}')

    def _lockstep(self) -> bool:
        """True when physics advances one video frame per tick (see Time)."""
        return (self.publish_clock and self.render_mode == 'video'
                and self.video_writer is not None)

    def _advance(self, duration: float):
        """Step physics by ``duration`` seconds, carrying any remainder over."""
        self.pending_time += duration
        steps = int(self.pending_time / self.physics_dt)
        self.pending_time -= steps * self.physics_dt
        for _ in range(steps):
            self.sim.step()
            self.sim_time += self.physics_dt
            self._record_frame_if_due()

    def _record_frame_if_due(self):
        """Record a frame each time simulated time passes 1/record_rate."""
        if self.recording is None or self.sim_time < self.next_record_time:
            return
        self.recording.add_frame(self.sim_time, self.sim.data, self.sim.in_collision())
        interval = 1.0 / self.record_rate
        # Schedule on a fixed grid, so frames stay evenly spaced
        while self.next_record_time <= self.sim_time:
            self.next_record_time += interval

    def _publish_clock(self):
        if self.clock_publisher is None:
            return
        seconds = int(self.sim_time)
        nanoseconds = int(round((self.sim_time - seconds) * 1e9))
        if nanoseconds >= 1_000_000_000:
            seconds, nanoseconds = seconds + 1, nanoseconds - 1_000_000_000
        self.clock_publisher.publish(Clock(clock=Time(sec=seconds, nanosec=nanoseconds)))

    def timer_callback(self):
        now = time.monotonic()
        elapsed = now - self.last_wall_time
        self.last_wall_time = now

        if self._lockstep():
            # One frame's worth of simulated time, however long rendering takes
            self._advance(self.video_speedup / self.video_fps)
        else:
            # Keep up with the wall clock
            self._advance(elapsed * self.realtime_factor)

        # Publish the new time and state before rendering, so that the other
        # nodes compute their response while this frame renders and it is
        # applied from the next physics step on.
        self._publish_clock()
        self._publish_state()
        self._render()

    def _publish_state(self):
        """Publish wind, boat state and collision for the current physics state."""
        status = self.sim.get_status()

        # Publish current wind information (degrees from get_status())
        self.wind_publisher.publish(Wind(angle=status['wind_dir'], speed=status['wind_speed']))

        # Publish current boat information (all angles in degrees from get_status())
        self.boat_info_publisher.publish(BoatInfo(
            x=status['x'],
            y=status['y'],
            velocity_x=status['vel_x'],
            velocity_y=status['vel_y'],
            velocity_magnitude=status['speed'],
            yaw_rate=status['yaw_rate'],
            heading=status['heading'],
            rudder_angle=status['rudder'],
            sail_angle=status['sail_angle'],
        ))
        self._report_collision(status)

    def _report_collision(self, status: dict):
        """Publish the collision state, and log when the boat runs aground or frees itself."""
        colliding = bool(status['collision'])
        self.collision_publisher.publish(Bool(data=colliding))
        if colliding != self.was_colliding:
            where = f"({status['x']:.1f}, {status['y']:.1f})"
            if colliding:
                self.get_logger().warn(f"Collision: boat ran aground at {where}")
            else:
                self.get_logger().info(f"Boat is clear of land again at {where}")
        self.was_colliding = colliding

    def _on_rudder_control(self, msg: RudderAngle):
        # msg.angle in degrees [-45, 45]
        self.sim.set_rudder(msg.angle)

    def _on_sail_control(self, msg: SailAngle):
        # msg.sail_angle in degrees [0, 90]
        self.sim.set_sail_angle(msg.sail_angle)

    def _on_wind_update(self, msg: Wind):
        # msg.angle in degrees [0, 360), msg.speed in m/s
        self.sim.set_wind(direction_deg=msg.angle, speed=msg.speed)

    def _on_desired_heading(self, msg: Heading):
        # msg.heading in degrees [0, 360) using nautical convention (0=North, 90=East)
        # Convert to mathematical convention (0=East, 90=North) for visualization
        math_heading_deg = 90 - msg.heading
        self.sim.update_course_line(np.deg2rad(math_heading_deg))

    def _on_current_route(self, msg: Route):
        # Convert waypoints to list of (east, north) tuples
        waypoints = [(wp.east, wp.north) for wp in msg.waypoints]

        # Prepend current boat position to show the complete path from boat to destination
        boat_east = self.sim.data.qpos[self.sim.qadr_x]
        boat_north = self.sim.data.qpos[self.sim.qadr_y]
        waypoints_with_current = [(boat_east, boat_north)] + waypoints

        self.sim.update_route(waypoints_with_current)
        if self.minimap is not None:
            self.minimap.set_route(waypoints_with_current)
        if self.recording is not None:
            self.recording.set_route(waypoints_with_current)

        if hasattr(self, 'current_route'):
            if waypoints != self.current_route:
                self.get_logger().info(f'Updated route with {len(waypoints)} waypoints.')
                for wp in waypoints:
                    self.get_logger().info(f'  Waypoint: east={wp[0]}, north={wp[1]}')
        else:
            self.get_logger().info(f'Set initial route with {len(waypoints)} waypoints.')
            for wp in waypoints:
                self.get_logger().info(f'  Waypoint: east={wp[0]}, north={wp[1]}')

        self.current_route = waypoints

    def _on_route_completed(self, msg: Bool):
        if msg.data:
            self.get_logger().info('Route completed, clearing route visualization')
            self.sim.update_route([])
            if self.minimap is not None:
                self.minimap.set_route([])
            if self.recording is not None:
                self.recording.set_route([])

    def _on_reset_simulation(self, msg: Bool):
        if msg.data:
            self.sim.reset()

    def _on_create_environment(self, msg: EnvironmentCreation):
        # Convert buoys to tuples with optional color (east, north, r, g, b)
        buoys = [
            (buoy.east, buoy.north, buoy.color.r, buoy.color.g, buoy.color.b)
            for buoy in msg.buoys
        ]
        # Create a fresh simulation instance (releasing the old one's renderers)
        self.sim.close()
        self.sim = SailboatSimulation(
            buoys=buoys,
            include_course_line=True,
            wind_direction_deg=msg.wind_direction,
            wind_speed=msg.wind_speed,
            obstacles=self.obstacles,
            world_half_size=self.world_half_size,
        )
        self.sim.set_boat_position(msg.boat_east, msg.boat_north)
        if self.render_mode == 'video' and self.minimap_inset_scale > 0:
            width, height = inset_size(self.video_width, self.minimap_inset_scale)
            self.minimap = Minimap(width, height, water=self.map_water,
                                   boat=(msg.boat_east, msg.boat_north), buoys=buoys,
                                   view_size=self.minimap_view_size)
        if self.recording is not None:
            self.recording.start_segment(Scene(
                buoys=[tuple(float(v) for v in b) for b in buoys],
                boat_east=float(msg.boat_east),
                boat_north=float(msg.boat_north),
                wind_direction=float(msg.wind_direction),
                wind_speed=float(msg.wind_speed),
            ))
        # Reinitialize rendering with the new model, but keep recording
        self._init_rendering(clear_frames=True)
        self.get_logger().info(f'Environment created with {len(buoys)} buoys at boat position ({msg.boat_east}, {msg.boat_north})')

    def _save_recording(self):
        """Write the recording (render_mode "record") to recording_output."""
        if self.recording is None:
            return
        if not self.recording.segments:
            self.get_logger().warn('Nothing recorded: no environment was created (/env/create)')
            return
        try:
            self.recording.save(self.recording_output)
            self.get_logger().info(
                f'Recording saved: {self.recording_output} '
                f'({self.recording.frame_count} frames, {self.sim_time:.0f} s simulated)'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to save recording: {e}')

    def destroy_node(self):
        """Clean up resources before shutting down."""
        # Tear down the ROS side first. On Ctrl+C the context is usually
        # already shut down, and while the node exists its logger still tries
        # to publish to /rosout ("publisher's context is invalid"); once it is
        # destroyed, the log lines below go to the console only.
        super().destroy_node()

        # Close video writer if recording
        self._close_video()
        self._save_recording()

        # Close viewer if open
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def main(args=None):
    """Main function for the simulation node."""
    rclpy.init(args=args)
    node = SimulationNode()
    
    # Keep the node running
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    # Shutdown is triggered by SIGINT, and it often arrives twice: once from
    # whoever stopped the run (e.g. the run scripts signal the whole process
    # group) and once forwarded by ros2 launch. A second one landing while the
    # video is being finalized would abort the write and leave an unplayable
    # file, so ignore further interrupts while cleaning up.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    node.destroy_node()
    rclpy.try_shutdown()  # the interrupt may already have shut the context down


if __name__ == '__main__':
    main()