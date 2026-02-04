import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
import numpy as np

from simulation.sailboat_simulation import SailboatSimulation
from boat_msgs.msg import Wind, RudderAngle, SailAngle, Heading, BoatInfo, EnvironmentCreation, Route
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

    Parameters:
        render_mode (str): "none", "viewer" (live window), or "video" (record to file)
        video_output (str): Output path for video recording (only used when render_mode="video")
        video_fps (int): Frames per second for video recording (default: 30)
        video_width (int): Video width in pixels (default: 1280)
        video_height (int): Video height in pixels (default: 720)
    """

    def __init__(self):
        super().__init__('simulation_node')

        # Declare parameters for rendering
        self.declare_parameter('render_mode', 'video')  # "none", "viewer", or "video"
        self.declare_parameter('video_output', 'simulation.mp4')
        self.declare_parameter('video_fps', 30)
        self.declare_parameter('video_width', 1280)
        self.declare_parameter('video_height', 720)

        self.render_mode = self.get_parameter('render_mode').value
        self.video_output = self.get_parameter('video_output').value
        self.video_fps = self.get_parameter('video_fps').value
        self.video_width = self.get_parameter('video_width').value
        self.video_height = self.get_parameter('video_height').value

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

        # Track time for physics stepping
        self.last_time = self.get_clock().now()
        self.physics_dt = self.sim.model.opt.timestep

        # Rendering state
        self._init_rendering()

        # Timer at 60 Hz (suitable for rendering later)
        self.timer = self.create_timer(1.0 / 60.0, self.timer_callback)

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
            self.viewer.sync()

        elif self.render_mode == 'video':
            if self.video_writer is None:
                return

            # Rate limit frame capture
            now = time.time()
            if self.last_render_time is not None:
                if now - self.last_render_time < self.render_interval:
                    return
            self.last_render_time = now

            # Render using chase_cam (attached to boat, follows automatically)
            self.renderer.update_scene(self.sim.data, self.camera)
            frame = self.renderer.render()
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

    def timer_callback(self):
        # Step physics to match real-time
        now = self.get_clock().now()
        elapsed = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        steps = int(elapsed / self.physics_dt)
        for _ in range(steps):
            self.sim.step()

        # Render after physics step
        self._render()

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

    def _on_reset_simulation(self, msg: Bool):
        if msg.data:
            self.sim.reset()

    def _on_create_environment(self, msg: EnvironmentCreation):
        # Convert buoys to tuples with optional color (east, north, r, g, b)
        buoys = [
            (buoy.east, buoy.north, buoy.color.r, buoy.color.g, buoy.color.b)
            for buoy in msg.buoys
        ]
        # Create a fresh simulation instance
        self.sim = SailboatSimulation(
            buoys=buoys,
            include_course_line=True,
            wind_direction_deg=msg.wind_direction,
            wind_speed=msg.wind_speed
        )
        self.sim.set_boat_position(msg.boat_east, msg.boat_north)
        # Reinitialize rendering with the new model, but keep recording
        self._init_rendering(clear_frames=True)
        self.get_logger().info(f'Environment created with {len(buoys)} buoys at boat position ({msg.boat_east}, {msg.boat_north})')

    def destroy_node(self):
        """Clean up resources before shutting down."""
        # Close video writer if recording
        self._close_video()

        # Close viewer if open
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        super().destroy_node()


def main(args=None):
    """Main function for the simulation node."""
    rclpy.init(args=args)
    node = SimulationNode()
    
    # Keep the node running
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()