import rclpy
from rclpy.node import Node
import numpy as np

from boat_msgs.msg import BoatInfo, RudderAngle, SailAngle, Wind, Heading
from controller.pid import PIDController
from controller.utils import ControllerInfo
from controller.sail_controller import SailController


class ControllerNode(Node):
    """
    ROS2 node that controls the sailboat's rudder and sail.

    Subscribes to:
        /sense/boat_info (BoatInfo): Boat state from simulation (degrees)
        /sense/wind (Wind): Wind conditions from simulation (degrees)
        /planning/desired_heading (Heading): Target heading from planner (degrees)

    Publishes:
        /control/rudder (RudderAngle): Rudder command in degrees [-45, 45]
        /control/sail (SailAngle): Sail angle command in degrees [0, 90]

    Unit conversions:
        - All ROS messages use DEGREES
        - Internal ControllerInfo uses RADIANS
        - PID outputs DEGREES for direct publishing
    """

    def __init__(self):
        super().__init__('controller_node')

        # Subscribe to boat and environment information
        self.create_subscription(BoatInfo, '/sense/boat_info', self._on_boat_info, 1)
        self.create_subscription(Wind, '/sense/wind', self._on_wind_info, 1)

        # Subscribe to desired heading
        self.create_subscription(Heading, '/planning/desired_heading', self._on_desired_heading, 1)

        # Publish control commands
        self.rudder_publisher = self.create_publisher(RudderAngle, '/control/rudder', 1)
        self.sail_publisher = self.create_publisher(SailAngle, '/control/sail', 1)

        # Create Timer to run control loop at 100 Hz
        self.timer = self.create_timer(0.01, self._timer_callback)

        self.rudder_controller = PIDController(kp=0.8, ki=0.0, kd=1.0)
        self.sail_controller = SailController()
        self.info = ControllerInfo(desired_heading=np.deg2rad(300.0))

        self._rudder_actions = []
        self._headings = []
        self._desired_headings = []

    def _on_boat_info(self, msg: BoatInfo):
        # Convert from message (degrees) to internal state (radians for angles)
        self.info.boat_x = msg.x
        self.info.boat_y = msg.y
        self.info.boat_heading = np.deg2rad(msg.heading)
        self.info.boat_vel_x = msg.velocity_x
        self.info.boat_vel_y = msg.velocity_y
        self.info.boat_speed = msg.velocity_magnitude
        self.info.boat_yaw_rate = msg.yaw_rate

    def _on_wind_info(self, msg: Wind):
        # Convert from message (degrees) to internal state (radians)
        self.info.wind_angle = np.deg2rad(msg.angle)
        self.info.wind_speed = msg.speed

    def _on_desired_heading(self, msg: Heading):
        self.info.desired_heading = np.deg2rad(msg.heading)

    def _timer_callback(self):
        if not self.info.is_complete():
            self.get_logger().warning("ControllerInfo incomplete, skipping control step.")
            return
        
        rudder_angle = self.rudder_controller.compute_action(self.info, dt=0.01)
        sail_angle = self.sail_controller.compute_sail_angle(
            self.info.wind_angle, self.info.boat_heading, self.info.boat_speed
        )

        self.rudder_publisher.publish(RudderAngle(angle=rudder_angle))
        self.sail_publisher.publish(SailAngle(sail_angle=sail_angle))
        self._rudder_actions.append(rudder_angle)
        self._headings.append(self.info.boat_heading)
        self._desired_headings.append(self.info.desired_heading)

    def destroy_node(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self._rudder_actions)
        plt.title('Rudder Actions Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Rudder Angle (degrees)')
        plt.subplot(2, 1, 2)
        plt.plot(np.rad2deg(self._headings), label='Actual Heading')
        plt.plot(np.rad2deg(self._desired_headings), label='Desired Heading')
        plt.legend()

        plt.title('Boat Heading Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Heading (degrees)')
        plt.tight_layout()
        plt.savefig('controller_performance.png')
        plt.close()
        super().destroy_node()

def main(args=None):
    """Main function for the simulation node."""
    rclpy.init(args=args)
    node = ControllerNode()
    
    # Keep the node running
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()