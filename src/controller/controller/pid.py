import numpy as np
from controller.utils import ControllerInfo, normalize_angle


class PIDController:
    """
    PID controller for rudder/course following.

    Controls rudder via PID on heading error. Includes stall recovery
    (hard rudder turn when stuck in irons).

    Unit conventions:
        Input (ControllerInfo): All angles in RADIANS, yaw_rate in rad/s
        Output: rudder_angle in DEGREES [-45, 45]
    """

    def __init__(self, kp=1.0, ki=0.0, kd=5.0):
        """
        Args:
            kp: Proportional gain for heading error.
            ki: Integral gain (often 0 works best for boats).
            kd: Derivative gain (damping to prevent overshoot).
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0

    def reset(self):
        """Reset integrator state."""
        self.integral = 0.0

    def compute_action(self, info: ControllerInfo, dt=0.01) -> float:
        """
        Compute rudder angle command.

        Args:
            info: ControllerInfo with angles in RADIANS, yaw_rate in rad/s.
            dt: Time step in seconds for integral/derivative calculation.

        Returns:
            rudder_angle_deg: Rudder command in degrees [-45, 45].
        """
        # All internal calculations in radians
        course_error = normalize_angle(info.boat_heading - info.desired_heading)

        # heading_error = target - heading = -course_error
        heading_error = -course_error

        # Wind angle relative to boat heading (for sail trim)
        apparent_wind_angle = normalize_angle(info.wind_angle - info.boat_heading)
        apparent_wind_deg = abs(np.degrees(apparent_wind_angle))

        # --- Stall recovery ---
        # If stuck in irons (slow + pointing into wind), override to recovery mode
        if info.boat_speed < 0.5 and apparent_wind_deg < 50:
            # Turn toward the desired course (shorter path)
            # In nautical convention (clockwise positive):
            # heading_error > 0 means target is to starboard (right), need positive rudder
            # heading_error < 0 means target is to port (left), need negative rudder
            turn_sign = np.sign(heading_error) if heading_error != 0 else 1.0
            rudder_deg = turn_sign * 45.0  # Hard over toward target
            return rudder_deg

        # --- Normal PID control ---
        self.integral += heading_error * dt
        self.integral = np.clip(self.integral, -0.5, 0.5)  # anti-windup

        # Dynamic D: low damping when far (turn fast), high damping when close (stable)
        error_magnitude = min(abs(heading_error), 1.0)  # cap at 1 radian (~57°)
        d_scale = 1.0 - 0.7 * error_magnitude  # 1.0 when on target, 0.3 when far off

        # PID output is desired rudder angle (radians, then converted to degrees)
        # No negation: nautical convention (clockwise positive) matches rudder convention
        rudder_desired = (
            self.kp * heading_error +
            self.ki * self.integral -
            self.kd * info.boat_yaw_rate * d_scale
        )
        rudder_desired_deg = np.degrees(rudder_desired)
        rudder_desired_deg = np.clip(rudder_desired_deg, -45.0, 45.0)

        return rudder_desired_deg
