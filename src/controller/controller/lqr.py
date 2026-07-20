import numpy as np
from scipy.linalg import solve_continuous_are, solve
from typing import TYPE_CHECKING

from controller.utils import ControllerInfo, normalize_angle

if TYPE_CHECKING:
    from simulation.sailboat_simulation import SailboatSimulation


class LQRController:
    """
    Linear Quadratic Regulator (LQR) controller for rudder/course following.

    Controls rudder via state feedback on heading error and yaw rate.

    Unit conventions:
        Input: All angles in RADIANS, yaw_rate in rad/s
        Output: rudder_angle in DEGREES [-45, 45]
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        max_rudder_angle_deg: float = 45.0,
        design_speed: float = 2.0,
        max_gain_boost: float = 4.0,
    ):
        """
        Args:
            A: State matrix.
            B: Action matrix.
            C: Output matrix.
            Q: State cost matrix.
            R: Action cost matrix.
            max_rudder_angle_deg: Rudder travel limit.
            design_speed: The boat speed, in m/s, that A and B were linearised
                at. Rudder authority goes as speed squared, so a gain set that
                is right at 2 m/s is four times too hot at 4 m/s; the command
                is rescaled by (design_speed / boat_speed)^2 to keep the loop
                gain — and so the damping — constant across the speed range.
            max_gain_boost: Cap on that rescaling, for the slow end. Below
                roughly half the design speed the rudder cannot produce the
                moment the gains ask for, and boosting harder only pins it at
                the stop; the planner's stall recovery owns that regime.
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.max_rudder_angle_deg = max_rudder_angle_deg
        self.design_speed = design_speed
        self.max_gain_boost = max_gain_boost

        # Pre-compute LQR gain matrix K
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = solve(self.R, self.B.T @ self.P)
        self.V = np.linalg.inv(-self.C @ np.linalg.inv(self.A - self.B @ self.K) @ self.B)

    def compute_action(self, info: ControllerInfo, dt=0.01) -> float:
        """
        Compute rudder angle command.

        Args:
            info: ControllerInfo with angles in RADIANS (NAUTICAL convention: 0=North).
                  yaw_rate in rad/s (nautical convention: clockwise positive).
            dt: Time step in seconds (not used in LQR but kept for interface consistency).

        Returns:
            rudder_angle_deg: Rudder command in degrees [-45, 45].

        Note:
            This controller expects inputs in NAUTICAL convention (same as PID controller)
            but internally converts to MATHEMATICAL convention for computation, since
            the A, B matrices were derived in math convention.
        """
        # Convert from nautical (0=North) to math (0=East) convention
        # Math convention: 0=East, counterclockwise positive
        boat_heading_math = (np.pi/2 - info.boat_heading) % (2 * np.pi)
        desired_heading_math = (np.pi/2 - info.desired_heading) % (2 * np.pi)
        # Negate yaw_rate for math convention (counterclockwise positive)
        yaw_rate_math = -info.boat_yaw_rate

        heading_error = normalize_angle(desired_heading_math - boat_heading_math)
        desired_heading_wrapped = boat_heading_math + heading_error

        state = np.array([[boat_heading_math], [yaw_rate_math]])
        desired_output = np.array([[desired_heading_wrapped]])

        u = -self.K @ state + self.V @ desired_output

        # Hold the loop gain constant as the boat speeds up: the rudder moment
        # is proportional to speed squared, so without this the same gains that
        # steer calmly at the design speed drive a limit cycle a knot faster.
        speed = info.boat_speed if info.boat_speed is not None else self.design_speed
        u = u * min((self.design_speed / max(speed, 1e-3)) ** 2, self.max_gain_boost)

        rudder_angle_rad = u.item()
        rudder_angle_deg = np.rad2deg(rudder_angle_rad)
        rudder_angle_deg = np.clip(rudder_angle_deg, -self.max_rudder_angle_deg, self.max_rudder_angle_deg)

        return rudder_angle_deg


# Yaw plant constants, all taken from the simulation rather than guessed.
#
#   YAW_INERTIA    mj_fullM() at the boat_yaw dof of sailboat.xml.
#   RUDDER_MOMENT  compute_rudder_forces(): the yaw moment is
#                  0.5 * water_density * v|v| * rudder_area * rudder_cl * rudder_arm
#                  = 0.5 * 1000 * 0.2 * 1.0 * 1.1 = 110 per v^2, per sin(delta).
#   YAW_DAMPING    compute_water_drag() contributes params.yaw_damping = 150,
#                  and the boat_yaw joint in sailboat.xml adds damping="2".
YAW_INERTIA = 110.45      # kg m^2
RUDDER_MOMENT = 110.0     # N m per (m/s)^2 per unit sin(rudder angle)
YAW_DAMPING = 152.0       # N m per rad/s


def get_lqr_controller(
    sim: "SailboatSimulation | None" = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    logger = None,
    design_speed: float = 2.0,
    max_gain_boost: float = 4.0,
) -> LQRController:
    """
    Create and return an LQRController instance.

    The yaw plant is built analytically from the simulation's own physics
    constants rather than from sim.get_heading_dynamics(). That function
    linearises with mujoco.mjd_transitionFD, which perturbs the MuJoCo model
    only — and in this simulation the entire hydrodynamic yaw moment is applied
    from Python each step through data.qfrc_applied, which the finite
    difference holds constant. What it measures for the rudder is therefore not
    steering at all, but the reaction torque of swinging the 3 kg rudder blade:
    a plant roughly 25x too weak and 50x too lightly damped. LQR against that
    model returns gains far too hot for the real boat, and the rudder ends up in
    a full-travel limit cycle. get_heading_dynamics remains correct for what it
    says it does; it is just not the yaw plant the controller flies.

    Args:
        sim: Unused, kept for call compatibility. The plant no longer depends on
            a live simulation instance.
        Q: State cost matrix (2x2) over [heading error, yaw rate]. If None,
            uses diag([1.0, 0.5]).
        R: Control cost matrix (1x1) over rudder angle. If None, uses [[4.0]].
        logger: Optional ROS logger.
        design_speed: Speed in m/s to linearise the rudder authority at.
        max_gain_boost: Cap on the low-speed gain boost, see LQRController.

    Returns:
        Configured LQRController instance.
    """
    # Continuous-time yaw dynamics in MATHEMATICAL convention, state
    # [heading, yaw_rate], input the rudder angle in radians:
    #
    #     heading_dot  = yaw_rate
    #     yaw_rate_dot = -(RUDDER_MOMENT v^2 / I) * delta - (YAW_DAMPING / I) * yaw_rate
    #
    # The input sign is negative because positive (starboard) rudder yaws the
    # bow clockwise, which is negative in the counterclockwise-positive frame.
    rudder_authority = RUDDER_MOMENT * design_speed ** 2 / YAW_INERTIA
    yaw_damping = YAW_DAMPING / YAW_INERTIA

    A = np.array([[0.0, 1.0], [0.0, -yaw_damping]])
    B = np.array([[0.0], [-rudder_authority]])

    # Output matrix: we care about heading (first state)
    C = np.array([[1.0, 0.0]])

    # Default cost matrices if not provided
    if Q is None:
        Q = np.diag([1.0, 0.5])
    if R is None:
        R = np.array([[4.0]])

    controller = LQRController(
        A, B, C, Q, R,
        design_speed=design_speed,
        max_gain_boost=max_gain_boost,
    )

    if logger is not None:
        poles = np.linalg.eigvals(A - B @ controller.K)
        logger.info(
            f"LQR yaw plant at {design_speed} m/s: rudder authority "
            f"{rudder_authority:.3f} rad/s^2 per rad, damping {yaw_damping:.3f} 1/s. "
            f"K={np.round(controller.K.ravel(), 4)}, closed-loop poles={np.round(poles, 3)}"
        )

    return controller
