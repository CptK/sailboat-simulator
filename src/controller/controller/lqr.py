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
    ):
        """
        Args:
            A: State matrix.
            B: Action matrix.
            C: Output matrix.
            Q: State cost matrix.
            R: Action cost matrix.
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.max_rudder_angle_deg = max_rudder_angle_deg

        # Pre-compute LQR gain matrix K
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = solve(self.R, self.B.T @ self.P)
        self.V = np.linalg.inv(-self.C @ np.linalg.inv(self.A - self.B @ self.K) @ self.B)

    def compute_action(self, info: ControllerInfo, dt=0.01) -> float:
        """
        Compute rudder angle command.

        Args:
            info: ControllerInfo with angles in RADIANS, yaw_rate in rad/s.
            dt: Time step in seconds (not used in LQR but kept for interface consistency).

        Returns:
            rudder_angle_deg: Rudder command in degrees [-45, 45].
        """
        heading_error = normalize_angle(info.desired_heading - info.boat_heading)
        desired_heading_wrapped = info.boat_heading + heading_error

        state = np.array([[info.boat_heading], [info.boat_yaw_rate]])
        desired_output = np.array([[desired_heading_wrapped]])

        u = -self.K @ state + self.V @ desired_output

        rudder_angle_rad = u.item()
        rudder_angle_deg = np.rad2deg(rudder_angle_rad)
        rudder_angle_deg = np.clip(rudder_angle_deg, -self.max_rudder_angle_deg, self.max_rudder_angle_deg)

        return rudder_angle_deg


def get_lqr_controller(
    sim: "SailboatSimulation | None" = None,
    Q: np.ndarray | None = None, 
    R: np.ndarray | None = None
) -> LQRController:
    """
    Create and return an LQRController instance.

    Args:
        sim: SailboatSimulation instance to extract dynamics from. If None, uses hardcoded matrices.
        Q: State cost matrix (2x2). If None, uses default [0.5, 0.5] diagonal.
        R: Control cost matrix (1x1). If None, uses default [[1.0]].

    Returns:
        Configured LQRController instance.
    """
    if sim is not None:
        # Get dynamics from simulation
        A_discrete, B_discrete, dt_estimated = sim.get_heading_dynamics()
    else:
        # Fallback to hardcoded values (for backward compatibility)
        A_discrete = np.array([[1.00000000e+00, 4.99957821e-03],
                               [-3.99031927e-15, 9.99915642e-01]])
        B_discrete = np.array([[-5.62762090e-06],
                               [-1.12552418e-03]])
        dt_estimated = 0.005

    # Convert discrete-time to continuous-time dynamics
    A = (A_discrete - np.eye(2)) / dt_estimated
    B = B_discrete / dt_estimated

    # Output matrix: we care about heading (first state)
    C = np.array([[1.0, 0.0]])

    # Default cost matrices if not provided
    if Q is None:
        Q = np.diag([0.5, 0.5])
    if R is None:
        R = np.array([[1.0]])

    return LQRController(A, B, C, Q, R)
