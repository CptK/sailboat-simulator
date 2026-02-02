#!/usr/bin/env python3
"""
Sailboat Simulation using MuJoCo

A physics-based sailboat simulation with wind propulsion and water dynamics.

Controls:
  Use the sliders in the right panel of the viewer:
    - rudder_ctrl: Steer left/right (-45 to +45 degrees)
    - sheet_ctrl: Sail angle (0=tight, 90=fully out)
      The wind automatically pushes the sail to the correct side.

Camera:
  - Click and drag to rotate
  - Scroll to zoom

The simulation applies:
  - Wind force on the sail based on apparent wind
  - Sail automatically swings to leeward based on wind direction
  - Rudder turning force based on boat speed
  - Water drag proportional to velocity
"""

import mujoco
import numpy as np
from pathlib import Path

from simulation.sailboat_physics import (
    PhysicsParams,
    BoatState,
    WindState,
    SailControl,
    compute_total_forces,
)
from simulation.build_env import build_env, MAX_ROUTE_SEGMENTS


def get_resource_path(resource_name: str) -> str:
    """Get path to a resource file, works both in development and installed package."""
    # Try ROS 2 package share directory first (installed package)
    try:
        from ament_index_python.packages import get_package_share_directory
        package_share = get_package_share_directory('simulation')
        resource_path = Path(package_share) / 'resource' / resource_name
        if resource_path.exists():
            return str(resource_path)
    except (ImportError, Exception):
        pass

    # Fallback: relative to this file (development mode)
    package_dir = Path(__file__).parent.parent
    resource_path = package_dir / 'resource' / resource_name
    if resource_path.exists():
        return str(resource_path)

    raise FileNotFoundError(f"Could not find resource: {resource_name}")


class SailboatSimulation:
    def __init__(
        self,
        model_path="sailboat.xml",
        buoys: list[tuple[float, float] | tuple[float, float, float, float, float]] = [],
        wind_direction_deg=45.0,
        wind_speed=6.0,
        include_course_line=False,
    ):
        """Initialize the sailboat simulation.

        All angles use DEGREES for public API consistency.

        Args:
            model_path: Path to the MuJoCo XML model file (filename only, or absolute path).
            buoys: List of buoy definitions. Each can be:
                - (x, y) tuple for default orange color
                - (x, y, r, g, b) tuple for custom RGB color [0.0-1.0]
            wind_direction_deg: Initial wind direction in degrees [0, 360).
                                Direction wind comes FROM. 0 = North, 90 = East.
            wind_speed: Initial wind speed in m/s [0, 25].
            include_course_line: If True, adds a course line mocap body.
        """
        # Resolve path - if just a filename, look in resource folder
        if not Path(model_path).is_absolute():
            model_path = get_resource_path(model_path)
        modified_xml = build_env(model_path, buoys, include_course_line)
        self.model = mujoco.MjModel.from_xml_string(modified_xml)
        self.data = mujoco.MjData(self.model)

        # Physics parameters
        self.physics = PhysicsParams()

        # Wind parameters (internal state uses radians in mathematical convention)
        # Convert from nautical (0=North) to mathematical (0=East)
        math_wind_direction_deg = 90 - wind_direction_deg
        self.wind = WindState(
            direction=np.radians(math_wind_direction_deg),
            speed=wind_speed
        )

        # Desired (target) angles - what controller requests
        self.desired_rudder = 0.0  # degrees
        self.desired_sail_angle = 45.0  # degrees

        # Actual (current) angles - physical actuator position
        self.rudder_angle = 0.0  # degrees
        self.sail_angle = 45.0  # 0-90 degrees, max angle boom can swing out

        # Actuator rate limits (degrees per second)
        # Note: High rudder rate needed for stable async ROS control
        # (rate limiting + async control causes instability with realistic rates)
        self.MAX_RUDDER_RATE = 30.0
        self.MAX_SAIL_RATE = 9.0

        # Get joint indices
        self.jnt_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "boat_x")
        self.jnt_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "boat_y")
        self.jnt_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "boat_yaw")
        self.jnt_boom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "boom_joint")
        self.jnt_rudder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rudder_joint")
        self.jnt_roll = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "boat_roll")

        # Get qpos/qvel addresses for joints
        self.qadr_x = self.model.jnt_qposadr[self.jnt_x]
        self.qadr_y = self.model.jnt_qposadr[self.jnt_y]
        self.qadr_yaw = self.model.jnt_qposadr[self.jnt_yaw]
        self.qadr_boom = self.model.jnt_qposadr[self.jnt_boom]

        self.vadr_x = self.model.jnt_dofadr[self.jnt_x]
        self.vadr_y = self.model.jnt_dofadr[self.jnt_y]
        self.vadr_yaw = self.model.jnt_dofadr[self.jnt_yaw]
        self.vadr_roll = self.model.jnt_dofadr[self.jnt_roll]
        self.vadr_boom = self.model.jnt_dofadr[self.jnt_boom]

        # Get actuator indices
        self.act_rudder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rudder_ctrl")
        self.act_sheet = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "sheet_ctrl")

    def get_boat_state(self) -> BoatState:
        """Get current boat state for physics calculations."""
        return BoatState(
            heading=self.data.qpos[self.qadr_yaw],
            velocity=np.array([
                self.data.qvel[self.vadr_x],
                self.data.qvel[self.vadr_y],
                0.0
            ]),
            boom_angle=self.data.qpos[self.qadr_boom],
            rudder_angle=self.rudder_angle,
            yaw_rate=self.data.qvel[self.vadr_yaw],
        )

    def get_sail_control(self) -> SailControl:
        """Get current sail control settings."""
        return SailControl(sail_angle=self.sail_angle)

    def apply_forces(self):
        """Compute and apply all physics forces."""
        boat = self.get_boat_state()
        sail = self.get_sail_control()

        forces = compute_total_forces(boat, self.wind, sail, self.physics)

        self.data.qfrc_applied[self.vadr_x] = forces.fx
        self.data.qfrc_applied[self.vadr_y] = forces.fy
        self.data.qfrc_applied[self.vadr_yaw] = forces.yaw
        self.data.qfrc_applied[self.vadr_roll] = forces.roll
        self.data.qfrc_applied[self.vadr_boom] = forces.boom

    def update_wind_indicator(self):
        """Update wind arrow and masthead vane to show wind direction."""
        boat_x = self.data.qpos[self.qadr_x]
        boat_y = self.data.qpos[self.qadr_y]
        boat_heading = self.data.qpos[self.qadr_yaw]

        # Large floating arrow - shows TRUE wind direction
        # Position it offset from the boat so it's always visible
        arrow_offset = 10
        arrow_pos = np.array([
            boat_x + arrow_offset * np.cos(boat_heading + np.pi/4),
            boat_y + arrow_offset * np.sin(boat_heading + np.pi/4),
            5
        ])
        # Arrow points where wind blows TO (opposite of where it comes FROM)
        wind_to_angle = self.wind.direction + np.pi

        self.data.mocap_pos[0] = arrow_pos
        self.data.mocap_quat[0] = [np.cos(wind_to_angle/2), 0, 0, np.sin(wind_to_angle/2)]

        # Masthead wind vane - shows APPARENT wind relative to boat
        # Position at top of mast (follows boat position)
        mast_top = np.array([
            boat_x - 0.1 * np.cos(boat_heading),  # Mast is slightly aft
            boat_y - 0.1 * np.sin(boat_heading),
            4.5  # Top of mast
        ])

        # Calculate apparent wind
        boat_vel = np.array([self.data.qvel[self.vadr_x], self.data.qvel[self.vadr_y]])
        true_wind = np.array([
            -self.wind.speed * np.cos(self.wind.direction),
            -self.wind.speed * np.sin(self.wind.direction)
        ])
        apparent_wind = true_wind - boat_vel
        apparent_angle = np.arctan2(apparent_wind[1], apparent_wind[0])

        self.data.mocap_pos[1] = mast_top
        self.data.mocap_quat[1] = [np.cos(apparent_angle/2), 0, 0, np.sin(apparent_angle/2)]

    def set_rudder(self, angle_deg):
        """Set desired rudder angle in degrees (-45 to +45).

        The actual rudder position moves toward this target at a limited rate
        (MAX_RUDDER_RATE degrees/second) during each simulation step.
        """
        self.desired_rudder = np.clip(angle_deg, -45, 45)

    def set_sail_angle(self, angle):
        """Set desired sail angle (0-90 degrees, max angle boom can swing out).

        The actual sail position moves toward this target at a limited rate
        (MAX_SAIL_RATE degrees/second) during each simulation step.
        """
        self.desired_sail_angle = np.clip(angle, 0, 90)

    def _update_actuators(self):
        """Move actuators toward desired positions at limited rates."""
        dt = self.model.opt.timestep

        # Rudder: move toward desired at limited rate
        rudder_error = self.desired_rudder - self.rudder_angle
        max_rudder_delta = self.MAX_RUDDER_RATE * dt
        self.rudder_angle += np.clip(rudder_error, -max_rudder_delta, max_rudder_delta)

        # Sail: move toward desired at limited rate
        sail_error = self.desired_sail_angle - self.sail_angle
        max_sail_delta = self.MAX_SAIL_RATE * dt
        self.sail_angle += np.clip(sail_error, -max_sail_delta, max_sail_delta)

        # Update MuJoCo actuators with actual positions
        self.data.ctrl[self.act_rudder] = np.radians(self.rudder_angle)
        self.data.ctrl[self.act_sheet] = self.sail_angle

    def set_wind(self, direction_deg=None, speed=None):
        """
        Set wind parameters.

        Args:
            direction_deg: Wind direction in degrees [0, 360), direction wind comes FROM.
                           0 = from North, 90 = from East (nautical convention).
            speed: Wind speed in m/s [0, 25].
        """
        if direction_deg is not None:
            # Convert from nautical (0=North) to mathematical (0=East) convention
            math_direction_deg = 90 - direction_deg
            self.wind.direction = np.radians(math_direction_deg)
        if speed is not None:
            self.wind.speed = np.clip(speed, 0, 25)

    def step(self):
        """Advance simulation by one timestep."""
        self.data.qfrc_applied[:] = 0
        self._update_actuators()
        self.apply_forces()
        self.update_wind_indicator()
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)

        # Reset both desired and actual to same initial values
        self.desired_rudder = 0.0
        self.desired_sail_angle = 25.0  # Moderate - works for most points of sail
        self.rudder_angle = 0.0
        self.sail_angle = 25.0

        # Initialize MuJoCo controls to match Python state
        self.data.ctrl[self.act_rudder] = 0
        self.data.ctrl[self.act_sheet] = self.sail_angle

        # Initialize boom near expected position based on wind
        # This prevents delay while boom swings at startup
        max_boom = np.radians(self.sail_angle)
        wind_to_boat = self.wind.direction - self.data.qpos[self.qadr_yaw]
        # Wind from port -> boom to starboard, wind from starboard -> boom to port
        if np.sin(wind_to_boat) > 0:  # Wind has component from port
            self.data.qpos[self.qadr_boom] = max_boom
        else:
            self.data.qpos[self.qadr_boom] = -max_boom

    def update_course_line(self, course_rad: float):
        """
        Update the course line to show the target/desired course direction.
        The line is anchored at the boat's current position.

        Args:
            course_rad: Target course in radians (0 = East, pi/2 = North).
        """
        # Course line is mocap body index 2 (after wind arrow and masthead vane)
        if self.data.mocap_pos.shape[0] < 3:
            return  # No course line mocap body

        # Get boat position
        boat_x = self.data.qpos[self.qadr_x]
        boat_y = self.data.qpos[self.qadr_y]

        # Position line center ahead of boat along the course
        line_length = 20.0
        cx = boat_x + (line_length / 2) * np.cos(course_rad)
        cy = boat_y + (line_length / 2) * np.sin(course_rad)

        self.data.mocap_pos[2] = [cx, cy, 0.01]
        # Quaternion for rotation around Z axis
        self.data.mocap_quat[2] = [np.cos(course_rad / 2), 0, 0, np.sin(course_rad / 2)]

    def update_route(self, waypoints: list[tuple[float, float]]):
        """
        Update the route visualization showing the planned path.

        Args:
            waypoints: List of (east, north) tuples defining the route.
        """
        # Route segments start at mocap index 3 (after wind arrow, masthead vane, course line)
        route_start_idx = 3
        if self.data.mocap_pos.shape[0] < route_start_idx + MAX_ROUTE_SEGMENTS:
            return  # No route segment mocap bodies

        # Position segments between consecutive waypoints
        num_segments = min(len(waypoints) - 1, MAX_ROUTE_SEGMENTS)

        for i in range(MAX_ROUTE_SEGMENTS):
            mocap_idx = route_start_idx + i

            if i < num_segments:
                # Get start and end points of this segment
                x1, y1 = waypoints[i]
                x2, y2 = waypoints[i + 1]

                # Calculate segment center, length, and angle
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.arctan2(y2 - y1, x2 - x1)

                # Position at segment center
                self.data.mocap_pos[mocap_idx] = [cx, cy, 0.005]

                # Rotate to align with segment direction
                self.data.mocap_quat[mocap_idx] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]

                # Scale the segment length by adjusting geom size
                # The geom half-size in x is 1.0 by default, so we scale to half the segment length
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"route_segment_geom_{i}")
                if geom_id >= 0:
                    self.model.geom_size[geom_id, 0] = length / 2
            else:
                # Hide unused segments underground
                self.data.mocap_pos[mocap_idx] = [0, 0, -100]

    def get_boat_speed(self):
        """Get boat speed (magnitude of horizontal velocity)."""
        vx = self.data.qvel[self.vadr_x]
        vy = self.data.qvel[self.vadr_y]
        return np.linalg.norm([vx, vy])

    def get_status(self):
        """Get current simulation status.

        Returns angles in nautical convention (0=North, 90=East).
        """
        # Boat heading in mathematical convention (0=East)
        heading_math_deg = np.degrees(self.data.qpos[self.qadr_yaw]) % 360
        if heading_math_deg < 0:
            heading_math_deg += 360
        # Convert to nautical convention (0=North)
        heading_deg = (90 - heading_math_deg) % 360

        # Wind direction in mathematical convention (internal storage)
        wind_math_deg = np.degrees(self.wind.direction) % 360
        if wind_math_deg < 0:
            wind_math_deg += 360
        # Convert to nautical convention (0=North)
        wind_dir_deg = (90 - wind_math_deg) % 360

        speed = self.get_boat_speed()

        return {
            'x': self.data.qpos[self.qadr_x],
            'y': self.data.qpos[self.qadr_y],
            'heading': heading_deg,
            'vel_x': self.data.qvel[self.vadr_x],
            'vel_y': self.data.qvel[self.vadr_y],
            'speed': speed,
            'speed_knots': speed * 1.944,
            'yaw_rate': -self.data.qvel[self.vadr_yaw],  # Negated to match nautical convention
            'rudder': self.rudder_angle,
            'sail_angle': self.sail_angle,
            'boom_angle': np.degrees(self.data.qpos[self.qadr_boom]),
            'wind_dir': wind_dir_deg,
            'wind_speed': self.wind.speed,
            'wind_knots': self.wind.speed * 1.944,
        }
    
    def set_boat_position(self, east: float, north: float):
        """Set the boat's position in the simulation.

        Args:
            east: East coordinate in meters.
            north: North coordinate in meters.
        """
        self.data.qpos[self.qadr_x] = east
        self.data.qpos[self.qadr_y] = north
        self.data.qvel[self.vadr_x] = 0.0
        self.data.qvel[self.vadr_y] = 0.0

    def get_heading_dynamics(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute linearized heading dynamics matrices around current state.

        Returns a tuple of (A_discrete, B_discrete, dt) where:
        - A_discrete: 2x2 discrete-time state matrix for [yaw, yaw_rate]
        - B_discrete: 2x1 discrete-time control matrix for rudder input
        - dt: Estimated timestep from the dynamics

        These matrices describe the discrete-time system:
            [yaw, yaw_rate]_{k+1} = A @ [yaw, yaw_rate]_k + B @ rudder_k
        """
        nv = self.model.nv  # number of velocity DOFs
        nu = self.model.nu  # number of controls

        # Initialize full system matrices
        A_full = np.zeros((2*nv, 2*nv))
        B_full = np.zeros((2*nv, nu))

        # Compute linearized dynamics around current state
        mujoco.mjd_transitionFD(
            self.model,
            self.data,
            eps=1e-6,
            flg_centered=1,
            A=A_full,
            B=B_full,
            C=None,
            D=None
        )

        # Extract 2x2 heading dynamics
        # qpos[2] = boat_yaw, qvel[2] = boat_yaw_rate (at index nv + 2 in full state)
        yaw_idx = self.qadr_yaw            # Position index for yaw
        yaw_rate_idx = nv + self.vadr_yaw  # Velocity index in full state vector

        indices = [yaw_idx, yaw_rate_idx]

        A_heading = A_full[np.ix_(indices, indices)]
        B_heading = B_full[indices, self.act_rudder:self.act_rudder+1]  # Rudder control only

        # Estimate timestep from the discrete dynamics
        # A[0,1] ≈ dt because: yaw_{k+1} ≈ yaw_k + dt * yaw_rate_k
        dt_estimated = A_heading[0, 1]

        return A_heading, B_heading, dt_estimated
