#!/usr/bin/env python3
"""Test rudder and sail controllers with interactive visualization.

This script tests controller performance by cycling through multiple target headings.
The boat automatically switches to a new heading every SECONDS_PER_HEADING.

Features:
- Choose between LQR or PID rudder controllers (via --controller flag)
- Automatic sail control using SailController
- Cycle through multiple target headings defined in DESIRED_HEADINGS
- Configurable wind conditions, timing, and controller parameters via constants

Configuration is done by editing the constants at the top of this file:
- DESIRED_HEADINGS: List of target headings to test
- SECONDS_PER_HEADING: Duration to maintain each heading
- WIND_DIRECTION_DEG, WIND_SPEED_MS: Wind conditions
- LQR_Q, LQR_R: LQR controller tuning parameters
- PID_KP, PID_KI, PID_KD: PID controller tuning parameters
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse

from simulation.sailboat_simulation import SailboatSimulation
from controller.pid import PIDController
from controller.lqr import get_lqr_controller
from controller.sail_controller import SailController
from controller.utils import ControllerInfo


INITIAL_HEADING_DEG = 90.0  # Initial boat heading in degrees
INITIAL_SAIL_ANGLE_DEG = 25.0  # Initial sail angle in degrees
WIND_DIRECTION_DEG = 45.0  # Wind direction in degrees (from)
WIND_SPEED_MS = 5.0  # Wind speed in m/s
DESIRED_HEADINGS = [180, 120, 270, 100, 200]  # Sequence of desired headings to test
SECONDS_PER_HEADING = 20  # Duration to hold each heading

# LQR rudder controller parameters
LQR_Q = np.diag([0.5, 0.5])  # State cost matrix
LQR_R = np.array([1.0])      # Action cost matrix

# PID rudder controller parameters
PID_KP=0.8
PID_KI=0.0
PID_KD=1.0


def main():
    parser = argparse.ArgumentParser(description="Controller Test for Sailboat Simulation")
    parser.add_argument('--controller', type=str, default="lqr",
                        choices=["lqr", "pid"], help="Type of rudder controller to use")
    args = parser.parse_args()

    print(__doc__)
    print("\n" + "="*60)
    print("CONTROLLER TEST")
    print("="*60)
    print(f"Controller: {args.controller.upper()}")
    print(f"Initial heading: {INITIAL_HEADING_DEG}°")
    print(f"Wind: {WIND_SPEED_MS} m/s from {WIND_DIRECTION_DEG}°")
    print(f"Target headings: {DESIRED_HEADINGS}")
    print(f"Duration per heading: {SECONDS_PER_HEADING}s")
    print("="*60 + "\n")

    # Create simulation with specified wind parameters
    sim = SailboatSimulation(
        wind_direction_deg=WIND_DIRECTION_DEG,
        wind_speed=WIND_SPEED_MS,
        include_course_line=True,
    )
    sim.reset()

    # Set initial heading
    initial_heading_math_deg = 90 - INITIAL_HEADING_DEG
    sim.data.qpos[sim.qadr_yaw] = np.radians(initial_heading_math_deg)

    # Set initial sail angle
    sim.set_sail_angle(INITIAL_SAIL_ANGLE_DEG)

    # Initialize rudder controller
    if args.controller == "lqr":
        controller = get_lqr_controller(sim, Q=LQR_Q, R=LQR_R)
    elif args.controller == "pid":
        controller = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD)
    else:
        raise ValueError(f"Unknown controller type: {args.controller}")

    # Initialize sail controller
    sail_controller = SailController()

    # Track which heading we're currently targeting
    heading_index = 0
    heading_start_time = time.time()
    last_print_time = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Set up camera
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 12
        viewer.cam.lookat[:] = [0, 0, 0]

        print("Simulation running. Press Ctrl+C to stop.\n")

        try:
            while viewer.is_running():
                step_start = time.time()

                # Check if it's time to switch to the next heading
                elapsed_time = time.time() - heading_start_time
                if elapsed_time >= SECONDS_PER_HEADING:
                    heading_index = (heading_index + 1) % len(DESIRED_HEADINGS)
                    heading_start_time = time.time()
                    print(f"\n\nSwitching to heading {heading_index + 1}/{len(DESIRED_HEADINGS)}: {DESIRED_HEADINGS[heading_index]}°\n")

                # Get current desired heading
                current_desired_heading_deg = DESIRED_HEADINGS[heading_index]

                # Convert desired heading from nautical (0=North) to mathematical (0=East) convention
                # and then to radians, normalized to [0, 2π)
                desired_heading_math_deg = 90 - current_desired_heading_deg
                desired_heading_rad = np.radians(desired_heading_math_deg) % (2 * np.pi)

                # Get current boat state
                status = sim.get_status()

                # Convert heading from nautical to mathematical convention and to radians
                boat_heading_math_deg = 90 - status['heading']
                boat_heading_rad = np.radians(boat_heading_math_deg) % (2 * np.pi)

                # Convert wind direction to math convention and normalize to [0, 2π)
                wind_angle_rad = np.radians(90 - status['wind_dir']) % (2 * np.pi)

                # Create ControllerInfo for controller
                info = ControllerInfo(
                    boat_x=status['x'],
                    boat_y=status['y'],
                    boat_heading=boat_heading_rad,
                    boat_vel_x=status['vel_x'],
                    boat_vel_y=status['vel_y'],
                    boat_speed=status['speed'],
                    boat_yaw_rate=-status['yaw_rate'],  # Negate to convert from nautical to math convention
                    wind_angle=wind_angle_rad,
                    wind_speed=status['wind_speed'],
                    desired_heading=desired_heading_rad,
                )

                # Compute rudder command from controller
                rudder_deg = controller.compute_action(info)

                # Compute sail angle from sail controller
                sail_deg = sail_controller.compute_sail_angle(
                    info.wind_angle, info.boat_heading, info.boat_speed
                )

                # Apply controller commands
                sim.set_rudder(rudder_deg)
                sim.set_sail_angle(sail_deg)

                # Update course line to show desired heading
                sim.update_course_line(desired_heading_rad)

                # Step physics multiple times for smoother visualization
                for _ in range(5):
                    sim.step()

                # Update camera to follow boat
                viewer.cam.lookat[0] = status['x']
                viewer.cam.lookat[1] = status['y']

                viewer.sync()

                # Print status periodically
                current_time = time.time()
                if current_time - last_print_time > 0.3:
                    # Calculate heading error
                    heading_error = (current_desired_heading_deg - status['heading'] + 180) % 360 - 180
                    time_remaining = SECONDS_PER_HEADING - elapsed_time

                    print(f"\r[{heading_index + 1}/{len(DESIRED_HEADINGS)}] "
                          f"T-{time_remaining:4.1f}s | "
                          f"Speed: {status['speed_knots']:5.1f} kts | "
                          f"Hdg: {status['heading']:6.1f}° | "
                          f"Tgt: {current_desired_heading_deg:6.1f}° | "
                          f"Err: {heading_error:+5.1f}° | "
                          f"Rudder: {status['rudder']:+5.1f}° | "
                          f"Sail: {status['sail_angle']:4.1f}°",
                          end="", flush=True)
                    last_print_time = current_time

                # Maintain real-time simulation
                elapsed = time.time() - step_start
                sleep_time = sim.model.opt.timestep * 5 - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user.")

    print("\nSimulation ended.")


if __name__ == "__main__":
    main()
