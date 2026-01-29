"""This script runs an interactive simulation of a sailboat using MuJoCo.

Users can control the rudder and sail sheet length via UI sliders in the right panel of the viewer.
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

from simulation.sailboat_simulation import SailboatSimulation


def main():
    print(__doc__)

    sim = SailboatSimulation(buoys=[(0, 0), (50, 0), (0, 50), (50, 50)])
    sim.reset()

    last_print_time = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 12
        viewer.cam.lookat[:] = [0, 0, 0]

        print("\n" + "="*60)
        print("SAILBOAT SIMULATION")
        print("="*60)
        print("\nUse the Control sliders in the right panel:")
        print("  - rudder_ctrl: steer left/right (-45 to +45)")
        print("  - sheet_ctrl: sail angle (0=tight, 90=fully out)")
        print("\nWind: 6 m/s from the Northeast")
        print("="*60 + "\n")

        try:
            while viewer.is_running():
                step_start = time.time()

                # Read control values from viewer UI sliders
                # Update both desired and actual angles to bypass rate limiting for interactive control
                rudder_value = np.degrees(sim.data.ctrl[sim.act_rudder])
                sail_value = sim.data.ctrl[sim.act_sheet]

                sim.desired_rudder = rudder_value
                sim.rudder_angle = rudder_value
                sim.desired_sail_angle = sail_value
                sim.sail_angle = sail_value

                # Step physics
                for _ in range(5):
                    sim.step()

                # Update camera to follow boat
                status = sim.get_status()
                viewer.cam.lookat[0] = status['x']
                viewer.cam.lookat[1] = status['y']

                viewer.sync()

                # Print status
                current_time = time.time()
                if current_time - last_print_time > 0.3:
                    print(f"\rSpeed: {status['speed_knots']:5.1f} kts | "
                          f"Heading: {status['heading']:6.1f}° | "
                          f"Rudder: {status['rudder']:+5.1f}° | "
                          f"Sail: {status['sail_angle']:4.1f}° | "
                          f"Boom: {status['boom_angle']:+5.1f}° | "
                          f"Wind: {status['wind_knots']:4.1f} kts  ",
                          f"Wind Dir: {status['wind_dir']:6.1f}°",
                          end="", flush=True)
                    last_print_time = current_time

                # Maintain real-time
                elapsed = time.time() - step_start
                sleep_time = sim.model.opt.timestep * 5 - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user.")

    print("\nSimulation ended.")


if __name__ == "__main__":
    main()
