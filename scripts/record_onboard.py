"""Record a short video of the onboard camera next to the chase camera.

The boat sails past a few coloured buoys with a gentle constant turn. Left half of
the video: what the boat sees (bow_cam, no debug overlays). Right half: the usual
third-person chase view.

    python scripts/record_onboard.py [output.mp4]

Headless machines need MUJOCO_GL=osmesa (or egl).
"""
import sys

import imageio
import mujoco
import numpy as np

from simulation.sailboat_simulation import SailboatSimulation

DURATION_S = 30.0
FPS = 15
WIDTH, HEIGHT = 480, 360
RUDDER_DEG = 4.0  # gentle turn so the buoys sweep through the view


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "onboard.mp4"

    sim = SailboatSimulation(
        buoys=[
            (25, 5, 0.0, 0.8, 0.0),    # green
            (45, -10, 1.0, 0.0, 0.0),  # red
            (60, 20, 1.0, 1.0, 0.0),   # yellow
            (70, 26, 1.0, 1.0, 0.0),   # yellow
            (35, 35, 1.0, 0.5, 0.0),   # orange
        ],
        wind_direction_deg=0.0,  # from North: boat starts on a beam reach heading East
        wind_speed=5.0,
    )
    sim.reset()
    sim.set_rudder(RUDDER_DEG)

    chase = mujoco.Renderer(sim.model, height=HEIGHT, width=WIDTH)
    steps_per_frame = int(round(1.0 / (FPS * sim.model.opt.timestep)))

    with imageio.get_writer(output, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=None) as writer:
        for _ in range(int(DURATION_S * FPS)):
            for _ in range(steps_per_frame):
                sim.step()
            onboard = sim.render_onboard(WIDTH, HEIGHT)
            chase.update_scene(sim.data, camera="chase_cam")
            writer.append_data(np.concatenate([onboard, chase.render()], axis=1))

    chase.close()
    sim.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
