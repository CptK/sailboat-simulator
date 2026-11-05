"""Render the video of a recorded simulation run, in parallel.

The simulation node, with render_mode "record", saves what every frame would
show (see simulation.recording). This renders those frames to an mp4 that is
identical to what live rendering draws — chase camera, onboard inset, minimap,
collision label — but with the work split across processes.

Why processes: MuJoCo renders through llvmpipe (software OpenGL) here, which
spreads a frame's pixels over threads but processes its geometry on one. A map
with many land meshes (each drawn again for shadows and reflections) is
geometry-heavy, so one renderer with 12 threads still takes ~110 ms per frame
with the onboard inset. Several renderers with a few threads each get about
1.5x as many frames through. Each worker renders a
contiguous stretch of the video to its own file; the files are then joined
without re-encoding.

Runs: run_simulation_graph.py keeps each run in its own directory,

    runs/<scenario>_<date>_<time>/
        scenario.yaml       the scenario as it was run
        recording.npz       what every frame shows
        video_4x.mp4        the video, named after its speedup

and render_video takes such a directory. Its defaults come from the video
section of the run's scenario.yaml (speedup 4 if it has none), and the video is
written next to the recording, so another speedup adds a file rather than
replacing one.

Usage:
    render_video                              # the latest run in runs/, as configured
    render_video --speedup 1                  # ... at real speed, as video_1x.mp4
    render_video runs/my_course_2026-11-05_14-07-12
    render_video some/recording.npz -o out.mp4 --speedup 3 --workers 6
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from tqdm import tqdm

from simulation.recording import Recording, route_at

#: File names inside a run directory (see the module docstring).
RECORDING_NAME = 'recording.npz'
SCENARIO_NAME = 'scenario.yaml'
#: Where run_simulation_graph.py puts run directories, relative to the repo.
RUNS_DIR = 'runs'
#: Speedup when the run's scenario does not set one.
DEFAULT_SPEEDUP = 4.0


def video_name(speedup: float) -> str:
    """The video's file name in a run directory, e.g. video_4x.mp4."""
    return f'video_{speedup:g}x.mp4'


def latest_run(runs_dir: str = RUNS_DIR) -> str | None:
    """The most recently started run directory that holds a recording."""
    try:
        runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)]
    except FileNotFoundError:
        return None
    runs = [d for d in runs if os.path.isfile(os.path.join(d, RECORDING_NAME))]
    return max(runs, key=os.path.getmtime) if runs else None


def run_video_settings(run_dir: str) -> dict:
    """The video section of a run's scenario.yaml: speedup, fps, enabled.

    Missing settings get their defaults (speedup 4, fps None = the run's,
    enabled). scenarios are validated when a run starts, so this only reads.
    """
    import yaml

    settings = {'speedup': DEFAULT_SPEEDUP, 'fps': None, 'enabled': True}
    path = os.path.join(run_dir, SCENARIO_NAME)
    if os.path.isfile(path):
        with open(path) as fh:
            video = (yaml.safe_load(fh) or {}).get('video') or {}
        settings.update({k: video[k] for k in settings if k in video})
    return settings


#: Encoder settings, as live video recording uses.
CODEC = dict(codec='libx264', quality=8, pixelformat='yuv420p', macro_block_size=None)


def frame_schedule(recording: Recording, fps: float, speedup: float) -> list[tuple[int, int]]:
    """Which recorded frame each video frame shows.

    Video frame k shows simulated time t0 + k * speedup / fps, where t0 is the
    first recorded frame; it uses the nearest recorded frame. Segments follow
    one another, as they did in the run.

    Returns:
        (segment index, frame index within the segment) per video frame.
    """
    times = []
    index = []
    for s, seg in enumerate(recording.segments):
        times.extend(seg.time)
        index.extend((s, f) for f in range(len(seg.time)))
    if not times:
        return []
    times = np.asarray(times)
    step = speedup / fps
    # Up to the last recorded frame, never past it (it would repeat at the end)
    targets = np.arange(times[0], times[-1] + 1e-9, step)
    # Nearest recorded frame to each target time
    pos = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    nearer_left = (targets - times[pos - 1]) <= (times[pos] - targets)
    picks = np.where(nearer_left, pos - 1, pos)
    return [index[i] for i in picks]


class SegmentPlayer:
    """Rebuilds a recorded segment's model and puts it in any recorded frame's state."""

    def __init__(self, segment, obstacles=(), world_half_size=None):
        from simulation.sailboat_simulation import SailboatSimulation

        scene = segment.scene
        self.segment = segment
        self.sim = SailboatSimulation(
            buoys=scene.buoys,
            include_course_line=True,
            wind_direction_deg=scene.wind_direction,
            wind_speed=scene.wind_speed,
            obstacles=list(obstacles),
            world_half_size=world_half_size,
        )
        self.route = None

    def show(self, frame: int) -> bool:
        """Put the model in the state of ``frame``.

        Returns:
            True if the route drawn changed (the minimap needs it too).
        """
        import mujoco

        seg, sim = self.segment, self.sim
        route = route_at(seg, frame)
        changed = route is not None and route != self.route
        if changed:
            sim.update_route(route)  # the route's segment lengths live in the model
            self.route = route
        sim.data.qpos[:] = seg.qpos[frame]
        sim.data.mocap_pos[:] = seg.mocap_pos[frame]
        sim.data.mocap_quat[:] = seg.mocap_quat[frame]
        mujoco.mj_forward(sim.model, sim.data)
        return changed

    def boat_pose(self, frame: int) -> tuple[float, float, float]:
        """(east, north, yaw) of the boat at ``frame``."""
        q = self.segment.qpos[frame]
        return q[self.sim.qadr_x], q[self.sim.qadr_y], q[self.sim.qadr_yaw]

    def close(self):
        self.sim.close()


def _render_part(job: dict) -> str:
    """Worker: render a stretch of the schedule to its own mp4. Returns its path."""
    # Must be set before the first GL context is created in this process
    os.environ['LP_NUM_THREADS'] = str(job['threads'])
    os.environ.setdefault('MUJOCO_GL', 'osmesa')

    import imageio
    import mujoco

    from simulation.build_env import load_map_world
    from simulation.minimap import Minimap
    from simulation.video_overlay import (COLLISION_TEXT, draw_label, inset_size,
                                          picture_in_picture)

    recording = Recording.load(job['recording'])
    opts = job['render']

    obstacles, water, half_size = [], None, None
    if recording.map_path:
        _, water, half_size, obstacles = load_map_world(recording.map_path)

    writer = imageio.get_writer(job['output'], fps=opts['fps'], **CODEC)
    segment_index, player, renderer, camera, minimap = None, None, None, None, None
    try:
        for s, f in job['schedule']:
            if s != segment_index:
                if player is not None:
                    renderer.close()
                    player.close()
                segment_index = s
                seg = recording.segments[s]
                player = SegmentPlayer(seg, obstacles, half_size)
                renderer = mujoco.Renderer(player.sim.model, width=opts['width'],
                                           height=opts['height'])
                camera = mujoco.mj_name2id(player.sim.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                           'chase_cam')
                minimap = None
                if opts['minimap_inset_scale'] > 0:
                    w, h = inset_size(opts['width'], opts['minimap_inset_scale'])
                    minimap = Minimap(w, h, water=water,
                                      boat=(seg.scene.boat_east, seg.scene.boat_north),
                                      buoys=seg.scene.buoys,
                                      view_size=opts['minimap_view_size'])
                    # The track so far: this worker may start mid-run
                    for earlier in range(f):
                        east, north, _ = player.boat_pose(earlier)
                        minimap.update_boat(east, north)

            if player.show(f) and minimap is not None:
                minimap.set_route(player.route)

            renderer.update_scene(player.sim.data, camera)
            frame = renderer.render()
            if opts['onboard_inset_scale'] > 0:
                w, h = inset_size(opts['width'], opts['onboard_inset_scale'])
                frame = picture_in_picture(frame, player.sim.render_onboard(w, h))
            if minimap is not None:
                frame = picture_in_picture(frame, minimap.render(*player.boat_pose(f)),
                                           corner='bottom-left')
            if player.segment.collision[f]:
                frame = draw_label(frame, COLLISION_TEXT)
            writer.append_data(frame)

            if job['progress'] is not None:
                with job['progress'].get_lock():
                    job['progress'].value += 1
    finally:
        writer.close()
        if player is not None:
            renderer.close()
            player.close()
    return job['output']


def _concat(parts: list[str], output: str):
    """Join mp4 parts that share codec settings, without re-encoding."""
    import imageio_ffmpeg

    if len(parts) == 1:
        shutil.move(parts[0], output)
        return
    listing = os.path.join(os.path.dirname(parts[0]), 'parts.txt')
    with open(listing, 'w') as fh:
        for part in parts:
            fh.write(f"file '{part}'\n")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat',
         '-safe', '0', '-i', listing, '-c', 'copy', output],
        check=True,
    )


def default_workers() -> tuple[int, int]:
    """(workers, render threads per worker) for this machine."""
    # Measured on a 12-CPU Docker Desktop VM (Apple Silicon) with the Kornaten
    # map: 4 x 3 and 6 x 2 are best (~14 fps), 12 x 1 is half that. The VM's
    # 12 CPUs do ~7 cores' worth of work in parallel, and llvmpipe processes
    # geometry on one thread per renderer, so a few threads each wins.
    cores = os.cpu_count() or 1
    threads = min(3, cores)
    return max(1, cores // threads), threads


def render(recording_path: str, output: str, speedup: float = 1.0,
           workers: int | None = None, threads: int | None = None,
           fps: float | None = None, quiet: bool = False) -> str:
    """Render a recording to ``output``. See the module docstring.

    Args:
        recording_path: The .npz the simulation node saved.
        output: The mp4 to write.
        speedup: Simulated seconds per second of video (1 = real speed).
        workers: Render processes; None picks from the core count.
        threads: llvmpipe threads per worker; None picks from the core count.
        fps: Video frame rate; None uses the run's.
        quiet: Do not print progress.

    Returns:
        ``output``.
    """
    if speedup <= 0:
        raise ValueError(f'speedup must be positive, got {speedup}')
    recording = Recording.load(recording_path)
    opts = dict(recording.render)
    if fps is not None:
        opts['fps'] = fps
    schedule = frame_schedule(recording, opts['fps'], speedup)
    if not schedule:
        raise ValueError(f'{recording_path}: the recording has no frames')

    auto_workers, auto_threads = default_workers()
    workers = max(1, min(workers or auto_workers, len(schedule)))
    threads = threads or auto_threads
    chunk = math.ceil(len(schedule) / workers)

    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    sim_seconds = len(schedule) * speedup / opts['fps']
    say(f'Rendering {len(schedule)} frames ({sim_seconds:.0f} s simulated at {speedup:g}x, '
        f'{opts["fps"]:g} fps) with {workers} workers x {threads} threads...')

    tmpdir = tempfile.mkdtemp(prefix='render_video_',
                              dir=os.path.dirname(os.path.abspath(output)))
    ctx = multiprocessing.get_context('spawn')  # fresh GL state per worker
    progress = ctx.Value('i', 0)
    jobs = [
        dict(recording=os.path.abspath(recording_path), render=opts, threads=threads,
             schedule=schedule[i * chunk:(i + 1) * chunk],
             output=os.path.join(tmpdir, f'part_{i:03d}.mp4'))
        for i in range(workers) if schedule[i * chunk:(i + 1) * chunk]
    ]
    start = time.monotonic()
    try:
        with ctx.Pool(len(jobs), initializer=_init_worker, initargs=(progress,)) as pool:
            result = pool.map_async(_render_part_shared, jobs)
            with tqdm(total=len(schedule), unit='frame', disable=quiet,
                      dynamic_ncols=True) as bar:
                while not result.ready():
                    result.wait(1)
                    bar.update(progress.value - bar.n)
                bar.update(progress.value - bar.n)
            parts = result.get()
        _concat(parts, output)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    elapsed = time.monotonic() - start
    say(f'Video saved: {output} ({len(schedule)} frames in {elapsed:.0f} s, '
        f'{len(schedule) / elapsed:.1f} fps)')
    return output


# A multiprocessing Value cannot be pickled into map() arguments; workers get
# it once, through the pool initializer.
_shared_progress = None


def _init_worker(progress):
    global _shared_progress
    _shared_progress = progress


def _render_part_shared(job: dict) -> str:
    job = dict(job, progress=_shared_progress)
    return _render_part(job)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Render the video of a recorded simulation run, in parallel.',
        epilog='See the module docstring (simulation/render_video.py) for run directories.')
    parser.add_argument('run', nargs='?', default=None,
                        help='a run directory, or a recording .npz '
                             f'(default: the latest run in {RUNS_DIR}/)')
    parser.add_argument('-o', '--output', default=None,
                        help='video to write (default: video_<speedup>x.mp4 next to the recording)')
    parser.add_argument('--speedup', type=float, default=None,
                        help="simulated seconds per second of video (default: the run's "
                             f"scenario, else {DEFAULT_SPEEDUP:g})")
    parser.add_argument('--fps', type=float, default=None,
                        help="video frame rate (default: the run's scenario, else 30)")
    auto_workers, auto_threads = default_workers()
    parser.add_argument('--workers', type=int, default=None,
                        help=f'render processes (default here: {auto_workers})')
    parser.add_argument('--threads', type=int, default=None,
                        help=f'render threads per process (default here: {auto_threads})')
    args = parser.parse_args(argv)

    target = args.run or latest_run()
    if target is None:
        parser.error(f'no run given, and none found in {RUNS_DIR}/')
    if os.path.isdir(target):
        run_dir, recording = target, os.path.join(target, RECORDING_NAME)
    else:
        run_dir, recording = os.path.dirname(os.path.abspath(target)), target
    settings = run_video_settings(run_dir)
    speedup = args.speedup or settings['speedup']
    fps = args.fps or settings['fps']
    output = args.output or os.path.join(run_dir, video_name(speedup))
    if args.run is None:
        print(f'Latest run: {run_dir}')
    try:
        render(recording, output, speedup=speedup, workers=args.workers,
               threads=args.threads, fps=fps)
    except (OSError, ValueError) as err:
        print(f'render_video: {err}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
