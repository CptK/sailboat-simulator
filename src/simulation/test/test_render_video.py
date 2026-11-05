"""
Tests for recording a run and rendering its video afterwards.

The point of offline rendering is that it draws exactly what live rendering
would have: the key test replays a recorded frame and compares it pixel for
pixel with the live simulation it was recorded from.

Rendering needs an OpenGL backend; on a headless machine set MUJOCO_GL=osmesa
(or egl). If no backend is available the rendering tests are skipped.
"""

import imageio
import mujoco
import numpy as np
import pytest

from simulation.recording import Recording, Scene, route_at
from simulation.render_video import (SegmentPlayer, frame_schedule, latest_run, render,
                                    run_video_settings, video_name)
from simulation.sailboat_simulation import SailboatSimulation

BUOYS = [(12.0, 3.0, 1.0, 0.5, 0.0), (20.0, -6.0, 0.0, 0.8, 0.0)]
SCENE = Scene(buoys=BUOYS, boat_east=0.0, boat_north=0.0, wind_direction=45.0,
              wind_speed=6.0)
ROUTE = [(0.0, 0.0), (15.0, 5.0), (30.0, -4.0)]
RENDER = dict(fps=30, width=160, height=120, onboard_inset_scale=0.25,
              minimap_inset_scale=0.25, minimap_view_size=300.0)
WIDTH, HEIGHT = 160, 120


def live_sim() -> SailboatSimulation:
    """A simulation built the way the node builds it for SCENE."""
    sim = SailboatSimulation(buoys=BUOYS, include_course_line=True,
                             wind_direction_deg=SCENE.wind_direction,
                             wind_speed=SCENE.wind_speed)
    sim.set_boat_position(SCENE.boat_east, SCENE.boat_north)
    return sim


def record_run(steps: int = 400, every: int = 7) -> tuple[Recording, SailboatSimulation]:
    """Sail a little, recording like the node does; route set, then changed."""
    sim = live_sim()
    rec = Recording(map_path='', rate=1 / (every * sim.model.opt.timestep), render=RENDER)
    rec.start_segment(SCENE)
    sim.set_rudder(10.0)
    sim.update_course_line(np.radians(20.0))
    sim.update_route(ROUTE)
    rec.set_route(ROUTE)
    t = 0.0
    for i in range(1, steps + 1):
        sim.step()
        t += sim.model.opt.timestep
        if i == steps // 2:
            sim.update_route(ROUTE[1:])
            rec.set_route(ROUTE[1:])
        if i % every == 0:
            rec.add_frame(t, sim.data, sim.in_collision())
    return rec, sim


def render_chase(sim: SailboatSimulation) -> np.ndarray:
    renderer = mujoco.Renderer(sim.model, width=WIDTH, height=HEIGHT)
    camera = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, 'chase_cam')
    renderer.update_scene(sim.data, camera)
    image = renderer.render()
    renderer.close()
    return image


@pytest.fixture
def require_gl():
    sim = live_sim()
    try:
        sim.render_onboard(WIDTH, HEIGHT)
    except Exception as exc:  # no GL context available
        pytest.skip(f"No OpenGL backend for rendering: {exc}")
    finally:
        sim.close()


# --- recording -----------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    rec, _ = record_run()
    path = str(tmp_path / 'rec.npz')
    rec.save(path)
    loaded = Recording.load(path)

    assert loaded.rate == pytest.approx(rec.rate)
    assert loaded.render == RENDER
    assert len(loaded.segments) == 1
    seg, orig = loaded.segments[0], rec.segments[0]
    assert loaded.frame_count == rec.frame_count > 0
    assert seg.scene == SCENE
    assert seg.time == pytest.approx(orig.time)
    np.testing.assert_array_equal(np.stack(seg.qpos), np.stack(orig.qpos))
    np.testing.assert_array_equal(np.stack(seg.mocap_pos), np.stack(orig.mocap_pos))
    assert seg.routes == orig.routes


def test_route_at_follows_the_route_events():
    rec, _ = record_run(steps=400, every=7)
    seg = rec.segments[0]
    change = seg.routes[1][0]
    assert route_at(seg, 0) == ROUTE
    assert route_at(seg, change - 1) == ROUTE
    assert route_at(seg, change) == ROUTE[1:]
    assert route_at(seg, len(seg.time) - 1) == ROUTE[1:]


def test_nothing_is_recorded_before_an_environment_exists():
    sim = live_sim()
    rec = Recording(map_path='', rate=30, render=RENDER)
    rec.add_frame(0.1, sim.data, False)
    rec.set_route(ROUTE)
    assert rec.frame_count == 0


# --- schedule ------------------------------------------------------------------


def schedule_for(times, fps=30, speedup=1.0):
    rec = Recording(map_path='', rate=30, render=RENDER)
    rec.start_segment(SCENE)
    rec.segments[0].time = list(times)
    return frame_schedule(rec, fps, speedup)


def test_schedule_at_the_recording_rate_uses_every_frame():
    times = np.arange(90) / 30
    assert [f for _, f in schedule_for(times)] == list(range(90))


def test_schedule_speedup_skips_frames_evenly():
    times = np.arange(90) / 30
    assert [f for _, f in schedule_for(times, speedup=3)] == list(range(0, 90, 3))


def test_schedule_picks_the_nearest_frame_to_each_video_time():
    # Recorded on the 5 ms physics grid: 30 or 35 ms apart, never exactly 1/30 s
    times = [0.0, 0.035, 0.065, 0.1, 0.135, 0.165, 0.2]
    picks = [f for _, f in schedule_for(times)]
    assert picks == [0, 1, 2, 3, 4, 5, 6]


def test_schedule_spans_segments_in_order():
    rec = Recording(map_path='', rate=30, render=RENDER)
    for start in (0.0, 10.0):
        rec.start_segment(SCENE)
        rec.segments[-1].time = list(start + np.arange(30) / 30)
    schedule = frame_schedule(rec, 30, 1.0)
    assert schedule[0] == (0, 0)
    assert (1, 29) == schedule[-1]
    segments = [s for s, _ in schedule]
    assert segments == sorted(segments)


# --- rendering -------------------------------------------------------------------


def test_replayed_frame_matches_the_live_render(require_gl):
    """The whole point: offline rendering draws what live rendering would have."""
    rec, sim = record_run(steps=400, every=8)  # the last frame is the final state
    # mj_step leaves body poses as they were before its integration, one 5 ms
    # step behind qpos; a replay computes them from the recorded qpos. Bring
    # the live model up to date to compare like with like.
    mujoco.mj_forward(sim.model, sim.data)
    live = render_chase(sim)

    player = SegmentPlayer(rec.segments[0])
    player.show(len(rec.segments[0].time) - 1)
    replayed = render_chase(player.sim)

    np.testing.assert_array_equal(replayed, live)
    np.testing.assert_array_equal(player.sim.render_onboard(WIDTH, HEIGHT),
                                  sim.render_onboard(WIDTH, HEIGHT))
    player.close()
    sim.close()


def test_render_writes_one_video_from_parallel_parts(require_gl, tmp_path):
    rec, _ = record_run(steps=600, every=7)
    path = str(tmp_path / 'rec.npz')
    rec.save(path)
    out = str(tmp_path / 'out.mp4')

    render(path, out, workers=3, threads=1, quiet=True)

    expected = len(frame_schedule(rec, RENDER['fps'], 1.0))
    reader = imageio.get_reader(out)
    frames = [frame for frame in reader]
    reader.close()
    assert len(frames) == expected
    assert frames[0].shape == (HEIGHT, WIDTH, 3)



# --- run directories ---------------------------------------------------------------


def test_video_is_named_after_its_speedup():
    assert video_name(4.0) == 'video_4x.mp4'
    assert video_name(1.5) == 'video_1.5x.mp4'


def test_run_settings_come_from_the_runs_scenario(tmp_path):
    (tmp_path / 'scenario.yaml').write_text('map: x\nvideo:\n  speedup: 2.5\n  fps: 24\n')
    assert run_video_settings(str(tmp_path)) == {'speedup': 2.5, 'fps': 24, 'enabled': True}


def test_run_settings_default_without_a_video_section(tmp_path):
    (tmp_path / 'scenario.yaml').write_text('map: x\n')
    assert run_video_settings(str(tmp_path)) == {'speedup': 4.0, 'fps': None, 'enabled': True}
    assert run_video_settings(str(tmp_path / 'missing'))['speedup'] == 4.0


def test_latest_run_is_the_newest_with_a_recording(tmp_path):
    import os
    for name, mtime, has_recording in (('a', 100, True), ('b', 300, True), ('c', 500, False)):
        run = tmp_path / name
        run.mkdir()
        if has_recording:
            (run / 'recording.npz').write_bytes(b'')
        os.utime(run, (mtime, mtime))
    assert latest_run(str(tmp_path)) == str(tmp_path / 'b')
    assert latest_run(str(tmp_path / 'none')) is None


def test_render_video_cli_renders_a_run_directory(require_gl, tmp_path):
    from simulation.render_video import main
    rec, _ = record_run(steps=300, every=7)
    rec.save(str(tmp_path / 'recording.npz'))
    (tmp_path / 'scenario.yaml').write_text('map: x\nvideo:\n  speedup: 2\n')

    main([str(tmp_path), '--workers', '2', '--threads', '1'])

    assert (tmp_path / 'video_2x.mp4').is_file()
