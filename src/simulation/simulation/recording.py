"""Record a simulation run, so its video can be rendered afterwards.

Rendering on a machine without a GPU is far slower than the physics: a
1280x720 frame with the onboard inset takes about 80 ms, so rendering live
either slows the run down or makes the video jerky. Instead, the simulation
node records what each frame would show and render_video.py renders the
frames afterwards, in parallel.

A recording holds one or more segments. Each segment is one environment, built
from its /env/create message, so a renderer can rebuild the same MuJoCo model.
A segment stores, per frame:

    time        simulated seconds (the node's clock, which never runs backwards)
    qpos        joint positions: boat pose, heel, boom, rudder, ...
    mocap_pos   positions of the overlay markers (wind arrow, masthead vane,
    mocap_quat  course line, route segments)
    collision   whether the boat is touching land

and, as events, the route as it changes (the route's segment lengths live in
the model, not in qpos or mocap, and the minimap draws it too).

That is everything rendering reads from the simulation: given the model, a
frame is fully determined by qpos and mocap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

#: File format version, stored in the recording.
FORMAT_VERSION = 1


@dataclass
class Scene:
    """What a segment's MuJoCo model is built from (an /env/create message)."""

    #: (east, north, r, g, b) per buoy, colour channels in [0, 1].
    buoys: list[tuple[float, float, float, float, float]]
    boat_east: float
    boat_north: float
    wind_direction: float
    wind_speed: float


@dataclass
class Segment:
    """The frames recorded in one environment."""

    scene: Scene
    time: list[float] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    mocap_pos: list[np.ndarray] = field(default_factory=list)
    mocap_quat: list[np.ndarray] = field(default_factory=list)
    collision: list[bool] = field(default_factory=list)
    #: (index of the first frame it applies to, waypoints) per route change.
    routes: list[tuple[int, list[tuple[float, float]]]] = field(default_factory=list)


@dataclass
class Recording:
    """A whole run: the settings to render it with, and its segments.

    Attributes:
        map_path: The map the run used (its land is part of every model).
        rate: Frames recorded per simulated second.
        render: Rendering settings of the run (video size, insets), used as
            defaults by the renderer.
        segments: One per environment, in order.
    """

    map_path: str
    rate: float
    render: dict
    segments: list[Segment] = field(default_factory=list)

    # --- recording -----------------------------------------------------------

    def start_segment(self, scene: Scene):
        """Begin recording in a new environment."""
        self.segments.append(Segment(scene=scene))

    def add_frame(self, sim_time: float, data, collision: bool):
        """Record the state of MuJoCo ``data`` as the next frame."""
        if not self.segments:
            return
        seg = self.segments[-1]
        seg.time.append(float(sim_time))
        seg.qpos.append(np.array(data.qpos, dtype=np.float64))
        seg.mocap_pos.append(np.array(data.mocap_pos, dtype=np.float32))
        seg.mocap_quat.append(np.array(data.mocap_quat, dtype=np.float32))
        seg.collision.append(bool(collision))

    def set_route(self, waypoints: list[tuple[float, float]]):
        """The route drawn from the next recorded frame on; empty clears it."""
        if not self.segments:
            return
        seg = self.segments[-1]
        seg.routes.append((len(seg.time), [(float(e), float(n)) for e, n in waypoints]))

    @property
    def frame_count(self) -> int:
        return sum(len(seg.time) for seg in self.segments)

    # --- files ---------------------------------------------------------------

    def save(self, path: str):
        """Write the recording to ``path`` (.npz), atomically."""
        meta = {
            "version": FORMAT_VERSION,
            "map_path": self.map_path,
            "rate": self.rate,
            "render": self.render,
            "segments": [
                {"scene": vars(seg.scene), "routes": seg.routes}
                for seg in self.segments
            ],
        }
        arrays = {"meta": np.array(json.dumps(meta))}
        for i, seg in enumerate(self.segments):
            arrays[f"time_{i}"] = np.array(seg.time, dtype=np.float64)
            arrays[f"qpos_{i}"] = _stack(seg.qpos, np.float64)
            arrays[f"mocap_pos_{i}"] = _stack(seg.mocap_pos, np.float32)
            arrays[f"mocap_quat_{i}"] = _stack(seg.mocap_quat, np.float32)
            arrays[f"collision_{i}"] = np.array(seg.collision, dtype=bool)
        # Write next to the target and rename, so an interrupted save never
        # leaves a truncated file where a good one was.
        tmp = f"{path}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Recording":
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            if meta.get("version") != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported recording version {meta.get('version')}")
            recording = cls(map_path=meta["map_path"], rate=meta["rate"],
                            render=meta["render"])
            for i, seg_meta in enumerate(meta["segments"]):
                scene = seg_meta["scene"]
                scene["buoys"] = [tuple(b) for b in scene["buoys"]]
                seg = Segment(scene=Scene(**scene))
                seg.time = list(npz[f"time_{i}"])
                seg.qpos = list(npz[f"qpos_{i}"])
                seg.mocap_pos = list(npz[f"mocap_pos_{i}"])
                seg.mocap_quat = list(npz[f"mocap_quat_{i}"])
                seg.collision = [bool(c) for c in npz[f"collision_{i}"]]
                seg.routes = [(int(start), [tuple(wp) for wp in wps])
                              for start, wps in seg_meta["routes"]]
                recording.segments.append(seg)
        return recording


def _stack(rows: list[np.ndarray], dtype) -> np.ndarray:
    if rows:
        return np.stack(rows).astype(dtype)
    return np.zeros((0, 0), dtype=dtype)


def route_at(segment: Segment, frame: int) -> list[tuple[float, float]] | None:
    """The route drawn at ``frame`` of ``segment``; None if none was set yet."""
    route = None
    for start, waypoints in segment.routes:
        if start > frame:
            break
        route = waypoints
    return route
