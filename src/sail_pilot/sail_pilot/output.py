"""Run directories and the trajectory plot."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from . import REPO
from .harness import PilotRun
from .observation import Decision

RUNS_DIR = REPO / "runs"


def shown_path(path: Path) -> str:
    """A path for messages: relative to the repository when inside it."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def plot_trajectory(run: PilotRun, trace: list, decisions: list[Decision], path: Path,
                    title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from render_maps import render_map

    fig, ax = plt.subplots(figsize=(9, 9))
    render_map(ax, run.map, 0.0, title)
    if trace:
        xs, ys = [r[1] for r in trace], [r[2] for r in trace]
        ax.plot(xs, ys, "-", color="#c0392b", lw=1.5, zorder=6, label="track")
        # Zoom to the action, with some context.
        x0, x1 = min(xs + [run.start[0]]), max(xs + [run.start[0]])
        y0, y1 = min(ys + [run.start[1]]), max(ys + [run.start[1]])
        if run.task.target:
            x0, x1 = min(x0, run.task.target[0]), max(x1, run.task.target[0])
            y0, y1 = min(y0, run.task.target[1]), max(y1, run.task.target[1])
        pad = max(20.0, 0.25 * max(x1 - x0, y1 - y0))
        mx0, my0, mx1, my1 = run.map.bounds
        ax.set_xlim(max(mx0, x0 - pad), min(mx1, x1 + pad))
        ax.set_ylim(max(my0, y0 - pad), min(my1, y1 + pad))
        # Each decision: a dot, and a tick along the course it set.
        times = np.array([r[0] for r in trace])
        for d in decisions:
            r = trace[int(np.argmin(np.abs(times - d.t)))] if len(times) else None
            if r is None:
                continue
            ax.plot(r[1], r[2], "o", color="black" if not d.error else "orange", ms=3, zorder=7)
            if not d.error:
                a = math.radians(d.applied_heading)
                ax.plot([r[1], r[1] + 6 * math.sin(a)], [r[2], r[2] + 6 * math.cos(a)],
                        "-", color="black", lw=0.8, zorder=7)
    for b in run.scenario.buoys:
        ax.plot(b.east, b.north, "o", color=b.rgb, mec="black", ms=9, zorder=8)
    if run.task.target:
        ax.add_patch(matplotlib.patches.Circle(run.task.target, run.task.radius, fill=False,
                                               ec="black", ls="--", zorder=8))
    ax.plot(*run.start, "k^", ms=10, zorder=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def new_run_dir(scenario_name: str) -> Path:
    """A fresh run directory, with frames/; runs started in the same second
    (e.g. in parallel) get _2, _3, ... instead of sharing one."""
    base = RUNS_DIR / f"llm_{scenario_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        run_dir = base if n == 1 else base.with_name(f"{base.name}_{n}")
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        (run_dir / "frames").mkdir()
        return run_dir
    raise RuntimeError(f"could not create a run directory next to {base}")
