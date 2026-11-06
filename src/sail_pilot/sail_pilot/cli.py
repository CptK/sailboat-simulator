"""Let a vision-language model steer the boat: camera image + instruments in, course change out.

    python3 scripts/llm_pilot.py landgraben --target green
    python3 scripts/llm_pilot.py landgraben --target green --mode realtime
    python3 scripts/llm_pilot.py kornaten --target red --model claude-haiku-5-5 --effort low
    python3 scripts/llm_pilot.py landgraben --target green --pilot scripted   # no API needed

Each decision, the pilot gets
  * the onboard camera image, with a bearing scale (degrees off the bow) drawn on it,
  * instruments: compass heading, speed, wind relative to the bow, touching land,
  * the task ("Sail to the green buoy.") and its own last few decisions,
and answers with a short observation, a course change in degrees (+ = starboard)
and whether it thinks the task is done. The controller then holds that course
until the next decision — the same controller node, with the same parameters,
as the live system. The planner and route follower are not used.

Two modes:
  paused    (default) the simulation stops while the model thinks; a decision
            every --period simulated seconds. Separates "can it steer" from
            "can it keep up", and runs are comparable.
  realtime  the simulation runs on at wall-clock speed while a request is out;
            the next request goes out as soon as the answer arrives (no sooner
            than --period seconds after the previous one). The answer is
            applied to the heading the boat had when the image was taken.

A run ends when the task succeeds (boat within --radius of the target buoy),
when the pilot says it is done, after --max-time simulated seconds, or after
too many failed requests in a row. Everything goes to runs/llm_<scenario>_<time>/:

    scenario.yaml      the scenario as run (render_video reads its video section)
    run.json           settings and the summary (success, time, latency, tokens, cost)
    decisions.jsonl    one line per decision: state, model answer, latency, usage
    frames/NNNN.jpg    each image exactly as it was sent
    trace.csv          boat state at 10 Hz
    trajectory.png     the track on the map, with each decision
    recording.npz      for the video; video_<speedup>x.mp4 unless --no-video

Requires the ROS environment (source install/setup.bash), as scripts/closed_loop.py.
The Claude pilot needs credentials: ANTHROPIC_API_KEY in the environment or in
the repository's .env (git- and docker-ignored; copy .env.example).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from scenario import load_scenario

from . import REPO
from .harness import PilotRun
from .observation import Decision, Observation
from .output import new_run_dir, plot_trajectory, shown_path
from .pilots.claude import THINKING_OFF, ClaudePilot, estimate_cost
from .pilots.scripted import ScriptedPilot
from .render import CHART_VIEW
from .sailing import GUARD_DEG
from .task import make_task


def load_dotenv(path: Path = REPO / ".env") -> list[str]:
    """Put KEY=VALUE lines of a .env file into os.environ.

    Variables already set in the environment win, so `export` or
    `docker run -e` still override the file. Blank lines and # comments are
    skipped; an `export ` prefix and quotes around the value are allowed.

    Returns:
        The names of the variables it set.
    """
    import os

    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def default_memory(instruction: str | None, target: str | None) -> bool:
    """Memory is on for free instructions: they have several steps, and steps
    done must not drop out of the few decisions the model is shown. A single
    --target needs none."""
    return bool(instruction) and target is None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog=__doc__.split("\n\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", help="scenario name (scenarios/<name>.yaml) or path")
    parser.add_argument("--target", help="buoy to sail to: a colour, or its number in the scenario")
    parser.add_argument("--instruction", help="the task in words (default: from --target)")
    parser.add_argument("--radius", type=float, default=5.0,
                        help="success: within this many metres of the target (default 5)")
    parser.add_argument("--pilot", choices=("claude", "scripted"), default="claude")
    parser.add_argument("--model", default="claude-haiku-5-5",
                        help="any Claude model id, e.g. claude-haiku-5-5 (default), "
                             "claude-sonnet-5-5, claude-opus-5-5")
    parser.add_argument("--effort", default="low",
                        choices=("low", "medium", "high", "xhigh", "max", "default"),
                        help="the model's effort (default low, for speed)")
    parser.add_argument("--chart", action="store_true",
                        help="also send a top-down chart (north up, buoys, boat, its track)")
    parser.add_argument("--chart-view", type=float, default=CHART_VIEW,
                        help=f"widest area (m) the chart shows whole (default {CHART_VIEW:g}); "
                             "larger maps follow the boat")
    parser.add_argument("--no-guard", action="store_true",
                        help="let the autopilot steer into the no-go zone if asked")
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=None,
                        help="the model keeps a running progress note it sees next time "
                             "(default: on with --instruction, off with --target)")
    parser.add_argument("--stop-on-grounding", type=float, default=10.0, metavar="SECONDS",
                        help="end the run once the boat is stuck on land this long "
                             "(default 10; 0 sails on)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="turn the model's thinking off (faster; effort high or below)")
    parser.add_argument("--mode", choices=("paused", "realtime"), default="paused")
    parser.add_argument("--period", type=float, default=None,
                        help="paused: simulated seconds between decisions (default 2); "
                             "realtime: least seconds between requests (default 0)")
    parser.add_argument("--max-time", type=float, default=240.0,
                        help="simulated seconds before giving up (default 240)")
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 480),
                        metavar=("W", "H"), help="onboard image sent (default 640 480)")
    parser.add_argument("--start", type=float, nargs=2, metavar=("EAST", "NORTH"),
                        help="start somewhere else than the scenario's boat")
    parser.add_argument("--heading", type=float, help="initial compass heading")
    parser.add_argument("--ignore-done", action="store_true",
                        help="keep going when the pilot says it is done")
    parser.add_argument("--scripted-delay", type=float, default=0.0,
                        help="scripted pilot: seconds per decision, to mimic API latency")
    parser.add_argument("--no-video", action="store_true", help="do not render the video")
    parser.add_argument("--speedup", type=float, help="video speedup (default: the scenario's)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.pilot == "claude" and args.no_thinking:
        if THINKING_OFF.get(args.model, "disabled") is None:
            parser.error(f"thinking cannot be turned off on {args.model}; "
                         "use a lower --effort instead")
        if args.effort in ("xhigh", "max"):
            parser.error("--no-thinking needs --effort high or below")
    if args.memory is None:
        args.memory = default_memory(args.instruction, args.target)
    scenario = load_scenario(args.scenario)
    task = make_task(scenario, args.target, args.instruction, args.radius)
    period = args.period if args.period is not None else (2.0 if args.mode == "paused" else 0.0)
    period_note = ("The simulation waits while you decide; your next decision comes "
                   f"{period:g} s later." if args.mode == "paused" else
                   "The boat keeps sailing while you decide.")

    if args.pilot == "claude":
        load_dotenv()
        pilot = ClaudePilot(model=args.model,
                            effort=None if args.effort == "default" else args.effort,
                            thinking=not args.no_thinking, period_note=period_note,
                            memory=args.memory,
                            guard_deg=None if args.no_guard else GUARD_DEG)
        try:
            # Free, and fails fast on a missing, wrong or expired key, before
            # the simulation is built.
            pilot.client.models.retrieve(args.model)
        except Exception as err:   # noqa: BLE001 - any failure here means no access
            print(f"Cannot use the Anthropic API with model {args.model}: {err}\n"
                  "Put ANTHROPIC_API_KEY in .env (copy .env.example) or the "
                  "environment; see README.md, section 'LLM pilot'.\n"
                  "To try the harness without the API: --pilot scripted")
            return 2
    else:
        pilot = ScriptedPilot(task, delay=args.scripted_delay)

    run_dir = new_run_dir(scenario.name)
    shutil.copyfile(scenario.path, run_dir / "scenario.yaml")
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    say(f"Run directory: {shown_path(run_dir)}")
    say(f"Task: {task.instruction}  (pilot {args.pilot}"
        + (f", {args.model}, effort {args.effort}" + (", no thinking" if args.no_thinking else "")
           + (", memory" if args.memory else "") + (", chart" if args.chart else "")
           + (", no guard" if args.no_guard else "")
           if args.pilot == "claude" else "") + f", {args.mode})")

    harness = PilotRun(scenario, task, start=args.start, heading=args.heading,
                       image_size=tuple(args.image_size), guard=not args.no_guard,
                       chart=args.chart, chart_view=args.chart_view)
    if isinstance(pilot, ClaudePilot):
        pilot.half_fov = harness.half_fov
    log = open(run_dir / "decisions.jsonl", "w")

    def on_decision(obs: Observation, d: Decision) -> None:
        frame = run_dir / "frames" / f"{obs.index:04d}.jpg"
        frame.write_bytes(obs.jpeg)
        if obs.chart_jpeg is not None:
            (run_dir / "frames" / f"{obs.index:04d}_chart.jpg").write_bytes(obs.chart_jpeg)
        dist = task.distance(obs.east, obs.north)
        entry = {"index": obs.index, "t": round(obs.t, 2), "image": str(frame.name),
                 "state": {"east": round(obs.east, 2), "north": round(obs.north, 2),
                           "heading": round(obs.heading, 1), "speed": round(obs.speed, 2),
                           "wind_from_rel": round(obs.wind_from_rel, 1),
                           "colliding": obs.colliding,
                           "true_land_ahead": (None if obs.true_land_ahead is None
                                               else round(obs.true_land_ahead, 1)),
                           "target_distance": None if dist is None else round(dist, 1)},
                 "prompt": obs.instruments_text(period_note), **asdict(d),
                 "new_heading": round(d.new_heading, 1),
                 "applied_heading": round(d.applied_heading, 1)}
        log.write(json.dumps(entry) + "\n")
        log.flush()
        status = (f"ERROR {d.error}" if d.error else
                  f"turn {d.turn_deg:+5.0f}° -> {d.new_heading:5.1f}°"
                  + (f" (no-go: steering {d.steered_heading:.0f}°)"
                     if d.steered_heading is not None else "")
                  + ("  DONE" if d.done else "")
                  + (f"  ({d.maneuver})" if d.maneuver else "")
                  + (f"  [land ahead: {d.land_ahead}]" if d.land_ahead else "")
                  + f"  {d.observation}")
        say(f"[{obs.t:6.1f} s] #{obs.index:<3} {d.latency:4.1f} s"
            + (f"  dist {dist:5.1f} m" if dist is not None else "") + f"  {status}")
        if d.progress:
            say(f"{'':22}progress: {d.progress}")

    trace: list = []
    try:
        result = harness.run(pilot, mode=args.mode, period=period, max_time=args.max_time,
                             stop_on_done=not args.ignore_done, on_decision=on_decision,
                             trace=trace, stop_on_grounding=args.stop_on_grounding)
    except KeyboardInterrupt:
        say("Interrupted.")
        return 130
    finally:
        log.close()
        harness.recording.save(str(run_dir / "recording.npz"))
        harness.close()

    summary = result.summary()
    tacks = harness.maneuvers
    summary["tacks"] = len(tacks)
    summary["tacks_failed"] = sum(1 for m in tacks if not m["outcome"].startswith("tacked"))
    model = args.model if args.pilot == "claude" else None
    cost = estimate_cost(model, summary["input_tokens"], summary["output_tokens"])
    summary["cost_usd"] = None if cost is None else round(cost, 4)
    with open(run_dir / "trace.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "east", "north", "heading", "desired_heading", "speed", "colliding"])
        writer.writerows(trace)
    settings = {**pilot.settings(), "scenario": scenario.name, "task": asdict(task),
                "mode": args.mode, "period": period, "max_time": args.max_time,
                "image_size": list(args.image_size), "guard": not args.no_guard,
                "chart": args.chart, "chart_view": args.chart_view,
                "memory": args.memory, "stop_on_grounding": args.stop_on_grounding}
    (run_dir / "run.json").write_text(json.dumps({"settings": settings, "summary": summary,
                                                  "maneuvers": tacks}, indent=2))
    plot_trajectory(harness, trace, result.decisions, run_dir / "trajectory.png",
                    f"{task.instruction} — {'success' if result.success else 'no success'} "
                    f"({result.end_reason}), {result.sim_time:.0f} s")

    say("")
    say(("SUCCESS: " if result.success else "NO SUCCESS: ") + result.end_reason)
    say(f"  {summary['sim_time']} s simulated, {summary['wall_time']} s wall, "
        f"{summary['decisions']} decisions ({summary['errors']} failed)")
    if summary["min_distance"] is not None:
        say(f"  closest to target {summary['min_distance']} m, "
            f"finished {summary['final_distance']} m away")
    if tacks:
        say(f"  {len(tacks)} tacks, {summary['tacks_failed']} of them not completed as a tack")
    if summary["groundings"]:
        say(f"  ran aground {summary['groundings']} times, "
            f"{summary['time_aground']} s on land")
    if summary["latency_mean"] is not None and args.pilot == "claude":
        say(f"  latency mean {summary['latency_mean']} s, p90 {summary['latency_p90']} s, "
            f"max {summary['latency_max']} s")
        say(f"  tokens {summary['input_tokens']} in / {summary['output_tokens']} out"
            + (f", about ${cost:.4f}" if cost is not None else ""))
    say(f"  {shown_path(run_dir)}/trajectory.png, decisions.jsonl, frames/")

    if not args.no_video and scenario.video_enabled:
        from simulation.render_video import render, video_name

        speedup = args.speedup or scenario.video_speedup
        render(str(run_dir / "recording.npz"), str(run_dir / video_name(speedup)),
               speedup=speedup, quiet=args.quiet)
    return 0 if result.success else 1
