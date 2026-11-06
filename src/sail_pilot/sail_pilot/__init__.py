"""Pilots that steer the simulated sailboat from what it sees, and the harness
they sail in.

    sailing      the no-go zone, points of sail, when a turn is a tack
    task         what the pilot is asked to do, and success
    observation  what a pilot is shown each decision, and its answer
    render       camera-image marks and the chart
    harness      PilotRun: physics + controller node + pilot, tacks, groundings
    pilots/      ClaudePilot (Messages API) and ScriptedPilot (privileged baseline)
    output       run directories and the trajectory plot
    cli          the command line behind scripts/llm_pilot.py

Importing the package puts the repository's ROS packages (src/) and scripts/
on sys.path, as the harness imports from both. Needs the ROS environment
(source install/setup.bash).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _pkg in ("simulation", "controller", "route_follower", "graph_route_planner"):
    sys.path.insert(0, str(REPO / "src" / _pkg))
sys.path.insert(0, str(REPO / "scripts"))
