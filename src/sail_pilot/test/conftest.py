"""Shared fixtures. No API calls anywhere: the Claude pilot gets a fake client.

    source install/setup.bash && pytest src/sail_pilot/test

Tests that build a PilotRun need the ROS environment and are skipped without it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sail_pilot  # noqa: E402,F401 - puts src/ packages and scripts/ on sys.path


@pytest.fixture(scope="module")
def scenario():
    from scenario import load_scenario

    return load_scenario("landgraben")


@pytest.fixture(scope="module")
def orange(scenario):
    """The orange buoy (east, north): open water between it and the boat's start."""
    (buoy,) = [b for b in scenario.buoys if b.color == "orange"]
    return (buoy.east, buoy.north)


@pytest.fixture
def make_run(scenario, orange):
    """PilotRun factory; the default task is sailing to the orange buoy."""
    from sail_pilot.harness import PilotRun
    from sail_pilot.task import Task

    runs = []

    def make(task=None, **kw):
        run = PilotRun(scenario, task or Task("Sail to the orange buoy.", orange), **kw)
        runs.append(run)
        return run
    yield make
    for run in runs:
        run.close()
