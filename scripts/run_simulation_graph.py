#!/usr/bin/env python3
"""
Script to launch the ROS2 sailboat simulation using the graph-based planner.

The default scenario, open_water, is the same environment as run_simulation.py
— same boat, buoys, wind and via points. The one difference is who decides the
route between the via points:

    run_simulation.py       publishes the via points to /planning/target_route,
                            and route_planner sails straight at each in turn.

    run_simulation_graph.py publishes the via points to /planning/mission, and
                            graph_route_planner plans the water between them
                            first — tacking upwind where a direct heading is
                            unsailable — then hands the expanded route to
                            route_planner on /planning/target_route.

With the wind from the north, the leg north up the east side is dead upwind, so
that is where the two differ: the graph planner tacks it.

Usage:
    python scripts/run_simulation_graph.py                          # open_water (default)
    python scripts/run_simulation_graph.py --scenario landgraben    # a file in scenarios/
    python scripts/run_simulation_graph.py --scenario my/course.yaml
    python scripts/run_simulation_graph.py --list                   # list scenarios
    python scripts/run_simulation_graph.py --check -s kornaten      # validate, do not run

A scenario is a YAML file (see scenarios/README.md): the map, the boat's start,
wind, buoys, the mission via points and planner settings. It is checked —
structure, and that everything is on water — before anything is launched.
"""

import argparse
import subprocess
import time
import signal
import sys
import os
import threading

from scenario import (Scenario, ScenarioError, check_positions, list_scenarios,
                      load_scenario)

# =============================================================================
# CONFIGURATION - scenarios live in scenarios/*.yaml; these are timings only
# =============================================================================

DEFAULT_SCENARIO = "open_water"

# Delay settings (seconds)
STARTUP_DELAY = 5.0  # Time to wait for ROS system to initialize
MESSAGE_DELAY = 0.5  # Time between sending messages
# How long to let the graph planner expand the mission is per scenario
# (planner: planning_delay), since large maps need longer.
SHUTDOWN_DELAY = 5.0  # Time to wait after route completion before shutting down

# =============================================================================
# END CONFIGURATION
# =============================================================================


def build_env_create_message(scenario: Scenario) -> str:
    """Build the EnvironmentCreation message as a YAML string."""
    buoy_list = ", ".join(
        f"{{east: {b.east}, north: {b.north}, "
        f"color: {{r: {b.rgb[0]}, g: {b.rgb[1]}, b: {b.rgb[2]}}}}}"
        for b in scenario.buoys
    )
    east, north = scenario.boat
    return (f"{{boat_east: {east}, boat_north: {north}, buoys: [{buoy_list}], "
            f"wind_direction: {scenario.wind_direction}, wind_speed: {scenario.wind_speed}}}")


def build_wind_message(scenario: Scenario) -> str:
    """Build the Wind message as a YAML string."""
    return f"{{angle: {scenario.wind_direction}, speed: {scenario.wind_speed}}}"


def build_mission_message(scenario: Scenario) -> str:
    """Build the mission (Route of via points) as a YAML string."""
    waypoint_list = ", ".join(
        f"{{east: {e}, north: {n}}}" for e, n in scenario.waypoints
    )
    return f"{{waypoints: [{waypoint_list}]}}"


def print_scenarios() -> None:
    """List the scenario files with the first line of their description."""
    for name in list_scenarios():
        try:
            sc = load_scenario(name)
            summary = sc.description.split(". ")[0].rstrip(".")
            print(f"  {name:<16} map {sc.map:<14} {summary}")
        except ScenarioError as err:
            print(f"  {name:<16} INVALID: {err}")


def load_and_check(spec: str) -> Scenario | None:
    """Load a scenario and check it against its map; print problems.

    Returns:
        The scenario, or None if it cannot be run.
    """
    try:
        scenario = load_scenario(spec)
    except ScenarioError as err:
        print(f"Scenario error: {err}")
        return None
    try:
        errors, warnings = check_positions(scenario)
    except (FileNotFoundError, ValueError) as err:
        print(f"Scenario error: {scenario.path}: map '{scenario.map}': {err}")
        return None
    for warning in warnings:
        print(f"Warning ({scenario.name}): {warning}")
    if errors:
        print(f"Scenario '{scenario.name}' ({scenario.path}) cannot be run:")
        for error in errors:
            print(f"  - {error}")
        print("Check positions with: python scripts/render_maps.py "
              f"{scenario.map} --scenarios")
        return None
    return scenario


def publish_message(topic: str, msg_type: str, message: str) -> None:
    """Publish a single message to a ROS2 topic."""
    cmd = ["ros2", "topic", "pub", "--once", topic, msg_type, message]
    print(f"Publishing to {topic}: {message}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Scenarios: " + ", ".join(list_scenarios()))
    parser.add_argument("--scenario", "-s", default=DEFAULT_SCENARIO,
                        help=f"scenario name (a file in scenarios/) or path to a YAML "
                             f"file (default: {DEFAULT_SCENARIO})")
    # Older name for the same option, from when scenarios were named after maps.
    parser.add_argument("--map", dest="scenario", help=argparse.SUPPRESS)
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--check", action="store_true",
                        help="validate the scenario against its map and exit")
    args = parser.parse_args()

    if args.list:
        print_scenarios()
        return
    scenario = load_and_check(args.scenario)
    if scenario is None:
        sys.exit(1)
    if args.check:
        print(f"Scenario '{scenario.name}' is valid: map {scenario.map}, "
              f"{len(scenario.waypoints)} via points, {len(scenario.buoys)} buoys.")
        return

    ros_process = None
    shutting_down = False

    def shutdown():
        """Shut down the ROS process."""
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True

        print("\nShutting down...")
        if ros_process and ros_process.poll() is None:
            # Send SIGINT to the process group
            try:
                os.killpg(os.getpgid(ros_process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

            # Wait briefly for graceful shutdown
            try:
                ros_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                print("Force killing...")
                try:
                    os.killpg(os.getpgid(ros_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                ros_process.wait()

    def signal_handler(sig, frame):
        """Handle Ctrl+C to cleanly shut down."""
        shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("Sailboat Simulation Launcher (graph-based planner)")
    print("=" * 60)
    print(f"Scenario: {scenario.name} ({scenario.path})")
    print(f"Map: {scenario.map}")
    print(f"Boat position: {scenario.boat}")
    print("Buoys: " + (", ".join(f"{b.color or b.rgb} at ({b.east:g}, {b.north:g})"
                                 for b in scenario.buoys) or "none"))
    print(f"Wind: {scenario.wind_direction:g} deg @ {scenario.wind_speed:g} m/s")
    print(f"Mission via points: {scenario.waypoints}")
    print("=" * 60)

    # 1. Launch the ROS system, including the graph planner
    print("\n[1/5] Launching ROS system (with graph_route_planner)...")
    ros_process = subprocess.Popen(
        ["ros2", "launch", "launch/launch_graph.py",
         f"map:={scenario.map}", f"grid_spacing:={scenario.grid_spacing}",
         f"max_leg_distance:={scenario.max_leg_distance}"],
        start_new_session=True,
    )

    print(f"Waiting {STARTUP_DELAY}s for system to initialize...")
    time.sleep(STARTUP_DELAY)

    # 2. Send environment creation message
    print("\n[2/5] Initializing environment...")
    publish_message(
        "/env/create",
        "boat_msgs/msg/EnvironmentCreation",
        build_env_create_message(scenario)
    )
    time.sleep(MESSAGE_DELAY)

    # 3. Send wind message
    # The graph planner needs the wind before it can plan: the route depends on it.
    print("\n[3/5] Setting wind conditions...")
    publish_message(
        "/env/wind",
        "boat_msgs/msg/Wind",
        build_wind_message(scenario)
    )
    time.sleep(MESSAGE_DELAY)

    # 4. Send the mission. Unlike run_simulation.py, these via points are not the
    #    route: the graph planner expands them into one on /planning/target_route.
    print("\n[4/5] Sending mission via points to the graph planner...")
    publish_message(
        "/planning/mission",
        "boat_msgs/msg/Route",
        build_mission_message(scenario)
    )

    # 5. Report what the planner made of it.
    planning_delay = scenario.planning_delay
    print(f"\n[5/5] Waiting {planning_delay}s for the planner to expand the mission...")
    time.sleep(planning_delay)
    try:
        # The route is latched (RELIABLE + TRANSIENT_LOCAL), and it was published
        # seconds ago, so this late-joining reader must match BOTH of those to be
        # handed the held sample. A default reader receives it only by luck, and
        # setting durability alone (without reliability) matches neither.
        result = subprocess.run(
            ["ros2", "topic", "echo", "--once", "--timeout", "20",
             "--qos-reliability", "reliable", "--qos-durability", "transient_local",
             "/planning/target_route", "boat_msgs/msg/Route"],
            capture_output=True, text=True, timeout=25,
        )
        planned = result.stdout.count("east:")
        if planned:
            print(f"Planner expanded {len(scenario.waypoints)} via points into {planned} waypoints.")
        else:
            print("No route published yet — check the graph_route_planner log above.")
    except Exception as err:
        print(f"Could not read the planned route: {err}")

    print("\n" + "=" * 60)
    print("Simulation running. Press Ctrl+C to stop.")
    print("Waiting for route completion...")
    print("=" * 60)

    # Monitor for route completion in a background thread
    route_completed = threading.Event()

    def monitor_route_completion():
        """Monitor the route completion topic."""
        try:
            result = subprocess.run(
                ["ros2", "topic", "echo", "--once", "/planning/route_completed", "std_msgs/msg/Bool"],
                capture_output=True,
                text=True,
            )
            if "data: true" in result.stdout.lower():
                route_completed.set()
        except Exception:
            pass

    monitor_thread = threading.Thread(target=monitor_route_completion, daemon=True)
    monitor_thread.start()

    # Keep running until interrupted or route completed
    try:
        while ros_process.poll() is None:
            if route_completed.is_set():
                print("\n" + "=" * 60)
                print("Route completed!")
                print(f"Shutting down in {SHUTDOWN_DELAY} seconds...")
                print("=" * 60)
                time.sleep(SHUTDOWN_DELAY)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
