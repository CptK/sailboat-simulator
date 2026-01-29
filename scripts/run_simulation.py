#!/usr/bin/env python3
"""
Script to launch the ROS2 sailboat simulation and configure the environment.

Adjust the constants below to set:
- Boat starting position
- Buoy locations
- Wind conditions
- Route waypoints
"""

import subprocess
import time
import signal
import sys
import os
import threading

# =============================================================================
# CONFIGURATION - Adjust these values as needed
# =============================================================================

# Boat starting position (meters)
BOAT_EAST = -10.0
BOAT_NORTH = 60.0

# Buoy definitions as list of dicts with location and optional color
# Each buoy needs: east, north (meters)
# Optional: r, g, b (color components 0.0-1.0, defaults to orange if omitted)
BUOYS = [
    {"east": 0.0, "north": 0.0},                              # Default orange
    {"east": 0.0, "north": 50.0, "r": 1.0, "g": 0.0, "b": 0.0},  # Red
    {"east": 50.0, "north": 0.0, "r": 0.0, "g": 1.0, "b": 0.0},  # Green
    {"east": 50.0, "north": 50.0, "r": 0.0, "g": 0.0, "b": 1.0}, # Blue
]

# Wind settings
WIND_ANGLE = 0.1   # degrees [0, 360), 0 = from North, 90 = from East
WIND_SPEED = 4.0    # m/s [0, 25]

# Route waypoints as list of (east, north) tuples (meters)
WAYPOINTS = [
    (-10.0, -10.0),
    (60.0, -10.0),
    (60.0, 60.0),
    (-10.0, 60.0)
]

# Delay settings (seconds)
STARTUP_DELAY = 5.0  # Time to wait for ROS system to initialize
MESSAGE_DELAY = 0.5  # Time between sending messages
SHUTDOWN_DELAY = 5.0  # Time to wait after route completion before shutting down

# =============================================================================
# END CONFIGURATION
# =============================================================================


def build_env_create_message() -> str:
    """Build the EnvironmentCreation message as a YAML string."""
    buoy_strings = []
    for buoy in BUOYS:
        r = buoy.get("r", 1.0)
        g = buoy.get("g", 0.5)
        b = buoy.get("b", 0.0)
        buoy_strings.append(
            f"{{east: {buoy['east']}, north: {buoy['north']}, color: {{r: {r}, g: {g}, b: {b}}}}}"
        )
    buoy_list = ", ".join(buoy_strings)
    return f"{{boat_east: {BOAT_EAST}, boat_north: {BOAT_NORTH}, buoys: [{buoy_list}], wind_direction: {WIND_ANGLE}, wind_speed: {WIND_SPEED}}}"


def build_wind_message() -> str:
    """Build the Wind message as a YAML string."""
    return f"{{angle: {WIND_ANGLE}, speed: {WIND_SPEED}}}"


def build_route_message() -> str:
    """Build the Route message as a YAML string."""
    waypoint_list = ", ".join(
        f"{{east: {e}, north: {n}}}" for e, n in WAYPOINTS
    )
    return f"{{waypoints: [{waypoint_list}]}}"


def publish_message(topic: str, msg_type: str, message: str) -> None:
    """Publish a single message to a ROS2 topic."""
    cmd = ["ros2", "topic", "pub", "--once", topic, msg_type, message]
    print(f"Publishing to {topic}: {message}")
    subprocess.run(cmd, check=True)


def main():
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
    print("Sailboat Simulation Launcher")
    print("=" * 60)
    print(f"Boat position: ({BOAT_EAST}, {BOAT_NORTH})")
    print(f"Buoys: {BUOYS}")
    print(f"Wind: {WIND_ANGLE} deg @ {WIND_SPEED} m/s")
    print(f"Waypoints: {WAYPOINTS}")
    print("=" * 60)

    # 1. Launch the ROS system
    print("\n[1/4] Launching ROS system...")
    ros_process = subprocess.Popen(
        ["ros2", "launch", "launch/launch.py"],
        start_new_session=True,
    )

    print(f"Waiting {STARTUP_DELAY}s for system to initialize...")
    time.sleep(STARTUP_DELAY)

    # 2. Send environment creation message
    print("\n[2/4] Initializing environment...")
    publish_message(
        "/env/create",
        "boat_msgs/msg/EnvironmentCreation",
        build_env_create_message()
    )
    time.sleep(MESSAGE_DELAY)

    # 3. Send wind message
    print("\n[3/4] Setting wind conditions...")
    publish_message(
        "/env/wind",
        "boat_msgs/msg/Wind",
        build_wind_message()
    )
    time.sleep(MESSAGE_DELAY)

    # 4. Send route message
    print("\n[4/4] Setting route waypoints...")
    publish_message(
        "/planning/target_route",
        "boat_msgs/msg/Route",
        build_route_message()
    )

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
