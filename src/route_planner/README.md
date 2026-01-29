# route_planner

Autonomous route planning for sailboat navigation with tacking and jibing.

## Overview

This package implements a mission planner that generates and executes sailing routes to navigate through waypoints while respecting wind conditions and sailing constraints. The planner handles tacking (turning through the wind) and jibing (turning away from the wind) maneuvers, accounts for the no-sail zone, and dynamically replans routes when conditions change.

## Features

- **Waypoint navigation**: Plans routes through multiple waypoints
- **Wind-aware planning**: Accounts for no-sail zone and tacking requirements
- **Automatic maneuvers**: Executes tacks and jibes to reach target headings
- **Dynamic replanning**: Adapts route when wind changes or boat goes off-course
- **Safety logic**: Avoids dangerous jibes in strong winds
- **Soft waypoints**: Generates intermediate tacking waypoints for upwind navigation
- **Mission completion detection**: Monitors progress and signals when goals are reached

## Node: route_planner

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/planning/desired_heading` | `Heading` | Target heading for controller |
| `/planning/current_route` | `Route` | Current planned route for visualization |
| `/planning/route_completed` | `Bool` | Signals when mission is complete |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/sense/boat_info` | `BoatInfo` | Current boat position and heading |
| `/sense/wind` | `Wind` | Current wind conditions |
| `/planning/target_route` | `Route` | Target waypoints from mission supervisor |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_interval` | float | - | Time between planning steps (seconds) |
| `dist_threshold` | float | - | Distance to consider waypoint reached (meters) |
| `off_course_threshold` | float | - | Distance from route to trigger replanning (meters) |
| `min_tack_angle` | float | - | Minimum angle to wind for direct sailing (degrees) |
| `safe_jibe_threshold` | float | - | Wind speed above which jibing is avoided (m/s) |
| `maneuver_heading_step_size` | float | - | Heading change per step during maneuvers (degrees) |
| `command_step_size_multiplier` | float | - | Multiplier for command step size |
| `heading_tolerance` | float | - | Acceptable heading error (degrees) |
| `max_steps_to_cross_wind_when_tacking` | int | - | Max steps to complete tack before switching to jibe |

Parameters are typically loaded from `resource/parameters.yaml`.

## Planning Architecture

### Mission (planner.py)

Top-level mission planner that:
- Splits route into segments between hard waypoints
- Determines if segments require tacking or can be sailed directly
- Generates intermediate "soft" waypoints for tacking maneuvers
- Monitors progress and triggers replanning when needed

**Replanning Triggers:**
1. Waypoint reached
2. Wind direction changes critically
3. Boat drifts off-course beyond threshold

### WayAction Types (segment_action.py)

**StraightWayAction**: Direct path when wind allows sailing to target.

**TackingWayAction**: Zigzag path with soft waypoints when target is upwind. Generates waypoints at ±`min_tack_angle` to the wind.

### Maneuver Types (maneuver_action.py)

**TackManeuver**: Turn through the wind (bow through wind).
- Used when both current and target headings are upwind
- Executed in steps to allow controller to track smoothly
- Can detect if stuck (not crossing wind zone) and abort

**JibeManeuver**: Turn away from wind (stern through wind).
- Used when running or broad reach
- Avoided in high winds (> `safe_jibe_threshold`)
- Safer but requires more sea room than tacking

### Utilities (utils.py)

Helper functions for:
- Bearing calculations between waypoints
- Angular distance (shortest path between two angles)
- Point-to-line distance for off-course detection
- Sailability checks (is target within no-sail zone?)

### Waypoint (waypoint.py)

Represents positions in the local navigation frame:
- **Hard waypoints**: Mission goals that must be reached
- **Soft waypoints**: Intermediate tacking points that can be skipped if hard waypoint becomes reachable

## Navigation Logic

The planner operates in a state machine with three modes:

1. **Normal Navigation**: Follows planned route, publishes desired heading
2. **Maneuvering**: Executes tack or jibe when heading difference > 70°
3. **Waypoint Reached**: Moves to next waypoint, replans route

**Heading Threshold:** If heading error exceeds 70°, a maneuver is initiated. Otherwise, the controller handles minor course corrections.

## Unit Conventions

- All angles in messages are in **DEGREES**
- Headings follow **nautical convention** (0°=North, 90°=East)
- Positions in **meters** (East-North frame)
- Wind angle is direction wind is coming FROM

## Usage

### Launch Planner

```bash
ros2 launch route_planner route_planner.launch.py
```

### Send Target Route

```bash
ros2 topic pub /planning/target_route boat_msgs/msg/Route \
  "{waypoints: [{east: 10.0, north: 10.0}, {east: 20.0, north: 20.0}]}"
```

### Monitor Progress

```bash
ros2 topic echo /planning/current_route
ros2 topic echo /planning/route_completed
```

## Testing

Run unit tests:

```bash
pytest src/route_planner/test/
```

Tests cover:
- Waypoint distance calculations
- Bearing and angular distance functions
- Maneuver logic (tacking and jibing)
- Segment action planning
- Mission planning with various wind conditions

## Dependencies

- ROS2
- Pydantic (for configuration validation)
- boat_msgs
