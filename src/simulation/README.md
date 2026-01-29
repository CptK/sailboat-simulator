# simulation

Physics-based sailboat simulation using MuJoCo.

## Overview

This package provides a high-fidelity sailboat simulation with realistic sail aerodynamics, rudder hydrodynamics, and water resistance. The simulation is built on MuJoCo and publishes boat state to ROS2 topics while subscribing to control commands.

## Features

- **Realistic sail physics**: Lift/drag model with angle of attack, no-go zone, and stall recovery
- **Boom dynamics**: Physical sheet limits and wind-driven boom swing
- **Rudder control**: Hydrodynamic lift proportional to boat speed
- **Water resistance**: Quadratic drag with high lateral resistance from keel
- **Rendering modes**:
  - `none`: No visualization (fastest)
  - `viewer`: Live interactive window
  - `video`: Record simulation to MP4 file
- **Environment creation**: Dynamic buoy placement and wind configuration
- **Route visualization**: Visual feedback of planned routes and desired heading

## Node: simulation_node

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/sense/boat_info` | `BoatInfo` | Complete boat state (position, velocity, heading, control surfaces) |
| `/sense/wind` | `Wind` | Current wind conditions (direction and speed) |
| `/sense/heading` | `Heading` | Current boat heading |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/control/rudder` | `RudderAngle` | Rudder command (-45 to 45 degrees) |
| `/control/sail` | `SailAngle` | Sail angle command (0 to 90 degrees) |
| `/env/wind` | `Wind` | External wind override |
| `/env/reset_simulation` | `Bool` | Reset simulation to initial state |
| `/env/create` | `EnvironmentCreation` | Create new environment with buoys |
| `/planning/desired_heading` | `Heading` | Desired heading for visualization |
| `/planning/current_route` | `Route` | Current planned route for visualization |
| `/planning/route_completed` | `Bool` | Clear route visualization when completed |

### Parameters

- **render_mode** (string, default: "video"): Rendering mode ("none", "viewer", or "video")
- **video_output** (string, default: "simulation.mp4"): Output path for video recording
- **video_fps** (int, default: 30): Frames per second for video
- **video_width** (int, default: 1280): Video width in pixels
- **video_height** (int, default: 720): Video height in pixels

## Physics Module

### sailboat_physics.py

Contains the core physics models:

- **Sail aerodynamics**: Cambered sail lift/drag with angle of attack calculations
- **No-go zone**: Realistic luffing behavior when pointing too close to wind
- **Boom dynamics**: Wind-driven boom swing with sheet limits
- **Stall recovery**: Backing behavior when stuck in irons
- **Rudder forces**: Hydrodynamic lift proportional to V²
- **Water drag**: Separate forward and lateral resistance coefficients
- **Weather helm**: Torque from center of effort vs center of lateral resistance

### sailboat_simulation.py

High-level simulation interface managing:
- MuJoCo model and data
- Physics stepping
- Control input application
- State extraction
- Visualization elements (route, heading indicator)

### build_env.py

Dynamic environment construction:
- XML generation for buoys and navigation marks
- Model recompilation with new obstacles
- Camera setup and lighting

## Unit Conventions

- All ROS messages use **DEGREES** for angles
- Internal physics calculations use **RADIANS**
- Positions in **meters** (East-North coordinate frame)
- Velocities in **m/s**
- Heading follows **nautical convention** (0°=North, 90°=East, clockwise positive)

## Usage

### Launch Simulation

```bash
ros2 launch simulation simulation.launch.py
```

### Launch with Parameters

```bash
ros2 launch simulation simulation.launch.py render_mode:=viewer
```

## Dependencies

- ROS2
- MuJoCo (`mujoco` Python package)
- NumPy
- imageio (for video recording)
- boat_msgs

## Testing

Run physics unit tests:

```bash
pytest src/simulation/test/
```
