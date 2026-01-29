# ROS Workshop - Autonomous Sailboat Navigation

A complete ROS2-based autonomous sailboat navigation system with physics-based simulation, PID control, and intelligent route planning.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Packages](#packages)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Full System](#running-the-full-system)
  - [Interactive Simulation](#interactive-simulation)
  - [Launch Individual Nodes](#launch-individual-nodes)
- [Docker Deployment](#docker-deployment)
- [Key Features](#key-features)
- [Unit Conventions](#unit-conventions)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Structure](#code-structure)
  - [Adding New Features](#adding-new-features)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements an end-to-end autonomous sailing system that combines:
- **High-fidelity physics simulation** using MuJoCo
- **PID-based heading control** with stall recovery
- **Intelligent route planning** with tacking and jibing maneuvers
- **Real-time visualization** of boat state, wind conditions, and planned routes

The system is designed for the Sailing Team Darmstadt (STDA) and serves as both a research platform and educational tool for autonomous marine vehicle control.

## System Architecture

```
┌─────────────────────┐
│  route_planner      │  Plans routes considering wind constraints
│                     │  Handles tacking/jibing maneuvers
└──────────┬──────────┘
           │ /planning/desired_heading
           ↓
┌─────────────────────┐
│  controller         │  PID control for rudder
│                     │  Wind-relative sail trim
└──────────┬──────────┘
           │ /control/rudder, /control/sail
           ↓
┌─────────────────────┐
│  simulation         │  MuJoCo physics simulation
│                     │  Renders video or live viewer
└──────────┬──────────┘
           │ /sense/boat_info, /sense/wind
           └─────────────────────────────┘
```

## Packages

### [boat_msgs](src/boat_msgs)
Custom ROS2 message definitions for sailboat navigation.
- BoatInfo, Wind, Heading, Route, Location
- RudderAngle, SailAngle control messages
- Environment setup messages

### [simulation](src/simulation)
Physics-based sailboat simulation using MuJoCo.
- Realistic sail aerodynamics with lift/drag model
- Boom dynamics with sheet limits
- Rudder hydrodynamics and water resistance
- Video recording and live rendering
- Dynamic environment creation

### [controller](src/controller)
PID controller for rudder and lookup-based sail controller.
- Heading tracking with dynamic damping
- Stall recovery (escape from "in irons")
- Wind-relative sail trim
- Performance monitoring and plotting

### [route_planner](src/route_planner)
Autonomous route planning with sailing constraints.
- Waypoint navigation with tacking/jibing
- No-sail zone handling
- Dynamic replanning on wind changes
- Off-course detection and correction

## Quick Start

### Prerequisites

- ROS2 Jazzy (or compatible distribution)
- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
cd ~/
git clone <repository-url> ros_workshop
cd ros_workshop
```

2. Install dependencies:
```bash
# Install Python packages
pip install -r requirements.txt

# Install ROS dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the workspace:
```bash
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

### Running the Full System

The easiest way to run the complete autonomous sailing system:

```bash
python scripts/run_simulation.py
```

This script will:
1. Launch all three ROS2 nodes (simulation, controller, route_planner)
2. Create an environment with buoys
3. Send a route to navigate
4. Record a video of the mission
5. Automatically terminate when the route is completed

**Configuration:** Edit `scripts/run_simulation.py` to customize:
- Boat starting position
- Buoy locations and colors
- Wind direction and speed
- Route waypoints
- Video output settings

### Interactive Simulation

To manually control the sailboat with UI sliders:

```bash
python scripts/play.py
```

Use the sliders in the MuJoCo viewer to control rudder and sail angles directly.

### Launch Individual Nodes

You can also launch nodes separately for development and testing:

```bash
# Terminal 1 - Simulation
ros2 launch simulation simulation.launch.py

# Terminal 2 - Controller
ros2 launch controller controller.launch.py

# Terminal 3 - Route Planner
ros2 launch route_planner route_planner.launch.py
```

Then publish commands manually:

```bash
# Create environment
ros2 topic pub /env/create boat_msgs/msg/EnvironmentCreation \
  "{buoys: [{east: 0, north: 0, color: {r: 1, g: 0.5, b: 0}}], \
    wind_direction: 45.0, wind_speed: 5.0, \
    boat_east: -10.0, boat_north: -10.0}" -1

# Send target route
ros2 topic pub /planning/target_route boat_msgs/msg/Route \
  "{waypoints: [{east: 10.0, north: 10.0}, {east: 20.0, north: 20.0}]}"
```

## Docker Deployment

Pre-built multi-platform Docker images are available for both `amd64` and `arm64` architectures through GitHub Container Registry.

### Using Pre-built Images

Pull and run the latest image:

```bash
# Pull the latest image
docker pull ghcr.io/cptk/sailboat-simulator:main

# Run container
docker run -it --rm ghcr.io/cptk/sailboat-simulator:main

# Inside container, run the simulation
python scripts/run_simulation.py
```

Available tags:
- `main` - Latest build from main branch
- `v*` - Specific version releases (e.g., `v1.0.0`)
- `sha-<commit>` - Specific commit builds

### Building Locally

Alternatively, build the image from source:

```bash
# Build image
docker build -t ros_workshop .

# Run container
docker run -it --rm ros_workshop

# Inside container, run the simulation
python scripts/run_simulation.py
```

The Docker image includes:
- ROS2 Jazzy base
- All Python dependencies
- Pre-built workspace
- Headless rendering (MUJOCO_GL=osmesa)
- Support for both amd64 and arm64 architectures

## Key Features

### Physics Simulation
- **Sail aerodynamics**: Lift/drag model with angle of attack
- **No-go zone**: Realistic behavior when pointing into wind
- **Boom dynamics**: Physical sheet limits and wind-driven swing
- **Stall recovery**: Backing maneuvers when stuck in irons
- **Water resistance**: Quadratic drag with high lateral resistance from keel

### Route Planning
- **Wind-aware**: Accounts for no-sail zones and tacking requirements
- **Automatic maneuvers**: Tacking and jibing with safety checks
- **Dynamic replanning**: Adapts to wind changes and off-course drift
- **Soft waypoints**: Intermediate tacking points for upwind navigation

### Control
- **PID rudder control**: Proportional-Integral-Derivative with anti-windup
- **Dynamic damping**: Adaptive gains for stability near target
- **Stall recovery**: Hard rudder when stuck pointing into wind
- **Sail trim**: Automatic adjustment based on apparent wind angle

### Visualization
- **Live viewer**: Interactive 3D visualization with camera following boat
- **Video recording**: Automated MP4 export of missions
- **Route display**: Visual feedback of planned waypoints
- **Heading indicator**: Shows desired vs. actual heading

## Unit Conventions

Throughout the system:
- **Angles in ROS messages**: DEGREES
- **Angles in internal calculations**: RADIANS
- **Heading convention**: Nautical (0°=North, 90°=East, clockwise positive)
- **Wind direction**: Direction wind is coming FROM
- **Positions**: Meters in East-North coordinate frame
- **Velocities**: m/s

## Development

### Running Tests

```bash
# Test all packages
colcon test

# Test specific package
colcon test --packages-select route_planner

# Run pytest directly
pytest src/route_planner/test/
pytest src/simulation/test/
```

### Code Structure

```
ros_workshop/
├── src/
│   ├── boat_msgs/           # Message definitions
│   ├── simulation/          # MuJoCo simulation
│   │   ├── simulation/
│   │   │   ├── node.py              # ROS2 node
│   │   │   ├── sailboat_simulation.py  # High-level interface
│   │   │   ├── sailboat_physics.py     # Physics models
│   │   │   └── build_env.py            # Dynamic environment
│   │   └── test/
│   ├── controller/          # PID controller
│   │   ├── controller/
│   │   │   ├── node.py          # ROS2 node
│   │   │   ├── pid.py           # PID implementation
│   │   │   ├── sail_controller.py  # Sail trim
│   │   │   └── utils.py
│   │   └── test/
│   └── route_planner/       # Mission planning
│       ├── route_planner/
│       │   ├── node.py          # ROS2 node
│       │   ├── planner.py       # Mission planner
│       │   ├── waypoint.py      # Waypoint representation
│       │   ├── segment_action.py   # Straight/tacking segments
│       │   ├── maneuver_action.py  # Tack/jibe maneuvers
│       │   └── utils.py
│       └── test/
├── scripts/
│   ├── run_simulation.py    # Full system launcher
│   └── play.py              # Interactive manual control
├── launch/                  # Launch files
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
└── README.md               # This file
```

### Adding New Features

Each package contains detailed READMEs:
- [boat_msgs/README.md](src/boat_msgs/README.md) - Message definitions
- [simulation/README.md](src/simulation/README.md) - Physics and rendering
- [controller/README.md](src/controller/README.md) - Control algorithms
- [route_planner/README.md](src/route_planner/README.md) - Planning logic

## Performance Tips

- **Video recording**: Use `render_mode:=none` for faster-than-realtime simulation
- **Route planning**: Adjust `step_interval` parameter to balance responsiveness vs. CPU usage
- **Controller tuning**: Modify PID gains in `controller/node.py` for different boat characteristics

## Troubleshooting

### Video Recording Issues
If video output is missing frames or corrupted:
```bash
pip install --upgrade imageio imageio-ffmpeg
```

### MuJoCo Rendering Issues
For headless environments:
```bash
export MUJOCO_GL=osmesa
```

For GPU rendering:
```bash
export MUJOCO_GL=egl  # or 'glfw' for desktop
```

### ROS2 Build Errors
Clear build artifacts and rebuild:
```bash
rm -rf build/ install/ log/
colcon build --symlink-install
```
