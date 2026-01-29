# controller

Rudder and sail controller for autonomous sailboat navigation.

## Overview

This package implements a feedback controller that commands the rudder and sail to follow a desired heading. The controller uses a PID algorithm for rudder control with dynamic damping and stall recovery, plus a lookup-based sail controller that adjusts sail trim based on apparent wind angle.

## Features

- **PID rudder control**: Proportional-Integral-Derivative control for heading tracking
- **Dynamic damping**: Adaptive derivative gain that reduces overshoot
- **Stall recovery**: Automatic escape from "in irons" condition
- **Wind-relative sail trim**: Sail angle automatically adjusted for apparent wind
- **Performance logging**: Saves plots of rudder actions and heading tracking

## Node: controller_node

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/control/rudder` | `RudderAngle` | Rudder command in degrees (-45 to 45) |
| `/control/sail` | `SailAngle` | Sail angle command in degrees (0 to 90) |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/sense/boat_info` | `BoatInfo` | Boat state from simulation |
| `/sense/wind` | `Wind` | Wind conditions from simulation |
| `/planning/desired_heading` | `Heading` | Target heading from route planner |

### Control Loop

Runs at **100 Hz** for responsive control.

## Controller Components

### PIDController (pid.py)

Computes rudder angle command using PID control on heading error.

**Features:**
- Proportional gain: Turns harder when further from target
- Integral gain: Corrects for steady-state bias (often set to 0 for boats)
- Derivative gain: Damping based on yaw rate to prevent overshoot
- Dynamic D-gain: Higher damping when close to target for stability
- Anti-windup: Integral clamping to prevent overshoot

**Stall Recovery:**
When boat speed drops below 0.5 m/s and wind angle is < 50° (pointing into wind), the controller applies hard rudder (±45°) to escape "in irons" condition. Turn direction is chosen as the shorter path to the desired heading.

**Default Gains:**
- Kp = 0.8
- Ki = 0.0
- Kd = 1.0

### SailController (sail_controller.py)

Lookup-based sail trim controller.

| Apparent Wind Angle | Sail Angle | Point of Sail |
|---------------------|------------|---------------|
| < 50° | 20° | Close-hauled |
| 50-80° | 35° | Close reach |
| 80-120° | 50° | Beam reach |
| 120-150° | 70° | Broad reach |
| > 150° | 85° | Running |

**Stall Recovery:** When speed < 0.5 m/s and pointing into wind, sail is opened to 60° to catch wind when bow falls off.

## Unit Conventions

- ROS messages use **DEGREES** for angles
- Internal `ControllerInfo` uses **RADIANS** for angles and rad/s for yaw rate
- PID output is converted to **DEGREES** for publishing

## Performance Monitoring

On shutdown, the controller saves `controller_performance.png` with two plots:
1. Rudder angle commands over time
2. Actual vs. desired heading over time

## Usage

### Launch Controller

```bash
ros2 launch controller controller.launch.py
```

### Tuning PID Gains

Edit `src/controller/controller/node.py:47` to adjust gains:

```python
self.rudder_controller = PIDController(kp=0.8, ki=0.0, kd=1.0)
```

## Dependencies

- ROS2
- NumPy
- Matplotlib (for performance plots)
- boat_msgs

## Testing

The controller can be tested in simulation:

```bash
ros2 launch simulation simulation.launch.py &
ros2 launch controller controller.launch.py &
ros2 topic pub /planning/desired_heading boat_msgs/msg/Heading "{heading: 90.0}" -1
```
