# controller

Rudder and sail controller for autonomous sailboat navigation.

## Overview

This package implements feedback controllers that command the rudder and sail to follow a desired heading. Two rudder controller options are available:
- **PID**: Traditional PID control with dynamic damping and stall recovery
- **LQR**: Linear Quadratic Regulator for optimal state feedback control

Both controllers work with a lookup-based sail controller that adjusts sail trim based on apparent wind angle.

## Features

### Rudder Control
- **PID control**: Proportional-Integral-Derivative control with dynamic damping
- **LQR control**: Linear Quadratic Regulator for optimal state feedback
- **Stall recovery**: Automatic escape from "in irons" condition (PID)
- **Performance logging**: Saves plots of rudder actions and heading tracking

### Sail Control
- **Wind-relative sail trim**: Sail angle automatically adjusted for apparent wind
- **Point-of-sail optimization**: Different trim for close-hauled, reaching, and running

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

### LQRController (lqr.py)

Computes rudder angle command using Linear Quadratic Regulator (LQR) optimal control.

**Features:**
- State feedback on heading and yaw rate
- Optimal control minimizing quadratic cost function
- Automatic feedforward for reference tracking
- Handles angle wrapping for shortest-path turns
- Dynamics extracted directly from MuJoCo simulation

**Design:**
The LQR controller is designed based on the linearized heading dynamics extracted from the sailboat simulation using finite differences. The system matrices A and B model the relationship between rudder input and heading/yaw rate response.

**Cost Function:**
```
J = ∫ (x'Qx + u'Ru) dt
```
Where:
- Q: State cost matrix (penalizes heading error and yaw rate)
- R: Control cost matrix (penalizes rudder effort)

**Default Cost Matrices:**
- Q = diag([0.5, 0.5]) - Equal weight on heading and yaw rate
- R = [1.0] - Moderate rudder effort penalty

**Usage:**
```python
from controller.lqr import get_lqr_controller
from simulation.sailboat_simulation import SailboatSimulation

sim = SailboatSimulation()
controller = get_lqr_controller(sim, Q=custom_Q, R=custom_R)
rudder_deg = controller.compute_action(info)
```

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

### ROS2 Testing

The controller can be tested with ROS2 nodes:

```bash
ros2 launch simulation simulation.launch.py &
ros2 launch controller controller.launch.py &
ros2 topic pub /planning/desired_heading boat_msgs/msg/Heading "{heading: 90.0}" -1
```

### Standalone Controller Testing

For standalone controller testing and comparison without ROS2:

```bash
# Test LQR controller
python scripts/test_controller.py --controller lqr

# Test PID controller
python scripts/test_controller.py --controller pid
```

This script:
- Cycles through multiple target headings automatically
- Uses automatic sail control via SailController
- Provides real-time feedback on tracking performance
- Allows easy comparison between PID and LQR controllers

Configure test parameters by editing constants in `scripts/test_controller.py`:
- `DESIRED_HEADINGS`: Sequence of headings to test
- `SECONDS_PER_HEADING`: Duration for each heading
- `LQR_Q`, `LQR_R`: LQR cost matrices
- `PID_KP`, `PID_KI`, `PID_KD`: PID gains
