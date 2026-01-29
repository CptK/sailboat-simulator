# boat_msgs

Custom ROS2 message package for autonomous sailboat navigation.

## Overview

This package contains the message definitions required for autonomous sailboat control and simulation. All message definitions use standard units and conventions.

## Message Definitions

### BoatInfo.msg
Complete boat state information from the simulation.
- **Position**: `x`, `y` (meters, East-North coordinate frame)
- **Velocity**: `velocity_x`, `velocity_y`, `velocity_magnitude` (m/s)
- **Heading**: `heading` (degrees, 0-360, nautical convention: 0=North, 90=East)
- **Control surfaces**: `rudder_angle` (degrees, -45 to 45), `sail_angle` (degrees, 0-90)
- **Dynamics**: `yaw_rate` (rad/s)

### Wind.msg
Wind conditions.
- **angle**: Wind direction (degrees, 0-360, direction wind is coming FROM)
- **speed**: Wind speed (m/s)

### Heading.msg
Target or current heading.
- **heading**: Heading angle (degrees, 0-360, nautical convention)

### RudderAngle.msg
Rudder control command.
- **angle**: Rudder angle (degrees, -45 to 45, positive = starboard)

### SailAngle.msg
Sail control command.
- **sail_angle**: Maximum boom swing angle (degrees, 0-90, 0=tight, 90=fully out)

### Location.msg
Geographic position in local coordinate frame.
- **east**: East coordinate (meters)
- **north**: North coordinate (meters)

### Route.msg
Planned route as sequence of waypoints.
- **waypoints**: Array of `Location` messages

### Buoy.msg
Navigation mark or obstacle.
- **east**, **north**: Position (meters)
- **color**: RGB color for visualization

### Color.msg
RGB color specification.
- **r**, **g**, **b**: Red, green, blue (0.0-1.0)

### EnvironmentCreation.msg
Message for initializing simulation environment.
- **buoys**: Array of `Buoy` messages
- **wind_direction**, **wind_speed**: Initial wind conditions
- **boat_east**, **boat_north**: Initial boat position

## Dependencies

- ROS2 (ament_cmake build system)
- std_msgs
- builtin_interfaces

## Build

This package uses the standard ROS2 build process:

```bash
colcon build --packages-select boat_msgs
```
