# Boat state information published by the simulation.
# All angles are in DEGREES.

float32 x                    # Position x in meters (world frame)
float32 y                    # Position y in meters (world frame)

float32 velocity_x           # Velocity x in m/s (world frame)
float32 velocity_y           # Velocity y in m/s (world frame)
float32 velocity_magnitude   # Speed in m/s [0, inf)

float32 yaw_rate             # Yaw rate in rad/s, positive = turning left (CCW from above)

float32 heading              # Heading in degrees [0, 360), 0 = North, 90 = East (CW from above)
float32 rudder_angle         # Current rudder angle in degrees [-45, 45], positive = turn right
float32 sail_angle           # Current sail angle in degrees [0, 90], 0 = centered, 90 = fully out