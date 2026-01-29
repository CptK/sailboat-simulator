from pydantic import BaseModel, field_validator
import numpy as np


class ControllerInfo(BaseModel):
    """
    Internal state for the controller. All angles are in RADIANS.

    This is populated from ROS messages (which use degrees) by the ControllerNode,
    which performs the degree-to-radian conversion.

    Attributes:
        boat_x, boat_y: Position in meters (world frame)
        boat_heading: Heading in radians [0, 2π), 0 = North, π/2 = East
        boat_vel_x, boat_vel_y: Velocity in m/s (world frame)
        boat_speed: Speed magnitude in m/s
        boat_yaw_rate: Yaw rate in rad/s
        wind_angle: Wind direction in radians [0, 2π), direction wind comes FROM
        wind_speed: Wind speed in m/s
        desired_heading: Target heading in radians [0, 2π)
    """

    boat_x: float | None = None                # meters
    boat_y: float | None = None                # meters
    boat_heading: float | None = None          # radians [0, 2π)
    boat_vel_x: float | None = None            # m/s
    boat_vel_y: float | None = None            # m/s
    boat_speed: float | None = None            # m/s
    boat_yaw_rate: float | None = None         # rad/s
    wind_angle: float | None = None            # radians [0, 2π)
    wind_speed: float | None = None            # m/s
    desired_heading: float | None = None       # radians [0, 2π)


    def is_complete(self) -> bool:
        for value in self.model_dump().values():
            if value is None:
                return False
        return True
    
    def reset(self):
        for key in self.model_dump().keys():
            setattr(self, key, None)

    @field_validator('boat_heading', 'wind_angle', 'desired_heading')
    def validate_angle(cls, v):
        if v is not None and not (0.0 <= v <= 2 * np.pi):
            raise ValueError('Angle must be within [0, 2*pi) radians')
        return v

    # validate that wind speed is non-negative
    @field_validator('wind_speed')
    def validate_wind_speed(cls, v):
        if v is not None and v < 0.0:
            raise ValueError('wind_speed must be non-negative')
        return v
    

def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] radians.

    Args:
        angle: Angle in radians (any value)

    Returns:
        Equivalent angle in radians within [-π, π]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle