"""This module contains a lookup-based sail controller."""

import numpy as np


class SailController:
    """
    Simple sail controller based on apparent wind angle.

    Unit conventions:
        Input: All angles in RADIANS
        Output: sail_angle in DEGREES [0, 90]
    """

    def compute_sail_angle(self, wind_angle: float, boat_heading: float, boat_speed: float | None = None) -> float:
        """
        Compute sail angle command based on apparent wind angle.

        Args:
            wind_angle: Wind direction in RADIANS (global frame).
            boat_heading: Boat heading in RADIANS (global frame).
            boat_speed: Boat speed in m/s (optional, for stall recovery).

        Returns:
            sail_angle_deg: Sail angle command in degrees [0, 90].
        """
        # Apparent wind angle relative to boat heading
        apparent_wind_angle = wind_angle - boat_heading
        apparent_wind_angle = (apparent_wind_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        apparent_wind_deg = abs(np.degrees(apparent_wind_angle))

        # Stall recovery: if stuck in irons, open sail to catch wind when bow falls off
        if boat_speed is not None and boat_speed < 0.5 and apparent_wind_deg < 50:
            return 60.0

        # Sail angle based on apparent wind angle
        # Close-hauled (< 50°): tight
        # Close reach (50-80°): moderate
        # Beam reach (80-120°): eased
        # Broad reach (120-150°): loose
        # Running (> 150°): fully eased
        if apparent_wind_deg < 50:
            sail_angle_deg = 20.0
        elif apparent_wind_deg < 80:
            sail_angle_deg = 35.0
        elif apparent_wind_deg < 120:
            sail_angle_deg = 50.0
        elif apparent_wind_deg < 150:
            sail_angle_deg = 70.0
        else:
            sail_angle_deg = 85.0

        return sail_angle_deg