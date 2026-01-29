"""
Sailboat Physics Module

Contains the physical models for sailboat simulation:
- Sail aerodynamics (lift/drag)
- Rudder hydrodynamics
- Water drag and keel resistance
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PhysicsParams:
    """Physical parameters for the sailboat simulation."""
    # Air properties
    air_density: float = 1.225  # kg/m³

    # Sail properties
    sail_area: float = 10.0  # m²
    lift_coefficient: float = 1.2
    drag_coefficient: float = 0.1

    # Rudder properties
    water_density: float = 1000.0  # kg/m³
    rudder_area: float = 0.2  # m²
    rudder_cl: float = 1.0
    rudder_arm: float = 1.1  # m, distance from rudder to center of rotation

    # Hull drag coefficients
    drag_forward: float = 10.0  # Hull drag going forward
    drag_lateral: float = 2000.0  # Keel prevents sideways motion (high ratio needed)

    # Yaw damping
    yaw_damping: float = 150.0

    # Center of effort/resistance positions
    sail_ce_aft: float = -0.3  # Sail CE position, aft of boat center
    sail_ce_height: float = 2.0  # Sail CE height above waterline
    keel_clr_aft: float = -0.2  # Center of lateral resistance

    # Sailing limits
    no_go_zone: float = np.radians(140)  # Can't sail within 40° of head-to-wind


@dataclass
class BoatState:
    """Current state of the boat."""
    heading: float  # radians
    velocity: np.ndarray  # [vx, vy, vz] in world frame
    boom_angle: float  # radians
    rudder_angle: float  # degrees
    yaw_rate: float  # rad/s


@dataclass
class WindState:
    """Wind conditions."""
    direction: float  # Wind coming FROM this direction (radians)
    speed: float  # m/s


@dataclass
class SailControl:
    """Sail control settings."""
    sail_angle: float  # 0-90 degrees, max angle boom can swing out


@dataclass
class Forces:
    """Forces and torques to apply to the boat."""
    fx: float = 0.0  # World X force
    fy: float = 0.0  # World Y force
    yaw: float = 0.0  # Yaw torque
    roll: float = 0.0  # Roll torque (heeling)
    boom: float = 0.0  # Boom torque


def get_wind_vector(wind: WindState) -> np.ndarray:
    """Get wind velocity vector (direction wind is blowing TO, not FROM)."""
    return np.array([
        -wind.speed * np.cos(wind.direction),
        -wind.speed * np.sin(wind.direction),
        0.0
    ])


def get_boat_speed(boat: BoatState) -> float:
    """Get boat speed (magnitude of horizontal velocity)."""
    return np.linalg.norm(boat.velocity[:2])


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_sail_forces(
    boat: BoatState,
    wind: WindState,
    sail: SailControl,
    params: PhysicsParams
) -> Forces:
    """
    Compute forces from sail aerodynamics.

    Uses a lift/drag model where:
    - Lift is perpendicular to apparent wind
    - Drag is parallel to apparent wind
    - Coefficients vary with angle of attack
    """
    forces = Forces()

    wind_vec = get_wind_vector(wind)

    # Apparent wind (wind relative to moving boat)
    apparent_wind = wind_vec - boat.velocity
    apparent_speed = np.linalg.norm(apparent_wind[:2])

    if apparent_speed < 0.1:
        return forces

    # Apparent wind angle in world frame (direction wind is blowing TO)
    apparent_wind_angle = np.arctan2(apparent_wind[1], apparent_wind[0])

    # Wind angle relative to boat (0 = from stern, pi = from bow)
    wind_to_boat_angle = normalize_angle(apparent_wind_angle - boat.heading)
    abs_wind_angle = abs(wind_to_boat_angle)

    # Sail angle directly specifies max boom swing (0=tight, 90=fully out)
    max_sail_angle = np.radians(sail.sail_angle)

    # === BOOM PHYSICS ===
    # The sheet is a LIMIT, not a spring. The wind pushes the boom freely
    # until the sheet goes taut (boom reaches max_sail_angle).

    # Wind always pushes boom to leeward (away from wind source)
    # wind_to_boat_angle > 0 means wind blows toward port, so boom goes to port
    if wind_to_boat_angle > 0:
        leeward_direction = -1  # Boom to port (negative angles)
    else:
        leeward_direction = +1  # Boom to starboard (positive angles)

    # Wind torque on boom - always pushes toward leeward
    wind_pressure = 0.5 * params.air_density * apparent_speed ** 2
    wind_torque_on_boom = wind_pressure * params.sail_area * 0.8 * leeward_direction

    # Current boom position relative to limits
    boom_at_leeward_limit = (
        (leeward_direction < 0 and boat.boom_angle <= -max_sail_angle) or
        (leeward_direction > 0 and boat.boom_angle >= max_sail_angle)
    )
    boom_at_windward_limit = (
        (leeward_direction < 0 and boat.boom_angle >= max_sail_angle) or
        (leeward_direction > 0 and boat.boom_angle <= -max_sail_angle)
    )

    # Apply boom torque
    if boom_at_leeward_limit:
        # Sheet is taut - boom can't swing further out
        # Apply small restoring force to keep it at limit (sheet tension)
        sheet_tension = (abs(boat.boom_angle) - max_sail_angle) * 200
        forces.boom = -sheet_tension * np.sign(boat.boom_angle)
    else:
        # Sheet is slack - boom swings freely with wind
        forces.boom = wind_torque_on_boom

        # Add small centering force if boom is on wrong side (wind shifted)
        if (leeward_direction < 0 and boat.boom_angle > 0) or \
           (leeward_direction > 0 and boat.boom_angle < 0):
            # Boom is to windward - wind pushes it across
            forces.boom = wind_torque_on_boom * 2  # Extra push to cross centerline

    # === SAIL FORCE ON BOAT ===
    # Only generate propulsive force when sheet is taut (boom at limit)
    # and we're not in the no-go zone

    sheet_is_taut = abs(boat.boom_angle) >= max_sail_angle * 0.95

    if abs_wind_angle > params.no_go_zone:
        # In irons - sail luffs, minimal backward drift but boat turns off wind
        #
        # Real behavior: backward drift is only 1-3% of wind speed (very small)
        # but the boat naturally turns until wind is on beam or stern

        # Backward drag - very small coefficient (0.015) for realistic drift
        # At 12 knots wind, this gives ~0.2-0.3 knots backward drift
        backward_drag = 0.5 * params.air_density * apparent_speed ** 2 * params.sail_area * 0.015
        forces.fx += -backward_drag * np.cos(boat.heading)
        forces.fy += -backward_drag * np.sin(boat.heading)

        # Wind pressure for turning moments (separate from drift force)
        # The wind pushing on hull, rigging, and flapping sail creates yaw
        wind_pressure = 0.5 * params.air_density * apparent_speed ** 2

        # The boom position determines escape direction
        # Sailor holds boom to desired side to turn that way
        if abs(boat.boom_angle) > np.radians(10):
            # Backing yaw: boom to starboard (positive angle) creates force that
            # pushes the starboard-stern backward, turning bow to starboard (negative yaw)
            backing_yaw = -boat.boom_angle * wind_pressure * params.sail_area * 0.15
            forces.yaw += backing_yaw
        else:
            # Boom centered - natural tendency to fall off wind
            # "the wind will force the bow to fall off, forcing the boat into a turn"
            # This comes from asymmetric wind loading on hull and rigging
            if wind_to_boat_angle != 0:
                # How far off head-to-wind (0 at head-to-wind, grows as we deviate)
                off_center = np.pi - abs_wind_angle  # 0 to 40° in no-go zone
                # Turn away from wind - stronger as we're further off center
                natural_turn = -np.sign(wind_to_boat_angle) * wind_pressure * off_center * 2.0
                forces.yaw += natural_turn

        return forces

    if not sheet_is_taut:
        # Sheet is slack - sail flaps, no propulsive force on boat
        # (all wind energy goes into swinging the boom)
        return forces

    # Sail chord direction in world frame (boom points aft from mast)
    sail_chord_angle = boat.heading + np.pi + boat.boom_angle

    # Angle of attack: angle between apparent wind and sail chord
    angle_of_attack = normalize_angle(apparent_wind_angle - sail_chord_angle)
    aoa_magnitude = abs(angle_of_attack)

    # Lift and drag coefficients as continuous functions of angle of attack.
    # Real cambered sails generate lift over a wide range of angles:
    # - Peak lift around 20-30° AoA
    # - Gradual decline but still useful lift up to 60-70°
    # - Drag increases with AoA, peaks around 90°
    # - Both go to zero as AoA approaches 180° (wind parallel to sail)

    # Drag: peaks at 90°, zero at 0° and 180°
    cd_base = 1.2 * np.sin(aoa_magnitude)

    # Lift: cambered sail generates lift over wide AoA range
    # Use a curve that peaks around 25-30° and gradually declines
    # but maintains useful lift up to ~70°
    if aoa_magnitude < np.radians(90):
        # Front of sail to wind - normal sailing
        # Lift peaks around 25° then gradually decreases
        peak_aoa = np.radians(25)
        if aoa_magnitude < peak_aoa:
            # Building to peak
            cl = params.lift_coefficient * np.sin(2 * aoa_magnitude)
            cd = params.drag_coefficient + 0.2 * (aoa_magnitude ** 2)
        else:
            # Past peak - gradual decline but maintain useful lift
            # At 25°: full lift, at 90°: zero lift
            decline = (aoa_magnitude - peak_aoa) / (np.radians(90) - peak_aoa)
            cl = params.lift_coefficient * np.sin(2 * peak_aoa) * (1 - decline ** 1.5)
            cd = cd_base
    else:
        # AoA > 90°: wind coming from behind sail chord
        # Still generates some drag force (parachute effect) but no lift
        cl = 0.0
        cd = cd_base

    # Dynamic pressure: q = 0.5 * rho * V²
    dynamic_pressure = 0.5 * params.air_density * apparent_speed ** 2

    # Lift and drag magnitudes
    lift_magnitude = dynamic_pressure * params.sail_area * cl
    drag_magnitude = dynamic_pressure * params.sail_area * cd

    # Force directions
    wind_unit = apparent_wind[:2] / apparent_speed
    if angle_of_attack > 0:
        lift_dir = np.array([-wind_unit[1], wind_unit[0]])  # 90° CCW
    else:
        lift_dir = np.array([wind_unit[1], -wind_unit[0]])  # 90° CW

    # Total force
    force_world = lift_magnitude * lift_dir + drag_magnitude * wind_unit
    forces.fx = force_world[0]
    forces.fy = force_world[1]

    # Convert to boat-local frame for yaw and heel
    cos_h = np.cos(boat.heading)
    sin_h = np.sin(boat.heading)
    force_local_y = -sin_h * force_world[0] + cos_h * force_world[1]

    # Weather helm
    forces.yaw = params.sail_ce_aft * force_local_y * 0.2

    # Heeling moment
    forces.roll = force_local_y * params.sail_ce_height * 0.01

    return forces


def compute_rudder_forces(boat: BoatState, params: PhysicsParams) -> Forces:
    """
    Compute turning force from rudder.

    Hydrodynamic lift proportional to V².
    """
    forces = Forces()
    speed = get_boat_speed(boat)

    if speed < 0.05:
        return forces

    rudder_angle_rad = np.radians(boat.rudder_angle)

    # Hydrodynamic force: F = 0.5 * rho * V² * A * C * sin(angle)
    rudder_force = (
        0.5 * params.water_density * speed ** 2 *
        params.rudder_area * params.rudder_cl * np.sin(rudder_angle_rad)
    )

    forces.yaw = -rudder_force * params.rudder_arm
    return forces


def compute_water_drag(boat: BoatState, params: PhysicsParams) -> Forces:
    """
    Compute water resistance including keel lateral resistance.

    Uses quadratic drag model with different coefficients
    for forward motion vs lateral (keel resists sideways motion).
    """
    forces = Forces()
    speed = np.linalg.norm(boat.velocity[:2])

    if speed < 0.001:
        return forces

    cos_h = np.cos(boat.heading)
    sin_h = np.sin(boat.heading)

    # Velocity in boat frame
    vel_forward = cos_h * boat.velocity[0] + sin_h * boat.velocity[1]
    vel_lateral = -sin_h * boat.velocity[0] + cos_h * boat.velocity[1]

    # Quadratic drag in boat frame
    drag_f = -params.drag_forward * vel_forward * abs(vel_forward)
    drag_l = -params.drag_lateral * vel_lateral * abs(vel_lateral)

    # Convert back to world frame
    forces.fx = cos_h * drag_f - sin_h * drag_l
    forces.fy = sin_h * drag_f + cos_h * drag_l

    # Yaw damping
    forces.yaw = -params.yaw_damping * boat.yaw_rate

    # Keel pivot effect
    yaw_from_lateral_drag = -params.keel_clr_aft * drag_l * 0.1
    forces.yaw += yaw_from_lateral_drag

    return forces


def compute_total_forces(
    boat: BoatState,
    wind: WindState,
    sail: SailControl,
    params: PhysicsParams
) -> Forces:
    """Compute all forces acting on the boat."""
    sail_forces = compute_sail_forces(boat, wind, sail, params)
    rudder_forces = compute_rudder_forces(boat, params)
    drag_forces = compute_water_drag(boat, params)

    return Forces(
        fx=sail_forces.fx + rudder_forces.fx + drag_forces.fx,
        fy=sail_forces.fy + rudder_forces.fy + drag_forces.fy,
        yaw=sail_forces.yaw + rudder_forces.yaw + drag_forces.yaw,
        roll=sail_forces.roll + rudder_forces.roll + drag_forces.roll,
        boom=sail_forces.boom + rudder_forces.boom + drag_forces.boom,
    )
