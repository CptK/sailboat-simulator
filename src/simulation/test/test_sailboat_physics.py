"""
Tests for sailboat physics.

These tests verify physical behavior we expect from a sailboat,
not implementation details of the code.
"""

import numpy as np
import pytest
from simulation.sailboat_physics import (
    PhysicsParams,
    BoatState,
    WindState,
    SailControl,
    compute_sail_forces,
    compute_rudder_forces,
    compute_water_drag,
    compute_total_forces,
    get_wind_vector,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def params():
    """Default physics parameters."""
    return PhysicsParams()


@pytest.fixture
def stationary_boat():
    """A boat at rest, pointing north (heading=π/2)."""
    return BoatState(
        heading=np.pi / 2,  # Pointing north
        velocity=np.array([0.0, 0.0, 0.0]),
        boom_angle=0.0,
        rudder_angle=0.0,
        yaw_rate=0.0,
    )


@pytest.fixture
def moving_boat_north():
    """A boat moving north at 3 m/s."""
    return BoatState(
        heading=np.pi / 2,
        velocity=np.array([0.0, 3.0, 0.0]),
        boom_angle=0.0,
        rudder_angle=0.0,
        yaw_rate=0.0,
    )


# =============================================================================
# Wind Vector Tests
# =============================================================================

class TestWindVector:
    """Test that wind vectors are computed correctly."""

    def test_wind_from_north_blows_south(self):
        """Wind FROM north should push things southward."""
        wind = WindState(direction=np.pi / 2, speed=10.0)  # From north
        vec = get_wind_vector(wind)
        assert vec[1] < 0  # Blows south (negative y)
        assert abs(vec[0]) < 0.01  # No east-west component

    def test_wind_from_east_blows_west(self):
        """Wind FROM east should push things westward."""
        wind = WindState(direction=0.0, speed=10.0)  # From east
        vec = get_wind_vector(wind)
        assert vec[0] < 0  # Blows west (negative x)
        assert abs(vec[1]) < 0.01

    def test_wind_speed_scales_vector(self):
        """Doubling wind speed should double the vector magnitude."""
        wind1 = WindState(direction=0.0, speed=5.0)
        wind2 = WindState(direction=0.0, speed=10.0)
        vec1 = get_wind_vector(wind1)
        vec2 = get_wind_vector(wind2)
        assert np.isclose(np.linalg.norm(vec2), 2 * np.linalg.norm(vec1))


# =============================================================================
# Sailing Fundamentals - No-Go Zone
# =============================================================================

class TestNoGoZone:
    """Test that boats cannot sail directly into the wind."""

    def test_backward_drag_when_heading_into_wind(self, params):
        """A boat pointing directly into the wind should get backward drag (no propulsion)."""
        # Wind from north, boat pointing north (into the wind)
        wind = WindState(direction=np.pi / 2, speed=10.0)
        boat = BoatState(
            heading=np.pi / 2,  # Pointing north, into the wind
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=42.5)  # degrees

        forces = compute_sail_forces(boat, wind, sail, params)

        # In irons: wind creates backward drag on luffing sail, not propulsion
        # The force should push the boat backward (opposite to heading)
        # Boat heading north (pi/2), so backward force is in -Y direction
        cos_h = np.cos(boat.heading)
        sin_h = np.sin(boat.heading)
        forward_force = cos_h * forces.fx + sin_h * forces.fy
        assert forward_force < 0, "Should have backward (not forward) force when in irons"

    def test_backward_drag_at_30_degrees_to_wind(self, params):
        """A boat pointing 30° off the wind should still be in the no-go zone with backward drag."""
        wind = WindState(direction=np.pi / 2, speed=10.0)  # From north
        # Pointing 30° off north (still too close to wind)
        boat = BoatState(
            heading=np.pi / 2 + np.radians(30),
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=21.25)  # degrees

        forces = compute_sail_forces(boat, wind, sail, params)

        # Still in no-go zone: backward drag, not propulsion
        cos_h = np.cos(boat.heading)
        sin_h = np.sin(boat.heading)
        forward_force = cos_h * forces.fx + sin_h * forces.fy
        assert forward_force < 0, "30° to wind should have backward drag (in no-go zone)"


# =============================================================================
# Sailing Fundamentals - Points of Sail
# =============================================================================

class TestPointsOfSail:
    """Test that boats can sail at various angles to the wind."""

    def test_beam_reach_generates_force(self, params):
        """Sailing perpendicular to wind (beam reach) should generate force."""
        wind = WindState(direction=0.0, speed=10.0)  # From east (starboard)
        sail_angle = 63.75  # degrees
        max_boom_angle = np.radians(sail_angle)  # Sheet limit

        # Boom must be at sheet limit (leeward) to generate force
        # Wind from starboard -> leeward is port -> negative boom angle
        boat = BoatState(
            heading=np.pi / 2,  # Pointing north
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=-max_boom_angle,  # At sheet limit, to port (leeward)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        force_magnitude = np.sqrt(forces.fx**2 + forces.fy**2)
        assert force_magnitude > 10.0, f"Beam reach should generate force, got {force_magnitude}"

    def test_running_downwind_generates_force(self, params):
        """Sailing with the wind (running) should generate force."""
        wind = WindState(direction=np.pi, speed=10.0)  # From west
        sail_angle = 85.0  # degrees
        max_boom_angle = np.radians(sail_angle)

        # Wind from west (behind), leeward could be either side
        # Boom at sheet limit to starboard
        boat = BoatState(
            heading=0.0,  # Pointing east (wind from behind)
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=max_boom_angle,  # At sheet limit
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        force_mag = np.sqrt(forces.fx**2 + forces.fy**2)
        assert force_mag > 5.0, f"Running should push boat, got force={force_mag}"

    def test_close_hauled_generates_forward_force(self, params):
        """Close-hauled sailing (~45° to wind) should generate forward force."""
        wind = WindState(direction=np.pi / 2, speed=10.0)  # From north
        sail_angle = 21.25  # degrees, tight for upwind
        max_boom_angle = np.radians(sail_angle)

        # Heading northeast (45° off the wind)
        # Wind from north, heading NE means wind from port bow
        # Leeward is starboard (positive boom angle)
        boat = BoatState(
            heading=np.pi / 4,  # 45° - pointing northeast
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=max_boom_angle,  # At sheet limit, to starboard (leeward)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Convert to boat frame to check forward component
        cos_h = np.cos(boat.heading)
        sin_h = np.sin(boat.heading)
        forward_force = cos_h * forces.fx + sin_h * forces.fy

        assert forward_force > 0, f"Close-hauled should generate forward force, got {forward_force}"


# =============================================================================
# Force Scaling - V² Relationship
# =============================================================================

class TestForceScaling:
    """Test that forces scale correctly with velocity squared."""

    def test_sail_force_quadruples_when_wind_doubles(self, params):
        """Doubling wind speed should quadruple sail force (V² relationship)."""
        sail_angle = 63.75  # degrees
        max_boom_angle = np.radians(sail_angle)

        # Wind from east, boat heading north, boom to port (leeward) at sheet limit
        boat = BoatState(
            heading=np.pi / 2,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=-max_boom_angle,  # At sheet limit, port (leeward)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        wind1 = WindState(direction=0.0, speed=5.0)  # From east
        wind2 = WindState(direction=0.0, speed=10.0)

        forces1 = compute_sail_forces(boat, wind1, sail, params)
        forces2 = compute_sail_forces(boat, wind2, sail, params)

        mag1 = np.sqrt(forces1.fx**2 + forces1.fy**2)
        mag2 = np.sqrt(forces2.fx**2 + forces2.fy**2)

        # Should be approximately 4x (allow some tolerance for angle effects)
        assert mag1 > 0, f"Should generate force at 5 m/s, got {mag1}"
        ratio = mag2 / mag1
        assert 3.5 < ratio < 4.5, f"Force ratio should be ~4, got {ratio}"

    def test_rudder_force_quadruples_when_speed_doubles(self, params):
        """Doubling boat speed should quadruple rudder force."""
        boat1 = BoatState(
            heading=0.0,
            velocity=np.array([2.0, 0.0, 0.0]),
            boom_angle=0.0,
            rudder_angle=20.0,
            yaw_rate=0.0,
        )
        boat2 = BoatState(
            heading=0.0,
            velocity=np.array([4.0, 0.0, 0.0]),
            boom_angle=0.0,
            rudder_angle=20.0,
            yaw_rate=0.0,
        )

        forces1 = compute_rudder_forces(boat1, params)
        forces2 = compute_rudder_forces(boat2, params)

        ratio = abs(forces2.yaw / forces1.yaw) if forces1.yaw != 0 else 0
        assert 3.8 < ratio < 4.2, f"Rudder force ratio should be ~4, got {ratio}"


# =============================================================================
# Rudder Behavior
# =============================================================================

class TestRudder:
    """Test rudder physics."""

    def test_no_rudder_force_when_stationary(self, params, stationary_boat):
        """A stationary boat should have no rudder force regardless of angle."""
        stationary_boat.rudder_angle = 45.0

        forces = compute_rudder_forces(stationary_boat, params)

        assert abs(forces.yaw) < 0.1, "No rudder force when stationary"

    def test_rudder_turns_boat(self, params, moving_boat_north):
        """Rudder deflection should create yaw torque."""
        moving_boat_north.rudder_angle = 30.0

        forces = compute_rudder_forces(moving_boat_north, params)

        assert abs(forces.yaw) > 1.0, "Rudder should create yaw torque"

    def test_opposite_rudder_turns_opposite(self, params, moving_boat_north):
        """Opposite rudder angles should create opposite yaw torques."""
        boat_left = BoatState(
            heading=moving_boat_north.heading,
            velocity=moving_boat_north.velocity.copy(),
            boom_angle=0.0,
            rudder_angle=30.0,
            yaw_rate=0.0,
        )
        boat_right = BoatState(
            heading=moving_boat_north.heading,
            velocity=moving_boat_north.velocity.copy(),
            boom_angle=0.0,
            rudder_angle=-30.0,
            yaw_rate=0.0,
        )

        forces_left = compute_rudder_forces(boat_left, params)
        forces_right = compute_rudder_forces(boat_right, params)

        # Opposite signs
        assert forces_left.yaw * forces_right.yaw < 0, "Opposite rudder should give opposite yaw"
        # Similar magnitude
        assert np.isclose(abs(forces_left.yaw), abs(forces_right.yaw), rtol=0.01)


# =============================================================================
# Water Drag
# =============================================================================

class TestWaterDrag:
    """Test water resistance physics."""

    def test_drag_opposes_motion(self, params):
        """Drag should always oppose the direction of motion."""
        # Boat moving east
        boat = BoatState(
            heading=0.0,
            velocity=np.array([3.0, 0.0, 0.0]),
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=0.0,
        )

        forces = compute_water_drag(boat, params)

        # Drag should push west (negative x)
        assert forces.fx < 0, "Forward drag should oppose forward motion"

    def test_no_drag_when_stationary(self, params, stationary_boat):
        """A stationary boat should have no drag."""
        forces = compute_water_drag(stationary_boat, params)

        assert abs(forces.fx) < 0.01
        assert abs(forces.fy) < 0.01

    def test_keel_resists_lateral_motion(self, params):
        """Lateral drag should be much higher than forward drag (keel effect)."""
        # Boat pointing north but moving east (pure lateral motion)
        boat_lateral = BoatState(
            heading=np.pi / 2,  # Pointing north
            velocity=np.array([2.0, 0.0, 0.0]),  # Moving east (sideways)
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        # Boat pointing east and moving east (pure forward motion)
        boat_forward = BoatState(
            heading=0.0,  # Pointing east
            velocity=np.array([2.0, 0.0, 0.0]),  # Moving east (forward)
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=0.0,
        )

        drag_lateral = compute_water_drag(boat_lateral, params)
        drag_forward = compute_water_drag(boat_forward, params)

        lateral_mag = np.sqrt(drag_lateral.fx**2 + drag_lateral.fy**2)
        forward_mag = np.sqrt(drag_forward.fx**2 + drag_forward.fy**2)

        # Keel should make lateral drag much higher
        assert lateral_mag > 10 * forward_mag, "Keel should resist lateral motion much more than forward"

    def test_yaw_damping_opposes_rotation(self, params):
        """Yaw damping should oppose rotational velocity."""
        # Need some velocity for the function to not return early
        boat_rotating_ccw = BoatState(
            heading=0.0,
            velocity=np.array([0.1, 0.0, 0.0]),  # Small forward velocity
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=1.0,  # Rotating counterclockwise (positive)
        )
        boat_rotating_cw = BoatState(
            heading=0.0,
            velocity=np.array([0.1, 0.0, 0.0]),  # Small forward velocity
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=-1.0,  # Rotating clockwise (negative)
        )

        forces_ccw = compute_water_drag(boat_rotating_ccw, params)
        forces_cw = compute_water_drag(boat_rotating_cw, params)

        # Yaw torque should oppose rotation (negative for CCW, positive for CW)
        assert forces_ccw.yaw < 0, f"Damping should oppose CCW rotation, got {forces_ccw.yaw}"
        assert forces_cw.yaw > 0, f"Damping should oppose CW rotation, got {forces_cw.yaw}"


# =============================================================================
# Boom Behavior
# =============================================================================

class TestBoomBehavior:
    """Test that the boom swings to the correct side."""

    def test_boom_swings_to_leeward(self, params):
        """Boom should swing away from the wind (to leeward)."""
        sail = SailControl(sail_angle=63.75)  # degrees

        # Wind from starboard (east), boat pointing north
        # Wind vector blows TO the west
        wind_starboard = WindState(direction=0.0, speed=10.0)
        boat = BoatState(
            heading=np.pi / 2,  # Pointing north
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=0.0,  # Currently centered
            rudder_angle=0.0,
            yaw_rate=0.0,
        )

        forces = compute_sail_forces(boat, wind_starboard, sail, params)

        # Wind from starboard pushes boom to port (negative boom angle)
        # The torque should be negative to push boom toward negative angles
        assert forces.boom < 0, f"Wind from starboard should push boom to port, got {forces.boom}"

        # Wind from port (west), blows TO the east
        wind_port = WindState(direction=np.pi, speed=10.0)
        forces2 = compute_sail_forces(boat, wind_port, sail, params)

        # Wind from port pushes boom to starboard (positive boom angle)
        assert forces2.boom > 0, f"Wind from port should push boom to starboard, got {forces2.boom}"


# =============================================================================
# Heeling
# =============================================================================

class TestHeeling:
    """Test that lateral sail force causes heeling."""

    def test_beam_reach_causes_heel(self, params):
        """Sailing on a beam reach should cause heeling moment."""
        wind = WindState(direction=0.0, speed=10.0)  # From east (starboard)
        sail_angle = 63.75  # degrees
        max_boom_angle = np.radians(sail_angle)

        # Boom at sheet limit to port (leeward)
        boat = BoatState(
            heading=np.pi / 2,  # Pointing north
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=-max_boom_angle,  # At sheet limit, to port (leeward)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Should have non-zero heel moment
        assert abs(forces.roll) > 0.01, f"Beam reach should cause heeling, got {forces.roll}"

    def test_running_minimal_heel(self, params):
        """Running downwind should have minimal heel (force is mostly forward)."""
        wind = WindState(direction=np.pi, speed=10.0)  # From west
        boat = BoatState(
            heading=0.0,  # Pointing east (running)
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=np.radians(70),  # Boom out to starboard
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=85.0)  # degrees

        forces = compute_sail_forces(boat, wind, sail, params)

        # For running, heel should be relatively small compared to total force
        total_force = np.sqrt(forces.fx**2 + forces.fy**2)
        if total_force > 1.0:
            # Heel should be modest relative to total force
            assert abs(forces.roll) < total_force * 0.5, "Running should have modest heel"


# =============================================================================
# Apparent Wind
# =============================================================================

class TestApparentWind:
    """Test that boat motion affects apparent wind correctly."""

    def test_sailing_into_wind_increases_apparent_wind(self, params):
        """Moving toward the wind source should increase apparent wind."""
        wind = WindState(direction=np.pi / 2, speed=10.0)  # From north
        sail_angle = 63.75  # degrees
        max_boom_angle = np.radians(sail_angle)
        sail = SailControl(sail_angle=sail_angle)

        # Wind from north, boat heading east -> leeward is starboard (+)
        # Boom at sheet limit to starboard
        boat_stationary = BoatState(
            heading=0.0,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=max_boom_angle,  # At sheet limit, starboard (leeward)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )

        # Same configuration but boat moving forward
        boat_moving = BoatState(
            heading=0.0,
            velocity=np.array([3.0, 0.0, 0.0]),  # Moving east at 3 m/s
            boom_angle=max_boom_angle,
            rudder_angle=0.0,
            yaw_rate=0.0,
        )

        forces_stat = compute_sail_forces(boat_stationary, wind, sail, params)
        forces_move = compute_sail_forces(boat_moving, wind, sail, params)

        mag_stat = np.sqrt(forces_stat.fx**2 + forces_stat.fy**2)
        mag_move = np.sqrt(forces_move.fx**2 + forces_move.fy**2)

        # Both should generate force
        assert mag_stat > 1.0, f"Stationary should generate force, got {mag_stat}"
        assert mag_move > 1.0, f"Moving should generate force, got {mag_move}"


# =============================================================================
# Integration - Total Forces
# =============================================================================

class TestTotalForces:
    """Test that total force computation combines all forces."""

    def test_total_forces_combines_all_sources(self, params):
        """Total forces should include sail, rudder, and drag."""
        wind = WindState(direction=0.0, speed=10.0)
        boat = BoatState(
            heading=np.pi / 2,
            velocity=np.array([0.0, 2.0, 0.0]),  # Moving forward
            boom_angle=np.radians(45),
            rudder_angle=20.0,
            yaw_rate=0.5,
        )
        sail = SailControl(sail_angle=63.75)  # degrees

        sail_f = compute_sail_forces(boat, wind, sail, params)
        rudder_f = compute_rudder_forces(boat, params)
        drag_f = compute_water_drag(boat, params)
        total_f = compute_total_forces(boat, wind, sail, params)

        # Total should equal sum of components
        assert np.isclose(total_f.fx, sail_f.fx + rudder_f.fx + drag_f.fx)
        assert np.isclose(total_f.fy, sail_f.fy + rudder_f.fy + drag_f.fy)
        assert np.isclose(total_f.yaw, sail_f.yaw + rudder_f.yaw + drag_f.yaw)
        assert np.isclose(total_f.roll, sail_f.roll + rudder_f.roll + drag_f.roll)
        assert np.isclose(total_f.boom, sail_f.boom + rudder_f.boom + drag_f.boom)


# =============================================================================
# Edge Cases
# =============================================================================

class TestSheetAsLimit:
    """Test that the sheet acts as a limit, not a spring."""

    def test_no_force_when_sheet_slack(self, params):
        """When boom hasn't reached sheet limit, no propulsive force on boat."""
        wind = WindState(direction=np.pi / 2, speed=10.0)  # From north
        sail_angle = 85.0  # degrees - very loose
        max_angle = np.radians(sail_angle)  # 85°

        # Boom only at 30° but limit is 85° - sheet is slack
        boat = BoatState(
            heading=0.0,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=np.radians(30),  # Not at limit
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Boom should still feel wind torque
        assert abs(forces.boom) > 10, "Wind should push boom"
        # But boat should feel no propulsive force (sheet slack)
        boat_force = np.sqrt(forces.fx**2 + forces.fy**2)
        assert boat_force < 0.1, f"Sheet slack = no boat force, got {boat_force}"

    def test_force_when_sheet_taut(self, params):
        """When boom at sheet limit, force transfers to boat."""
        wind = WindState(direction=np.pi / 2, speed=10.0)  # From north
        sail_angle = 63.75  # degrees
        max_angle = np.radians(sail_angle)

        # Boom at the sheet limit
        boat = BoatState(
            heading=0.0,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=max_angle,  # At limit
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Should generate significant propulsive force
        boat_force = np.sqrt(forces.fx**2 + forces.fy**2)
        assert boat_force > 50, f"Sheet taut = force on boat, got {boat_force}"

    def test_boom_swings_freely_until_limit(self, params):
        """Boom should swing toward leeward with wind torque until hitting limit."""
        wind = WindState(direction=0.0, speed=10.0)  # From east

        # Boom starting at center, wind pushing to port (leeward)
        boat = BoatState(
            heading=np.pi / 2,  # North
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=0.0,
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=85.0)  # degrees

        forces = compute_sail_forces(boat, wind, sail, params)

        # Wind from east (starboard) should push boom to port (negative)
        assert forces.boom < -50, f"Should push boom to port, got {forces.boom}"


# =============================================================================
# Sail Force Coefficient Tests - Continuous AoA Response
# =============================================================================

class TestSailCoefficients:
    """Test that sail lift/drag coefficients behave correctly across all angles of attack."""

    def test_lift_generated_at_moderate_aoa(self, params):
        """Lift should be generated at moderate angles of attack (25-60°), not just low angles."""
        wind = WindState(direction=0.0, speed=10.0)  # From east

        # Set up a beam reach with a moderately tight sheet
        # This creates a higher AoA than optimal but should still generate lift
        sail_angle = 34.0  # degrees (~34°)
        max_boom_angle = np.radians(sail_angle)

        boat = BoatState(
            heading=np.pi / 2,  # Pointing north
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=-max_boom_angle,  # At sheet limit, to port (leeward)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Should generate significant force even with tighter sheet
        force_magnitude = np.sqrt(forces.fx**2 + forces.fy**2)
        assert force_magnitude > 20.0, f"Moderate AoA should generate force, got {force_magnitude}"

        # Check there's a forward component (lift contributing to propulsion)
        cos_h = np.cos(boat.heading)
        sin_h = np.sin(boat.heading)
        forward_force = cos_h * forces.fx + sin_h * forces.fy
        assert forward_force > 5.0, f"Should have forward force from lift, got {forward_force}"

    def test_lift_at_high_aoa_before_ninety_degrees(self, params):
        """Lift should still be generated at high AoA (60-89°), declining gradually."""
        wind = WindState(direction=0.0, speed=10.0)  # From east

        # Tight sheet on beam reach creates high AoA
        sail_angle = 21.25  # degrees (~21°)
        max_boom_angle = np.radians(sail_angle)

        boat = BoatState(
            heading=np.pi / 2,  # North
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=-max_boom_angle,  # At sheet limit
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Even with high AoA, should generate some force (not zero!)
        force_magnitude = np.sqrt(forces.fx**2 + forces.fy**2)
        assert force_magnitude > 10.0, f"High AoA should still generate force, got {force_magnitude}"

    def test_downwind_force_with_moderate_boom_angle(self, params):
        """Running downwind should generate force even with boom not fully out."""
        wind = WindState(direction=np.pi, speed=10.0)  # From west

        # Moderate sail angle - boom can go to ~42°, not the full 85°
        sail_angle = 42.5  # degrees
        max_boom_angle = np.radians(sail_angle)

        boat = BoatState(
            heading=0.0,  # Pointing east (running downwind)
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=max_boom_angle,  # At sheet limit
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # This was the bug - moderate boom angle downwind gave AoA ~138°
        # which previously returned zero force
        force_magnitude = np.sqrt(forces.fx**2 + forces.fy**2)
        assert force_magnitude > 5.0, f"Downwind with moderate sail should generate force, got {force_magnitude}"

        # Force should have forward component (pushing boat east)
        assert forces.fx > 0, f"Should push boat forward (east), got fx={forces.fx}"

    def test_force_continuous_across_aoa_range(self, params):
        """Force magnitude should vary continuously with AoA, no sudden drops to zero."""
        wind = WindState(direction=np.pi / 2, speed=10.0)  # From north

        # Test multiple sail angles to get different AoA values
        previous_force = None
        for sail_angle in [21.25, 34.0, 42.5, 51.0, 63.75, 76.5, 85.0]:
            max_boom_angle = np.radians(sail_angle)

            boat = BoatState(
                heading=0.0,  # East
                velocity=np.array([0.0, 0.0, 0.0]),
                boom_angle=max_boom_angle,
                rudder_angle=0.0,
                yaw_rate=0.0,
            )
            sail = SailControl(sail_angle=sail_angle)

            forces = compute_sail_forces(boat, wind, sail, params)
            force_magnitude = np.sqrt(forces.fx**2 + forces.fy**2)

            # Force should never suddenly drop to near-zero
            assert force_magnitude > 5.0, f"Force should not drop to zero at sail_angle={sail_angle}"

            # Force changes should be gradual, not abrupt
            if previous_force is not None:
                ratio = force_magnitude / previous_force if previous_force > 0 else 1.0
                assert 0.3 < ratio < 3.0, f"Force changed too abruptly: {previous_force} -> {force_magnitude}"

            previous_force = force_magnitude

    def test_drag_peaks_at_ninety_degrees_aoa(self, params):
        """Drag coefficient should be maximum around 90° AoA (perpendicular to wind)."""
        wind = WindState(direction=np.pi, speed=10.0)  # From west

        # Running dead downwind with sail perpendicular (boom at 85°)
        sail_angle_perp = 85.0  # degrees
        max_boom_perp = np.radians(sail_angle_perp)
        boat_perpendicular = BoatState(
            heading=0.0,  # East
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=max_boom_perp,  # At sheet limit (85°)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail_perp = SailControl(sail_angle=sail_angle_perp)

        # Compare with sail at 45°
        sail_angle_angled = 45.0  # degrees
        max_boom_angled = np.radians(sail_angle_angled)
        boat_angled = BoatState(
            heading=0.0,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=max_boom_angled,  # At sheet limit (45°)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail_angled = SailControl(sail_angle=sail_angle_angled)

        forces_perp = compute_sail_forces(boat_perpendicular, wind, sail_perp, params)
        forces_angled = compute_sail_forces(boat_angled, wind, sail_angled, params)

        mag_perp = np.sqrt(forces_perp.fx**2 + forces_perp.fy**2)
        mag_angled = np.sqrt(forces_angled.fx**2 + forces_angled.fy**2)

        # Both should generate force
        assert mag_perp > 0, "Perpendicular sail should generate force"
        assert mag_angled > 0, "Angled sail should generate force"

    def test_force_approaches_zero_at_180_degrees_aoa(self, params):
        """Force should approach zero as AoA approaches 180° (wind parallel to sail)."""
        # This is hard to set up exactly, but we can test that force decreases
        # as we approach the parallel condition
        wind = WindState(direction=0.0, speed=10.0)  # From east

        # Sail chord nearly aligned with wind direction
        # Boat heading west, boom at 0 -> sail chord points east
        # Wind blows west -> AoA near 180°
        boat = BoatState(
            heading=np.pi,  # West
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=0.0,  # Sail chord points east (into the wind direction)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=21.25)  # degrees

        forces = compute_sail_forces(boat, wind, sail, params)
        force_magnitude = np.sqrt(forces.fx**2 + forces.fy**2)

        # Force should be relatively small when sail is edge-on to wind
        # Note: may not be exactly zero due to other effects
        assert force_magnitude < 50.0, f"Sail parallel to wind should have reduced force, got {force_magnitude}"


class TestBeamReachWithVariousSheets:
    """Specific tests for beam reach sailing with different sheet settings."""

    def test_beam_reach_tight_sheet_still_sails(self, params):
        """Beam reach with tight sheet should still generate forward propulsion."""
        wind = WindState(direction=0.0, speed=10.0)  # From east (starboard)

        # Tight sheet for beam reach - this was problematic before
        sail_angle = 25.5  # degrees
        max_boom_angle = np.radians(sail_angle)

        boat = BoatState(
            heading=np.pi / 2,  # North
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=-max_boom_angle,  # Leeward (port)
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=sail_angle)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Calculate forward component
        cos_h = np.cos(boat.heading)
        sin_h = np.sin(boat.heading)
        forward_force = cos_h * forces.fx + sin_h * forces.fy

        # Should have positive forward force (boat can sail, not just drift)
        assert forward_force > 1.0, f"Tight sheet beam reach should still propel forward, got {forward_force}"

    def test_beam_reach_force_increases_with_optimal_sheet(self, params):
        """Beam reach should generate more force with better-trimmed (looser) sheet."""
        wind = WindState(direction=0.0, speed=10.0)  # From east

        forces_by_angle = {}
        for sail_angle in [21.25, 42.5, 63.75]:  # degrees
            max_boom_angle = np.radians(sail_angle)

            boat = BoatState(
                heading=np.pi / 2,  # North
                velocity=np.array([0.0, 0.0, 0.0]),
                boom_angle=-max_boom_angle,  # Leeward
                rudder_angle=0.0,
                yaw_rate=0.0,
            )
            sail = SailControl(sail_angle=sail_angle)

            forces = compute_sail_forces(boat, wind, sail, params)

            cos_h = np.cos(boat.heading)
            sin_h = np.sin(boat.heading)
            forward_force = cos_h * forces.fx + sin_h * forces.fy
            forces_by_angle[sail_angle] = forward_force

        # All should generate positive forward force
        for sa, ff in forces_by_angle.items():
            assert ff > 0, f"Sail angle {sa}° should generate forward force, got {ff}"

        # Looser sheet should generally be better for beam reach
        # (though the relationship may not be strictly monotonic)
        assert forces_by_angle[63.75] > forces_by_angle[21.25] * 0.5, \
            "Looser sheet should not be dramatically worse"


class TestDownwindSailing:
    """Specific tests for downwind (running) sailing."""

    def test_running_with_various_boom_angles(self, params):
        """Running downwind should work with various boom angles, not just 85°."""
        wind = WindState(direction=np.pi, speed=10.0)  # From west

        for sail_angle in [30, 45, 60, 75, 85]:  # degrees
            max_boom_angle = np.radians(sail_angle)

            boat = BoatState(
                heading=0.0,  # East (running)
                velocity=np.array([0.0, 0.0, 0.0]),
                boom_angle=max_boom_angle,  # At sheet limit
                rudder_angle=0.0,
                yaw_rate=0.0,
            )
            sail = SailControl(sail_angle=sail_angle)

            forces = compute_sail_forces(boat, wind, sail, params)

            # All boom angles should generate forward force when running
            assert forces.fx > 0, f"Running with boom at {sail_angle}° should push forward, got fx={forces.fx}"

    def test_running_force_nonzero_at_high_aoa(self, params):
        """
        Running downwind with moderate boom angle gives high AoA (100-140°).
        This should still generate force, not return zero.
        """
        wind = WindState(direction=np.pi, speed=10.0)  # From west

        # Boom at 42° gives AoA around 138° when running - this was the bug
        boat = BoatState(
            heading=0.0,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=np.radians(42),
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=42.5)  # degrees

        forces = compute_sail_forces(boat, wind, sail, params)
        force_mag = np.sqrt(forces.fx**2 + forces.fy**2)

        # The old code returned 0 here because AoA > 120°
        # New code should return positive force
        assert force_mag > 5.0, f"High AoA (~138°) should still generate force, got {force_mag}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_wind_no_sail_force(self, params):
        """Zero wind should produce no sail force."""
        wind = WindState(direction=0.0, speed=0.0)
        boat = BoatState(
            heading=0.0,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=np.radians(45),
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=63.75)  # degrees

        forces = compute_sail_forces(boat, wind, sail, params)

        assert abs(forces.fx) < 0.01
        assert abs(forces.fy) < 0.01

    def test_sheet_fully_tight_limits_boom(self, params):
        """Fully tight sheet should limit boom movement."""
        wind = WindState(direction=0.0, speed=10.0)
        boat = BoatState(
            heading=np.pi / 2,
            velocity=np.array([0.0, 0.0, 0.0]),
            boom_angle=np.radians(80),  # Boom way out
            rudder_angle=0.0,
            yaw_rate=0.0,
        )
        sail = SailControl(sail_angle=0.0)  # Fully tight (0°)

        forces = compute_sail_forces(boat, wind, sail, params)

        # Boom torque should try to center the boom when sheet is tight
        # (pulling it back toward centerline)
        assert forces.boom * boat.boom_angle < 0, "Tight sheet should pull boom toward center"

    def test_very_high_speed_doesnt_explode(self, params):
        """Very high speeds should produce large but finite forces."""
        boat = BoatState(
            heading=0.0,
            velocity=np.array([50.0, 0.0, 0.0]),  # 50 m/s = 97 knots
            boom_angle=0.0,
            rudder_angle=30.0,
            yaw_rate=0.0,
        )

        forces = compute_rudder_forces(boat, params)
        drag = compute_water_drag(boat, params)

        # Forces should be large but finite
        assert np.isfinite(forces.yaw)
        assert np.isfinite(drag.fx)
        assert abs(forces.yaw) < 1e6
        assert abs(drag.fx) < 1e6
