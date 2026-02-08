"""
Comprehensive test suite for the gravity factor theory.

Tests verify that the variable-c (gravity factor) description of light
propagation in gravitational fields matches:
  - Standard General Relativity predictions
  - Known observational data (solar redshift, GPS, Shapiro delay, etc.)
  - Limiting cases (far field, event horizon, photon sphere)

Run with: pytest gravity_light/verification/test_gravity_factor.py -v
"""

import numpy as np
import pytest

from gravity_light.verification.constants import (
    G, c, c2,
    M_sun, R_sun, rs_sun,
    M_earth, R_earth, rs_earth,
    M_ns, R_ns, rs_ns,
    M_sgra, rs_sgra,
    AU, r_gps,
    schwarzschild_radius,
    c_radial, c_tangential, c_effective,
    refractive_index,
    gravitational_redshift_ratio,
    deflection_angle, shapiro_delay,
    photon_sphere_radius,
    weak_field_c_eff,
    gravitational_potential,
    solar_redshift_observed,
    solar_deflection_arcsec,
    shapiro_delay_sun_us,
    gps_gravitational_shift_us_per_day,
    cassini_gamma, cassini_gamma_uncertainty,
    pound_rebka_height, pound_rebka_fractional_shift,
)
from gravity_light.verification.ray_tracer import (
    GravitationalRayTracer, trace_solar_deflection,
)


# =========================================================================
# Helper constants
# =========================================================================

ARCSEC_PER_RAD = 3600.0 * 180.0 / np.pi


# =========================================================================
# Test 1: c_eff(r -> infinity) = c
# =========================================================================

class TestFarFieldLimit:
    """Tests that the speed of light approaches c far from any mass."""

    def test_ceff_at_infinity_radial(self):
        """c_radial should approach c as r -> infinity."""
        r_far = 1e20  # very far away
        cr = c_radial(r_far, rs_sun)
        assert abs(cr / c - 1.0) < 1e-15, (
            f"c_radial at r=1e20 m should be ~c, got {cr/c}"
        )

    def test_ceff_at_infinity_tangential(self):
        """c_tangential should approach c as r -> infinity."""
        r_far = 1e20
        ct = c_tangential(r_far, rs_sun)
        assert abs(ct / c - 1.0) < 1e-15, (
            f"c_tangential at r=1e20 m should be ~c, got {ct/c}"
        )

    def test_ceff_at_infinity_general(self):
        """c_effective at arbitrary angle should approach c as r -> infinity."""
        r_far = 1e20
        for theta in [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]:
            ce = c_effective(r_far, rs_sun, theta)
            assert abs(ce / c - 1.0) < 1e-14, (
                f"c_effective at r=1e20, theta={theta:.2f} should be ~c, got {ce/c}"
            )


# =========================================================================
# Test 2: c_eff(r_s) = 0 for radial propagation (event horizon)
# =========================================================================

class TestEventHorizon:
    """Tests that the radial speed of light vanishes at the event horizon."""

    def test_c_radial_at_horizon(self):
        """c_radial should be exactly 0 at r = r_s."""
        cr = c_radial(rs_sun, rs_sun)
        assert cr == 0.0, f"c_radial at r=r_s should be 0, got {cr}"

    def test_c_tangential_at_horizon(self):
        """c_tangential should be 0 at r = r_s."""
        ct = c_tangential(rs_sun, rs_sun)
        assert abs(ct) < 1e-10, f"c_tangential at r=r_s should be 0, got {ct}"

    def test_c_effective_radial_at_horizon(self):
        """c_effective for radial propagation should be 0 at r = r_s."""
        ce = c_effective(rs_sun, rs_sun, theta=0)
        assert abs(ce) < 1e-10, (
            f"c_effective(r_s, theta=0) should be 0, got {ce}"
        )

    def test_c_effective_approaches_zero(self):
        """c_eff should monotonically approach 0 near the horizon."""
        r_values = np.array([1.001, 1.01, 1.1, 1.5, 2.0]) * rs_sun
        c_values = np.array([c_radial(r, rs_sun) for r in r_values])
        # Should be monotonically increasing (moving away from horizon)
        assert np.all(np.diff(c_values) > 0), (
            "c_radial should increase monotonically away from horizon"
        )
        # Closest value should be near zero
        assert c_values[0] / c < 0.002, (
            f"c_radial at 1.001*r_s should be very small, got {c_values[0]/c}"
        )


# =========================================================================
# Test 3: Gravitational redshift matches known values
# =========================================================================

class TestGravitationalRedshift:
    """Tests gravitational redshift predictions against observations."""

    def test_solar_redshift(self):
        """Solar gravitational redshift should match observed value of 2.12e-6."""
        # z = 1 - f_recv/f_emit = 1 - sqrt((1-rs/R_sun)/(1-rs/inf))
        # For receiver at infinity: z = 1 - sqrt(1 - rs/R_sun) ~ rs/(2*R_sun)
        freq_ratio = gravitational_redshift_ratio(R_sun, np.inf, rs_sun)
        z_predicted = 1.0 - freq_ratio
        # Allow 1% tolerance (the observed value has its own uncertainty)
        assert abs(z_predicted - solar_redshift_observed) / solar_redshift_observed < 0.01, (
            f"Solar redshift: predicted {z_predicted:.4e}, "
            f"observed {solar_redshift_observed:.4e}"
        )

    def test_solar_redshift_formula(self):
        """Solar redshift should equal GM_sun/(R_sun * c^2) in weak field."""
        z_weak = G * M_sun / (R_sun * c2)
        assert abs(z_weak - solar_redshift_observed) / solar_redshift_observed < 0.01, (
            f"Weak field solar redshift: {z_weak:.4e} vs {solar_redshift_observed:.4e}"
        )

    def test_gps_gravitational_time_dilation(self):
        """GPS gravitational time dilation should be ~45.85 us/day."""
        # Clocks at GPS altitude run faster due to weaker gravity
        freq_ratio = gravitational_redshift_ratio(R_earth, r_gps, rs_earth)
        # Time dilation: satellite clock gains time at rate (1/freq_ratio - 1)
        shift_us_per_day = (1.0 / freq_ratio - 1.0) * 86400.0 * 1e6
        # Allow 0.5% tolerance
        assert abs(shift_us_per_day - gps_gravitational_shift_us_per_day) / gps_gravitational_shift_us_per_day < 0.005, (
            f"GPS time dilation: predicted {shift_us_per_day:.2f} us/day, "
            f"expected {gps_gravitational_shift_us_per_day:.2f} us/day"
        )

    def test_pound_rebka(self):
        """Pound-Rebka experiment: redshift over 22.5 m at Earth surface."""
        # z = g*h/c^2 where g = GM/R^2
        g = G * M_earth / R_earth ** 2
        z_predicted = g * pound_rebka_height / c2
        # Allow 2% tolerance (the measurement had ~1% precision)
        assert abs(z_predicted - pound_rebka_fractional_shift) / pound_rebka_fractional_shift < 0.02, (
            f"Pound-Rebka: predicted {z_predicted:.4e}, "
            f"measured {pound_rebka_fractional_shift:.4e}"
        )

    def test_redshift_symmetry(self):
        """Redshift from A to B should be inverse of blueshift from B to A."""
        r1 = 2 * R_earth
        r2 = 5 * R_earth
        ratio_12 = gravitational_redshift_ratio(r1, r2, rs_earth)
        ratio_21 = gravitational_redshift_ratio(r2, r1, rs_earth)
        assert abs(ratio_12 * ratio_21 - 1.0) < 1e-14, (
            f"Redshift product should be 1, got {ratio_12 * ratio_21}"
        )


# =========================================================================
# Test 4: Shapiro time delay
# =========================================================================

class TestShapiroDelay:
    """Tests Shapiro time delay calculation."""

    def test_shapiro_delay_magnitude(self):
        """Shapiro delay for solar-limb grazing should be hundreds of us one-way.

        The standard formula gives dt = (4GM/c^3)[ln(4*r1*r2/b^2) + 1].
        For Earth-Mars at superior conjunction with b ~ R_sun, the one-way
        delay is ~277 us. The observed ~240 us for the Cassini experiment
        uses a different geometry and accounts for the actual spacecraft
        trajectory.
        """
        # Compute one-way delay for signal from Earth to Mars grazing the Sun
        r_mars = 2.5 * AU
        b = R_sun
        dt = shapiro_delay(M_sun, AU, r_mars, b)
        dt_us = dt * 1e6

        # One-way delay from the formula: ~277 us for this geometry
        assert 200.0 < dt_us < 400.0, (
            f"One-way Shapiro delay should be ~277 us, got {dt_us:.1f} us"
        )

    def test_shapiro_round_trip(self):
        """Round-trip Shapiro delay should be in the hundreds of us range."""
        r_mars = 2.5 * AU
        b = R_sun
        dt_one_way = shapiro_delay(M_sun, AU, r_mars, b)
        dt_round_trip_us = 2 * dt_one_way * 1e6

        # Round-trip: ~554 us for Earth-Mars geometry.
        # The ~240 us Cassini value is for different geometry/one-way.
        assert 400.0 < dt_round_trip_us < 700.0, (
            f"Round-trip Shapiro delay should be ~554 us, got {dt_round_trip_us:.1f} us"
        )

    def test_shapiro_delay_scales_with_mass(self):
        """Shapiro delay should scale linearly with mass."""
        b = R_sun
        r1 = AU
        r2 = 2 * AU
        dt1 = shapiro_delay(M_sun, r1, r2, b)
        dt2 = shapiro_delay(2 * M_sun, r1, r2, b)
        assert abs(dt2 / dt1 - 2.0) < 1e-10, (
            f"Shapiro delay should double with mass, ratio = {dt2/dt1}"
        )

    def test_shapiro_delay_positive(self):
        """Shapiro delay should always be positive (light is delayed)."""
        for b_factor in [1.0, 2.0, 10.0, 100.0]:
            b = b_factor * R_sun
            dt = shapiro_delay(M_sun, AU, 2 * AU, b)
            assert dt > 0, f"Shapiro delay should be positive, got {dt} for b={b}"


# =========================================================================
# Test 5: Light deflection angle
# =========================================================================

class TestLightDeflection:
    """Tests gravitational light deflection predictions."""

    def test_solar_limb_deflection(self):
        """Solar limb deflection should be 1.75 arcseconds."""
        delta = deflection_angle(M_sun, R_sun)
        delta_arcsec = delta * ARCSEC_PER_RAD
        # Allow 0.5% tolerance
        assert abs(delta_arcsec - solar_deflection_arcsec) / solar_deflection_arcsec < 0.005, (
            f"Solar deflection: {delta_arcsec:.4f} arcsec, "
            f"expected {solar_deflection_arcsec:.4f} arcsec"
        )

    def test_deflection_inversely_proportional_to_b(self):
        """Deflection angle should scale as 1/b."""
        b1 = R_sun
        b2 = 2 * R_sun
        d1 = deflection_angle(M_sun, b1)
        d2 = deflection_angle(M_sun, b2)
        assert abs(d1 / d2 - 2.0) < 1e-10, (
            f"Deflection should halve when b doubles, ratio = {d1/d2}"
        )

    def test_deflection_proportional_to_mass(self):
        """Deflection angle should scale linearly with mass."""
        b = R_sun
        d1 = deflection_angle(M_sun, b)
        d2 = deflection_angle(2 * M_sun, b)
        assert abs(d2 / d1 - 2.0) < 1e-10, (
            f"Deflection should double with mass, ratio = {d2/d1}"
        )


# =========================================================================
# Test 6: Photon sphere
# =========================================================================

class TestPhotonSphere:
    """Tests photon sphere radius calculation."""

    def test_photon_sphere_is_1_5_rs(self):
        """Photon sphere should be at r = 1.5 * r_s."""
        r_ph = photon_sphere_radius(rs_sun)
        assert abs(r_ph / rs_sun - 1.5) < 1e-15, (
            f"Photon sphere should be at 1.5 r_s, got {r_ph/rs_sun}"
        )

    def test_photon_sphere_for_black_hole(self):
        """Photon sphere for Sgr A* should be at 1.5 * r_s."""
        r_ph = photon_sphere_radius(rs_sgra)
        expected = 1.5 * rs_sgra
        assert abs(r_ph - expected) / expected < 1e-15, (
            f"Sgr A* photon sphere: {r_ph:.3e} m, expected {expected:.3e} m"
        )

    def test_photon_sphere_outside_horizon(self):
        """Photon sphere should always be outside the event horizon."""
        for rs in [rs_sun, rs_earth, rs_ns, rs_sgra]:
            r_ph = photon_sphere_radius(rs)
            assert r_ph > rs, (
                f"Photon sphere {r_ph} should be outside horizon {rs}"
            )


# =========================================================================
# Test 7: PPN parameter gamma
# =========================================================================

class TestPPNGamma:
    """Tests that the gravity factor gives PPN gamma = 1."""

    def test_gamma_from_deflection(self):
        """Extract gamma from deflection: delta = (1+gamma)*2GM/(bc^2)."""
        b = R_sun
        delta = deflection_angle(M_sun, b)
        # delta = 4GM/(bc^2) should equal (1+gamma)*2GM/(bc^2) with gamma=1
        gamma = delta * b * c2 / (2 * G * M_sun) - 1.0
        assert abs(gamma - 1.0) < 1e-10, (
            f"PPN gamma should be 1.0, got {gamma}"
        )

    def test_gamma_consistent_with_cassini(self):
        """Our gamma=1 should be consistent with Cassini measurement."""
        gamma_theory = 1.0
        assert abs(gamma_theory - cassini_gamma) < 3 * cassini_gamma_uncertainty, (
            f"gamma_theory={gamma_theory} inconsistent with "
            f"Cassini={cassini_gamma}+/-{cassini_gamma_uncertainty}"
        )


# =========================================================================
# Test 8: GPS time correction
# =========================================================================

class TestGPSCorrection:
    """Tests GPS gravitational time correction."""

    def test_gps_gravitational_correction(self):
        """Gravitational time correction for GPS should be ~45.85 us/day."""
        freq_ratio = gravitational_redshift_ratio(R_earth, r_gps, rs_earth)
        shift_us_per_day = (1.0 / freq_ratio - 1.0) * 86400.0 * 1e6

        # Should match to within 0.5%
        assert abs(shift_us_per_day - gps_gravitational_shift_us_per_day) < 0.25, (
            f"GPS correction: {shift_us_per_day:.2f} us/day, "
            f"expected {gps_gravitational_shift_us_per_day:.2f} us/day"
        )

    def test_gps_clocks_run_faster(self):
        """GPS satellite clocks should run faster than ground clocks (gravity)."""
        freq_ratio = gravitational_redshift_ratio(R_earth, r_gps, rs_earth)
        # freq_ratio < 1 means photon received at higher altitude has lower
        # frequency, i.e. the satellite clock runs faster
        shift = 1.0 / freq_ratio - 1.0
        assert shift > 0, "GPS clocks should run faster than ground clocks"


# =========================================================================
# Test 9: Weak field approximation
# =========================================================================

class TestWeakFieldApproximation:
    """Tests weak field approximation accuracy."""

    def test_weak_field_earth_surface(self):
        """Weak field should be excellent at Earth's surface (rs/r ~ 1.4e-9)."""
        c_exact = c_radial(R_earth, rs_earth)
        c_weak = weak_field_c_eff(M_earth, R_earth)
        rel_err = abs(c_exact - c_weak) / c_exact
        assert rel_err < 1e-17, (
            f"Weak field at Earth surface: relative error {rel_err:.2e} > 1e-17"
        )

    def test_weak_field_sun_surface(self):
        """Weak field should be good at Sun's surface (rs/r ~ 4.2e-6)."""
        c_exact = c_radial(R_sun, rs_sun)
        c_weak = weak_field_c_eff(M_sun, R_sun)
        rel_err = abs(c_exact - c_weak) / c_exact
        # For Sun, rs/R ~ 4.2e-6, so second-order error ~ (rs/R)^2 ~ 1.8e-11
        assert rel_err < 1e-10, (
            f"Weak field at Sun surface: relative error {rel_err:.2e} > 1e-10"
        )

    def test_weak_field_fails_near_ns(self):
        """Weak field should have noticeable error for tangential speed near NS.

        The weak-field approximation c*(1 + 2*Phi/c^2) = c*(1 - rs/r) is
        algebraically identical to the exact radial coordinate speed. However,
        the tangential speed c*sqrt(1 - rs/r) differs from the weak-field
        approximation c*(1 - rs/(2r)) at second order. For a neutron star
        where rs/r ~ 0.41, this difference is significant.
        """
        # Compare tangential speed (exact vs weak-field approximation)
        c_exact_tangential = c_tangential(R_ns, rs_ns)
        # Weak-field tangential: c * (1 - rs/(2r)) = c * (1 - GM/(r*c^2))
        # which is the first-order expansion of c*sqrt(1 - rs/r)
        c_weak_tangential = c * (1.0 - rs_ns / (2.0 * R_ns))
        rel_err = abs(c_exact_tangential - c_weak_tangential) / c_exact_tangential
        # rs_ns/R_ns ~ 0.41, so second-order term ~ (rs/(2r))^2 ~ 0.04
        assert rel_err > 0.01, (
            f"Weak field tangential should fail near NS: relative error "
            f"{rel_err:.2e} is suspiciously small"
        )

    def test_weak_field_always_less_than_c(self):
        """Weak field c_eff should be < c for any mass at any radius."""
        for M, r in [(M_earth, R_earth), (M_sun, R_sun), (M_ns, 10 * R_ns)]:
            c_weak = weak_field_c_eff(M, r)
            assert c_weak < c, (
                f"Weak field c_eff should be < c, got {c_weak/c} * c "
                f"for M={M:.2e}, r={r:.2e}"
            )


# =========================================================================
# Test 10: Schwarzschild radius calculation
# =========================================================================

class TestSchwarzschildRadius:
    """Tests Schwarzschild radius computations."""

    def test_schwarzschild_sun(self):
        """Schwarzschild radius of Sun should be ~2953 m."""
        rs = schwarzschild_radius(M_sun)
        assert abs(rs - 2953.0) < 5.0, (
            f"Sun rs = {rs:.1f} m, expected ~2953 m"
        )

    def test_schwarzschild_earth(self):
        """Schwarzschild radius of Earth should be ~8.87 mm."""
        rs = schwarzschild_radius(M_earth)
        assert abs(rs - 0.00887) < 0.0005, (
            f"Earth rs = {rs:.5f} m, expected ~0.00887 m"
        )

    def test_schwarzschild_scales_with_mass(self):
        """Schwarzschild radius should scale linearly with mass."""
        rs1 = schwarzschild_radius(M_sun)
        rs2 = schwarzschild_radius(2 * M_sun)
        assert abs(rs2 / rs1 - 2.0) < 1e-10


# =========================================================================
# Test 11: Refractive index properties
# =========================================================================

class TestRefractiveIndex:
    """Tests refractive index of gravitational field."""

    def test_refractive_index_at_infinity(self):
        """Refractive index should be 1 far from any mass."""
        n = refractive_index(1e20, rs_sun, theta=0)
        assert abs(n - 1.0) < 1e-14

    def test_refractive_index_greater_than_1(self):
        """Refractive index should be >= 1 everywhere outside horizon."""
        for r_factor in [1.01, 1.1, 2.0, 10.0, 1000.0]:
            r = r_factor * rs_sun
            for theta in [0, np.pi / 4, np.pi / 2]:
                n = refractive_index(r, rs_sun, theta)
                assert n >= 1.0, (
                    f"n should be >= 1, got {n} at r/rs={r_factor}, theta={theta}"
                )

    def test_refractive_index_diverges_at_horizon(self):
        """Refractive index should diverge as r -> r_s."""
        r = 1.001 * rs_sun
        n = refractive_index(r, rs_sun, theta=0)
        assert n > 100, (
            f"n should be very large near horizon, got {n} at r=1.001*rs"
        )

    def test_radial_index_larger_than_tangential(self):
        """Radial refractive index should be larger than tangential."""
        for r_factor in [2.0, 5.0, 10.0]:
            r = r_factor * rs_sun
            n_rad = refractive_index(r, rs_sun, theta=0)
            n_tan = refractive_index(r, rs_sun, theta=np.pi / 2)
            assert n_rad >= n_tan, (
                f"n_radial should >= n_tangential: "
                f"n_r={n_rad:.6f}, n_t={n_tan:.6f} at r/rs={r_factor}"
            )


# =========================================================================
# Test 12: c_effective direction dependence
# =========================================================================

class TestDirectionDependence:
    """Tests the angular dependence of c_effective."""

    def test_c_radial_less_than_tangential(self):
        """Radial speed of light should be <= tangential speed."""
        r = 5 * rs_sun  # where the effect is significant
        cr = c_radial(r, rs_sun)
        ct = c_tangential(r, rs_sun)
        assert cr <= ct, f"c_r={cr} should be <= c_t={ct}"

    def test_ceff_continuous_in_theta(self):
        """c_effective should vary continuously with theta."""
        r = 10 * rs_sun
        thetas = np.linspace(0, np.pi / 2, 100)
        c_vals = np.array([c_effective(r, rs_sun, th) for th in thetas])
        # Check that values are between c_r and c_t
        assert np.all(c_vals >= c_radial(r, rs_sun) - 1e-5)
        assert np.all(c_vals <= c_tangential(r, rs_sun) + 1e-5)

    def test_ceff_at_theta_0_is_radial(self):
        """c_effective at theta=0 should equal c_radial."""
        r = 3 * rs_sun
        ce = c_effective(r, rs_sun, theta=0)
        cr = c_radial(r, rs_sun)
        assert abs(ce - cr) < 1e-5, f"c_eff(theta=0) = {ce}, c_r = {cr}"

    def test_ceff_at_theta_pi2_is_tangential(self):
        """c_effective at theta=pi/2 should equal c_tangential."""
        r = 3 * rs_sun
        ce = c_effective(r, rs_sun, theta=np.pi / 2)
        ct = c_tangential(r, rs_sun)
        assert abs(ce - ct) < 1e-5, f"c_eff(theta=pi/2) = {ce}, c_t = {ct}"


# =========================================================================
# Test 13: Numerical ray tracer
# =========================================================================

class TestRayTracer:
    """Tests the numerical ray tracer."""

    def test_straight_ray_no_mass(self):
        """Ray with M=0 should travel in a straight line."""
        # Use a very small mass (effectively zero, but avoid division issues)
        tracer = GravitationalRayTracer(1.0)  # 1 kg mass, negligible
        x0 = -1e12
        y0 = 1e10
        vx0 = 1.0
        vy0 = 0.0

        xs, ys, ts, vxs, vys = tracer.trace_ray(
            x0, y0, vx0, vy0, ds=1e9, n_steps=2000, r_stop=1e12
        )

        # y should remain essentially constant
        y_deviation = np.max(np.abs(ys - y0))
        assert y_deviation < 1.0, (
            f"Straight ray y deviation: {y_deviation:.2e} m"
        )

    def test_solar_deflection_numerical(self):
        """Numerical deflection at solar limb should match analytical."""
        result = trace_solar_deflection(b_solar_radii=1.0, ds_factor=50.0)

        # Allow 5% tolerance for the numerical integration
        assert result['relative_error'] < 0.05, (
            f"Numerical deflection error {result['relative_error']:.2e} > 5%: "
            f"numerical={result['deflection_numerical_arcsec']:.4f}, "
            f"analytical={result['deflection_analytical_arcsec']:.4f}"
        )

    def test_deflection_scales_with_impact_parameter(self):
        """Numerical deflection should scale as ~1/b."""
        r1 = trace_solar_deflection(b_solar_radii=2.0, ds_factor=50.0)
        r2 = trace_solar_deflection(b_solar_radii=4.0, ds_factor=50.0)

        ratio = r1['deflection_numerical'] / r2['deflection_numerical']
        # Should be approximately 2.0 (since b2 = 2*b1)
        assert abs(ratio - 2.0) < 0.15, (
            f"Deflection ratio should be ~2.0, got {ratio:.3f}"
        )

    def test_ray_tracer_time_positive(self):
        """Coordinate time should always increase along the ray."""
        result = trace_solar_deflection(b_solar_radii=2.0, ds_factor=50.0)
        tracer = GravitationalRayTracer(M_sun)
        b = 2.0 * R_sun
        xs, ys, ts, vxs, vys = tracer.trace_ray(
            -2 * AU, b, 1.0, 0.0, ds=R_sun / 50.0, n_steps=500000, r_stop=2 * AU
        )
        assert np.all(np.diff(ts) > 0), "Time should always increase"


# =========================================================================
# Test 14: Consistency checks
# =========================================================================

class TestConsistency:
    """Cross-checks between different formulas."""

    def test_c_radial_from_c_effective(self):
        """c_radial should equal c_effective at theta=0."""
        for r_factor in [2.0, 5.0, 50.0, 1000.0]:
            r = r_factor * rs_sun
            cr = c_radial(r, rs_sun)
            ce = c_effective(r, rs_sun, theta=0.0)
            assert abs(cr - ce) / c < 1e-14

    def test_c_tangential_from_c_effective(self):
        """c_tangential should equal c_effective at theta=pi/2."""
        for r_factor in [2.0, 5.0, 50.0, 1000.0]:
            r = r_factor * rs_sun
            ct = c_tangential(r, rs_sun)
            ce = c_effective(r, rs_sun, theta=np.pi / 2)
            assert abs(ct - ce) / c < 1e-14

    def test_n_times_ceff_equals_c(self):
        """n(r) * c_eff(r) should equal c."""
        for r_factor in [2.0, 5.0, 50.0]:
            r = r_factor * rs_sun
            for theta in [0, np.pi / 4, np.pi / 2]:
                n = refractive_index(r, rs_sun, theta)
                ce = c_effective(r, rs_sun, theta)
                assert abs(n * ce - c) / c < 1e-14

    def test_deflection_self_consistency(self):
        """deflection_angle should equal 2*rs/b (since 4GM/(bc^2) = 2rs/b)."""
        b = R_sun
        delta = deflection_angle(M_sun, b)
        delta_from_rs = 2 * rs_sun / b
        assert abs(delta - delta_from_rs) / delta < 1e-10


# =========================================================================
# Test 15: Gravitational potential and field properties
# =========================================================================

class TestGravitationalField:
    """Tests gravitational potential and related quantities."""

    def test_potential_negative(self):
        """Gravitational potential should be negative."""
        phi = gravitational_potential(M_sun, R_sun)
        assert phi < 0, f"Potential should be negative, got {phi}"

    def test_potential_approaches_zero(self):
        """Potential should approach 0 at large distances."""
        # At r = 1e25 m, Phi = -GM/r ~ -1.33e-5 which is very small
        phi = gravitational_potential(M_sun, 1e25)
        assert abs(phi) < 1e-1, f"Potential at 1e25 m should be ~0, got {phi}"

    def test_potential_inversely_proportional_to_r(self):
        """Potential should scale as 1/r."""
        r1 = R_sun
        r2 = 2 * R_sun
        phi1 = gravitational_potential(M_sun, r1)
        phi2 = gravitational_potential(M_sun, r2)
        assert abs(phi1 / phi2 - 2.0) < 1e-10

    def test_schwarzschild_radius_from_potential(self):
        """r_s should be where |Phi|/c^2 = 1/2."""
        phi_at_rs = gravitational_potential(M_sun, rs_sun)
        assert abs(abs(phi_at_rs) / c2 - 0.5) < 1e-10


# =========================================================================
# Test 16: Extreme cases
# =========================================================================

class TestExtremeCases:
    """Tests behavior in extreme scenarios."""

    def test_very_large_mass(self):
        """Should handle supermassive black hole parameters."""
        M_smbh = 1e10 * M_sun
        rs_smbh = schwarzschild_radius(M_smbh)
        assert rs_smbh > 0
        r = 3 * rs_smbh
        cr = c_radial(r, rs_smbh)
        assert 0 < cr < c

    def test_very_small_mass(self):
        """For negligible mass, c_eff should be c everywhere."""
        M_tiny = 1.0  # 1 kg
        rs_tiny = schwarzschild_radius(M_tiny)
        r = 1.0  # 1 meter away
        cr = c_radial(r, rs_tiny)
        assert abs(cr / c - 1.0) < 1e-20

    def test_photon_sphere_tangential_speed(self):
        """At the photon sphere, c_tangential has a specific value."""
        r_ph = photon_sphere_radius(rs_sun)
        ct = c_tangential(r_ph, rs_sun)
        # At r = 1.5*rs: c_t = c*sqrt(1 - rs/(1.5*rs)) = c*sqrt(1/3)
        expected = c * np.sqrt(1.0 / 3.0)
        assert abs(ct - expected) / expected < 1e-14, (
            f"c_t at photon sphere: {ct:.6f}, expected {expected:.6f}"
        )

    def test_isco_speed(self):
        """At the innermost stable circular orbit (r=3*rs), verify c_eff."""
        r_isco = 3 * rs_sun  # 6 GM/c^2
        ct = c_tangential(r_isco, rs_sun)
        # c_t = c*sqrt(1 - rs/(3*rs)) = c*sqrt(2/3)
        expected = c * np.sqrt(2.0 / 3.0)
        assert abs(ct - expected) / expected < 1e-14


# =========================================================================
# Test 17: Array operations
# =========================================================================

class TestArrayOperations:
    """Tests that functions work correctly with numpy arrays."""

    def test_c_radial_array(self):
        """c_radial should work with arrays."""
        r = np.array([2, 5, 10, 100]) * rs_sun
        cr = c_radial(r, rs_sun)
        assert cr.shape == (4,)
        assert np.all(cr > 0)
        assert np.all(cr < c)
        # Should be monotonically increasing
        assert np.all(np.diff(cr) > 0)

    def test_c_effective_array(self):
        """c_effective should work with r as array."""
        r = np.linspace(2 * rs_sun, 100 * rs_sun, 50)
        ce = c_effective(r, rs_sun, theta=np.pi / 4)
        assert ce.shape == (50,)
        assert np.all(ce > 0)
        assert np.all(ce < c)

    def test_redshift_array_r(self):
        """gravitational_redshift_ratio should handle array inputs."""
        r_emit = np.array([R_sun, 2 * R_sun, 10 * R_sun])
        ratios = gravitational_redshift_ratio(r_emit, np.inf, rs_sun)
        assert ratios.shape == (3,)
        # All should be < 1 (redshift when emitted deeper in potential)
        assert np.all(ratios < 1.0)
        # Should increase (less redshift) as r_emit increases
        assert np.all(np.diff(ratios) > 0)
