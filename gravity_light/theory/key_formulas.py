"""
Gravity Factor Framework: Key Formulas
=======================================

A complete set of functions implementing the "gravity factor" reformulation
of General Relativity's Schwarzschild solution. In this framework, the
effects of gravity are described by a variable coordinate speed of light
in flat spacetime, reproducing all predictions of GR exactly.

The LOCAL speed of light is always c. The functions here compute the
COORDINATE speed as measured by a distant (asymptotic) observer using
Schwarzschild coordinates.

Physical constants are in SI units throughout.

References
----------
- Einstein (1911): Original variable-c insight
- Dicke (1957): Formal equivalence proof
- Schwarzschild (1916): The exact vacuum solution
- Will (1993): PPN formalism and experimental tests
- Bertotti, Iess, Tortora (2003): Cassini constraint on gamma

Author: Theoretical Physics Research Module
"""

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Physical Constants (SI)
# ---------------------------------------------------------------------------
C = 2.99792458e8        # speed of light in vacuum [m/s]
G = 6.67430e-11         # gravitational constant [m^3 kg^-1 s^-2]
M_SUN = 1.98892e30      # solar mass [kg]
M_EARTH = 5.9722e24     # Earth mass [kg]
R_SUN = 6.9634e8        # solar radius [m]
R_EARTH = 6.371e6       # Earth mean radius [m]
AU = 1.495978707e11     # astronomical unit [m]

# Derived
R_S_SUN = 2 * G * M_SUN / C**2   # Schwarzschild radius of the Sun [m] ~ 2953 m


# ===========================================================================
# Core Gravity Factor Functions
# ===========================================================================

def schwarzschild_radius(M: float) -> float:
    """
    Compute the Schwarzschild radius of a mass M.

    The Schwarzschild radius is the radius of the event horizon for a
    non-rotating, uncharged black hole of mass M.

        r_s = 2GM / c^2

    Parameters
    ----------
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Schwarzschild radius [m].

    Examples
    --------
    >>> schwarzschild_radius(M_SUN)  # ~2953 m
    2953.25...
    >>> schwarzschild_radius(M_EARTH)  # ~0.00887 m
    0.00887...
    """
    return 2 * G * M / C**2


def photon_sphere_radius(M: float) -> float:
    """
    Compute the photon sphere radius for a Schwarzschild black hole.

    The photon sphere is the unstable circular orbit for photons. It lies
    at r = 1.5 * r_s = 3GM/c^2. Inside this radius, no stable circular
    photon orbits exist; outside, photons escape.

        r_ps = (3/2) * r_s = 3GM / c^2

    Parameters
    ----------
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Photon sphere radius [m].

    Notes
    -----
    At the photon sphere:
      - Tangential coordinate speed: c_t = c / sqrt(3) ~ 0.577 c
      - Radial coordinate speed:     c_r = c / 3       ~ 0.333 c
    """
    return 1.5 * schwarzschild_radius(M)


def gravity_factor_radial(r: float, M: float) -> float:
    """
    Compute the radial gravity factor alpha_r(r).

    The radial gravity factor is the ratio of the coordinate speed of
    radially-propagating light to c, as measured by a distant observer:

        alpha_r(r) = 1 - r_s / r = 1 - 2GM / (r * c^2)

    This follows from setting ds^2 = 0, dOmega = 0 in the Schwarzschild
    metric and solving for |dr/dt|.

    Parameters
    ----------
    r : float
        Radial coordinate [m]. Must be > r_s for the result to be meaningful.
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Dimensionless radial gravity factor in [0, 1].
        - alpha_r = 1 at r -> infinity (flat spacetime)
        - alpha_r = 0 at r = r_s (event horizon)
        - alpha_r < 0 inside the horizon (coordinate artifact)

    Raises
    ------
    ValueError
        If r <= 0.
    """
    if r <= 0:
        raise ValueError(f"Radial coordinate must be positive, got r={r}")
    r_s = schwarzschild_radius(M)
    return 1.0 - r_s / r


def gravity_factor_tangential(r: float, M: float) -> float:
    """
    Compute the tangential gravity factor alpha_t(r).

    The tangential gravity factor is the ratio of the coordinate speed of
    tangentially-propagating light to c:

        alpha_t(r) = sqrt(1 - r_s / r) = sqrt(1 - 2GM / (r * c^2))

    This follows from setting ds^2 = 0, dr = 0 in the Schwarzschild metric
    and solving for r |dOmega/dt|.

    Note that alpha_t > alpha_r for all r > r_s (the tangential speed is
    always greater than the radial speed in Schwarzschild coordinates).

    Parameters
    ----------
    r : float
        Radial coordinate [m]. Must be > r_s.
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Dimensionless tangential gravity factor in [0, 1].

    Raises
    ------
    ValueError
        If r <= 0 or if r < r_s (would give imaginary result).
    """
    if r <= 0:
        raise ValueError(f"Radial coordinate must be positive, got r={r}")
    r_s = schwarzschild_radius(M)
    argument = 1.0 - r_s / r
    if argument < 0:
        raise ValueError(
            f"r={r} is inside the Schwarzschild radius r_s={r_s}. "
            "Tangential gravity factor is not real-valued inside the horizon."
        )
    return math.sqrt(argument)


def gravity_factor(r: float, M: float, theta: float) -> float:
    """
    Compute the unified gravity factor alpha(r, theta) for arbitrary
    propagation direction.

    For a photon propagating at angle theta relative to the radial
    direction, the coordinate speed is:

        c_eff = c * alpha(r, theta)

    where:

        alpha(r, theta) = (1 - r_s/r) / sqrt(cos^2(theta) + (1 - r_s/r) sin^2(theta))

    This is derived from the full null geodesic condition ds^2 = 0 in
    Schwarzschild coordinates with the photon's velocity decomposed into
    radial and tangential components.

    Parameters
    ----------
    r : float
        Radial coordinate [m]. Must be > r_s.
    M : float
        Mass of the gravitating body [kg].
    theta : float
        Angle between propagation direction and radial direction [radians].
        theta = 0   -> purely radial propagation
        theta = pi/2 -> purely tangential propagation

    Returns
    -------
    float
        Dimensionless gravity factor alpha(r, theta).

    Notes
    -----
    Limiting cases:
      - theta = 0:    alpha = 1 - r_s/r       (radial)
      - theta = pi/2: alpha = sqrt(1 - r_s/r)  (tangential)
      - r >> r_s:     alpha -> 1                (flat spacetime)
    """
    if r <= 0:
        raise ValueError(f"Radial coordinate must be positive, got r={r}")
    r_s = schwarzschild_radius(M)
    f = 1.0 - r_s / r
    if f < 0:
        raise ValueError(
            f"r={r} is inside the Schwarzschild radius r_s={r_s}."
        )
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    denominator = math.sqrt(cos_theta**2 + f * sin_theta**2)
    if denominator == 0:
        # This happens at the horizon (f=0) for radial propagation
        return 0.0
    return f / denominator


def effective_light_speed(r: float, M: float, theta: float = 0.0) -> float:
    """
    Compute the effective coordinate speed of light at position r,
    for propagation at angle theta to the radial direction.

        c_eff(r, theta) = c * alpha(r, theta)

    This is the speed of light as measured using Schwarzschild coordinate
    time and Schwarzschild radial distance, by an observer at infinity.

    Parameters
    ----------
    r : float
        Radial coordinate [m]. Must be > r_s.
    M : float
        Mass of the gravitating body [kg].
    theta : float, optional
        Angle between propagation direction and radial direction [radians].
        Default is 0 (radial propagation).

    Returns
    -------
    float
        Effective coordinate speed of light [m/s].

    Examples
    --------
    >>> effective_light_speed(R_SUN, M_SUN, theta=0)  # radial, at solar surface
    299791187.7...  # very close to c, since r_s/R_sun ~ 4e-6

    >>> effective_light_speed(schwarzschild_radius(M_SUN), M_SUN, theta=0)
    0.0  # zero at the event horizon
    """
    return C * gravity_factor(r, M, theta)


# ===========================================================================
# Observable Predictions
# ===========================================================================

def gravitational_redshift(r_emit: float, r_recv: float, M: float) -> float:
    """
    Compute the gravitational redshift ratio f_received / f_emitted.

    A photon emitted at r_emit and received at r_recv experiences a
    frequency shift due to the gravitational potential difference:

        f_recv / f_emit = sqrt( alpha_t(r_emit) / alpha_t(r_recv) )
                        = sqrt( (1 - r_s/r_emit) / (1 - r_s/r_recv) )

    If f_recv/f_emit < 1, the photon is redshifted (climbing out of
    a potential well). If > 1, it is blueshifted (falling in).

    Parameters
    ----------
    r_emit : float
        Radial coordinate of emission [m].
    r_recv : float
        Radial coordinate of reception [m].
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Frequency ratio f_received / f_emitted (dimensionless).

    Examples
    --------
    Photon emitted at Earth's surface, received at infinity:
    >>> gravitational_redshift(R_EARTH, 1e20, M_EARTH)
    0.999999999...  # very slightly redshifted

    Photon emitted near a black hole at r = 3*r_s, received at infinity:
    >>> rs = schwarzschild_radius(M_SUN)
    >>> gravitational_redshift(3 * rs, 1e20, M_SUN)
    0.8164...  # sqrt(2/3) -- significant redshift

    Notes
    -----
    This is exact for the Schwarzschild metric. The weak-field
    approximation gives:
        Delta_f / f ~ -g * Delta_h / c^2
    where g is local gravitational acceleration and Delta_h is the
    height difference.

    Experimental verification:
      - Pound-Rebka (1960): 1% precision
      - Gravity Probe A (1976): 0.007% precision
      - Modern atomic clocks: sub-meter height resolution
    """
    alpha_emit = gravity_factor_tangential(r_emit, M)
    alpha_recv = gravity_factor_tangential(r_recv, M)
    return alpha_emit / alpha_recv


def shapiro_delay(r1: float, r2: float, b: float, M: float) -> float:
    """
    Compute the Shapiro time delay for a light signal.

    The Shapiro delay is the excess travel time of a light signal passing
    through a gravitational field, compared to what would be expected in
    flat spacetime. This is a direct consequence of the reduced coordinate
    speed of light near a mass.

    For a signal traveling from distance r1 to distance r2 from a mass M,
    with closest approach distance (impact parameter) b:

        Delta_t = (2GM/c^3) * [ ln((r1 + sqrt(r1^2 - b^2)) *
                                    (r2 + sqrt(r2^2 - b^2)) / b^2)
                                 + (1 - b/r1)(approximate) + ... ]

    The simplified formula used here (valid for r1, r2 >> b) is:

        Delta_t = (2GM/c^3) * ln(4 * r1 * r2 / b^2)

    For a round trip, multiply by 2.

    Parameters
    ----------
    r1 : float
        Distance from the mass to the signal source (or reflector) [m].
    r2 : float
        Distance from the mass to the observer [m].
    b : float
        Impact parameter (closest approach distance) [m]. Must be > r_s.
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        One-way Shapiro delay [seconds].

    Examples
    --------
    Light grazing the Sun, Earth to Mars (superior conjunction):
    >>> shapiro_delay(1.5 * AU, AU, R_SUN, M_SUN)
    0.000120...  # ~120 microseconds one-way

    Round-trip delay (multiply by 2):
    >>> 2 * shapiro_delay(1.5 * AU, AU, R_SUN, M_SUN)
    0.000240...  # ~240 microseconds round-trip

    Notes
    -----
    The more precise formula includes corrections for the geometry:

        Delta_t = (1 + gamma) * GM/c^3 * ln( (r1 + x1)(r2 + x2) / b^2 )

    where x1, x2 are projections along the line of sight and gamma is
    the PPN parameter (gamma = 1 for GR).

    The Cassini spacecraft measurement (Bertotti et al., 2003) confirmed
    gamma = 1.000021 +/- 0.000023, the most precise test of GR.
    """
    if b <= 0:
        raise ValueError(f"Impact parameter must be positive, got b={b}")
    r_s = schwarzschild_radius(M)
    if b <= r_s:
        raise ValueError(
            f"Impact parameter b={b} must be greater than the "
            f"Schwarzschild radius r_s={r_s} for the signal to pass by."
        )

    # Prefactor: 2GM/c^3 (has units of time)
    prefactor = 2 * G * M / C**3

    # For the standard Shapiro formula, we compute using the exact
    # integral result. When r1, r2 >> b, the dominant term is:
    #   Delta_t = (2GM/c^3) * ln(4*r1*r2 / b^2)
    #
    # For better accuracy, we use the full expression:
    #   Delta_t = (2GM/c^3) * [ ln( (r1 + sqrt(r1^2 - b^2)) *
    #                               (r2 + sqrt(r2^2 - b^2)) / b^2 ) ]
    # This is valid as long as r1, r2 >= b.

    if r1 < b or r2 < b:
        # Use the simpler far-field formula with absolute value protection
        argument = 4.0 * r1 * r2 / b**2
        if argument <= 0:
            raise ValueError("Invalid geometry: logarithm argument non-positive.")
        return prefactor * math.log(argument)

    # Full formula
    x1 = math.sqrt(r1**2 - b**2)
    x2 = math.sqrt(r2**2 - b**2)
    argument = (r1 + x1) * (r2 + x2) / b**2
    return prefactor * math.log(argument)


def deflection_angle(b: float, M: float) -> float:
    """
    Compute the gravitational deflection angle for light passing a mass M
    with impact parameter b.

    In General Relativity, the deflection angle for a photon passing a
    spherically symmetric mass is:

        delta = 4GM / (b * c^2) = 2 * r_s / b

    This is twice the Newtonian prediction (which only accounts for the
    temporal part of the metric). The full GR result includes the spatial
    curvature contribution, which doubles the deflection.

    In the gravity factor framework, this factor of 2 arises from the
    anisotropy between radial and tangential coordinate speeds of light.
    Using Fermat's principle with the variable refractive index
    n(r) ~ 1 + 2GM/(rc^2), the deflection integral yields the full result.

    Parameters
    ----------
    b : float
        Impact parameter (closest approach distance) [m]. Must be > r_s.
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Deflection angle [radians].

    Examples
    --------
    Light grazing the Sun:
    >>> import math
    >>> delta = deflection_angle(R_SUN, M_SUN)
    >>> delta_arcsec = math.degrees(delta) * 3600
    >>> print(f"{delta_arcsec:.2f} arcseconds")
    1.75 arcseconds

    Notes
    -----
    This formula is the leading-order (weak-field) result. Higher-order
    corrections are:

        delta = 4GM/(bc^2) + (15*pi/4)(GM/(bc^2))^2 + ...

    For the Sun, the second-order term is ~3.5 microarcseconds, far below
    current measurement precision for solar deflection.

    In the PPN formalism:
        delta = (1 + gamma)/2 * 4GM/(bc^2)
    with gamma = 1 for GR.

    Historical significance:
      - Einstein (1911) predicted delta = 2GM/(bc^2) = 0.875" (Newtonian value)
      - Einstein (1915) corrected to delta = 4GM/(bc^2) = 1.75" (full GR)
      - Eddington (1919) measured ~1.75", confirming GR
    """
    if b <= 0:
        raise ValueError(f"Impact parameter must be positive, got b={b}")
    r_s = schwarzschild_radius(M)
    if b <= r_s:
        raise ValueError(
            f"Impact parameter b={b} is less than Schwarzschild radius "
            f"r_s={r_s}. Light would be captured, not deflected."
        )
    return 4 * G * M / (b * C**2)


def refractive_index(r: float, M: float, theta: float = 0.0) -> float:
    """
    Compute the effective gravitational refractive index.

    Gravity can be described as a medium with variable refractive index:

        n(r, theta) = c / c_eff(r, theta) = 1 / alpha(r, theta)

    This is the exact analog of an optical refractive index. Light bends
    toward regions of higher n (closer to the mass), just as in a
    gradient-index (GRIN) lens.

    Parameters
    ----------
    r : float
        Radial coordinate [m]. Must be > r_s.
    M : float
        Mass of the gravitating body [kg].
    theta : float, optional
        Angle between propagation direction and radial direction [radians].
        Default is 0 (radial).

    Returns
    -------
    float
        Effective refractive index (dimensionless, >= 1).

    Notes
    -----
    Properties of the gravitational refractive index:
      - n = 1 at r -> infinity (vacuum)
      - n -> infinity as r -> r_s (event horizon: infinite refractive index)
      - n > 1 everywhere outside the horizon (light is always slower)
      - Achromatic: n does not depend on photon frequency

    Special cases:
      - Radial:     n_r = 1 / (1 - r_s/r) = r / (r - r_s)
      - Tangential:  n_t = 1 / sqrt(1 - r_s/r) = sqrt(r / (r - r_s))

    At Earth's surface: n ~ 1 + 1.4e-9 (extremely close to unity)
    At the Sun's surface: n ~ 1 + 4.2e-6
    At the photon sphere: n_t = sqrt(3) ~ 1.732
    """
    alpha = gravity_factor(r, M, theta)
    if alpha <= 0:
        raise ValueError(
            "Gravity factor is zero or negative (at or inside horizon). "
            "Refractive index is undefined (infinite)."
        )
    return 1.0 / alpha


# ===========================================================================
# Additional Derived Quantities
# ===========================================================================

def orbital_precession(a: float, e: float, M: float) -> float:
    """
    Compute the relativistic orbital precession per orbit.

    For a body orbiting a mass M in an elliptical orbit with semi-major
    axis a and eccentricity e, General Relativity predicts an advance
    of the perihelion:

        delta_phi = 6 * pi * G * M / (a * c^2 * (1 - e^2))   [radians/orbit]

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit [m].
    e : float
        Orbital eccentricity (0 <= e < 1).
    M : float
        Mass of the central body [kg].

    Returns
    -------
    float
        Precession per orbit [radians].

    Examples
    --------
    Mercury's precession:
    >>> a_mercury = 5.791e10  # semi-major axis [m]
    >>> e_mercury = 0.2056
    >>> delta = orbital_precession(a_mercury, e_mercury, M_SUN)
    >>> # Convert to arcseconds per century
    >>> orbits_per_century = 100 * 365.25 / 87.969  # Mercury's period = 87.969 days
    >>> arcsec = math.degrees(delta) * 3600 * orbits_per_century
    >>> print(f"{arcsec:.2f} arcsec/century")
    42.98 arcsec/century

    Notes
    -----
    In the PPN formalism:
        delta_phi = (2 + 2*gamma - beta) / 3 * 6*pi*GM / (a*c^2*(1-e^2))
    For GR (gamma = beta = 1): (2+2-1)/3 = 1, recovering the standard result.

    Connection to the gravity factor: The precession arises from the
    anisotropy of the effective metric (different radial and tangential
    "speeds"), which modifies the relationship between radial and angular
    orbital dynamics compared to Newtonian gravity.
    """
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a}")
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must be in [0, 1), got e={e}")
    return 6 * math.pi * G * M / (a * C**2 * (1 - e**2))


def weak_field_gravity_factor(r: float, M: float) -> float:
    """
    Compute the isotropic weak-field gravity factor.

    In the weak-field limit (r >> r_s), the gravity factor can be
    approximated as isotropic:

        alpha(r) ~ 1 + 2*Phi/c^2 = 1 - 2GM/(r*c^2)

    where Phi = -GM/r is the Newtonian gravitational potential.

    This is the first-order PPN approximation with gamma = 1.

    Parameters
    ----------
    r : float
        Distance from the mass [m].
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Weak-field gravity factor (dimensionless).

    Notes
    -----
    This approximation is valid when r >> r_s, i.e., 2GM/(rc^2) << 1.
    For the Sun at its surface, 2GM/(Rc^2) ~ 4e-6, so the weak-field
    approximation is excellent throughout the solar system.
    """
    if r <= 0:
        raise ValueError(f"Distance must be positive, got r={r}")
    Phi = -G * M / r  # Newtonian potential (negative)
    return 1.0 + 2.0 * Phi / C**2


def isco_radius(M: float) -> float:
    """
    Compute the Innermost Stable Circular Orbit (ISCO) radius.

    For a Schwarzschild (non-spinning) black hole, the ISCO is at:

        r_isco = 3 * r_s = 6GM / c^2

    No stable circular orbits for massive particles exist inside this radius.

    Parameters
    ----------
    M : float
        Mass of the black hole [kg].

    Returns
    -------
    float
        ISCO radius [m].
    """
    return 3.0 * schwarzschild_radius(M)


def time_dilation_factor(r: float, M: float) -> float:
    """
    Compute the gravitational time dilation factor.

    A clock at radius r in a Schwarzschild field ticks slower than a
    clock at infinity by the factor:

        d_tau / dt = sqrt(1 - r_s/r) = alpha_t(r)

    This is identical to the tangential gravity factor.

    Parameters
    ----------
    r : float
        Radial coordinate [m]. Must be > r_s.
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    float
        Time dilation factor (0 at horizon, 1 at infinity).

    Examples
    --------
    GPS satellites (altitude ~20,200 km):
    >>> r_gps = R_EARTH + 20200e3
    >>> factor = time_dilation_factor(r_gps, M_EARTH)
    >>> # Difference from surface clock:
    >>> factor_surface = time_dilation_factor(R_EARTH, M_EARTH)
    >>> drift_per_day = (factor - factor_surface) * 86400  # seconds
    >>> print(f"GPS clock gains ~{drift_per_day*1e6:.1f} microseconds/day vs ground")
    GPS clock gains ~45.7 microseconds/day vs ground
    """
    return gravity_factor_tangential(r, M)


def coordinate_speed_ratio(r: float, M: float) -> dict:
    """
    Compute a summary of coordinate speed ratios at radius r.

    Returns a dictionary with radial, tangential, and isotropic (weak-field)
    gravity factors, plus the anisotropy ratio.

    Parameters
    ----------
    r : float
        Radial coordinate [m].
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    dict
        Dictionary with keys:
        - 'r': the input radius [m]
        - 'r_over_rs': ratio r / r_s
        - 'alpha_r': radial gravity factor
        - 'alpha_t': tangential gravity factor
        - 'alpha_weak': weak-field isotropic approximation
        - 'anisotropy': ratio alpha_r / alpha_t
        - 'c_r': radial coordinate speed [m/s]
        - 'c_t': tangential coordinate speed [m/s]
        - 'n_r': radial refractive index
        - 'n_t': tangential refractive index
    """
    r_s = schwarzschild_radius(M)
    alpha_r = gravity_factor_radial(r, M)
    alpha_t = gravity_factor_tangential(r, M)
    alpha_w = weak_field_gravity_factor(r, M)

    return {
        'r': r,
        'r_over_rs': r / r_s if r_s > 0 else float('inf'),
        'alpha_r': alpha_r,
        'alpha_t': alpha_t,
        'alpha_weak': alpha_w,
        'anisotropy': alpha_r / alpha_t if alpha_t > 0 else 0.0,
        'c_r': C * alpha_r,
        'c_t': C * alpha_t,
        'n_r': 1.0 / alpha_r if alpha_r > 0 else float('inf'),
        'n_t': 1.0 / alpha_t if alpha_t > 0 else float('inf'),
    }


# ===========================================================================
# Verification / Self-Test
# ===========================================================================

def _verify_all() -> None:
    """
    Run self-consistency checks on all formulas.

    This verifies:
    1. Schwarzschild radius of the Sun ~ 2953 m
    2. Photon sphere = 1.5 * r_s
    3. Gravity factors reduce to correct limits
    4. Redshift formula is self-consistent
    5. Deflection angle for the Sun ~ 1.75 arcsec
    6. Shapiro delay for Sun ~ 120 microseconds (one-way, grazing)
    7. Mercury precession ~ 42.98 arcsec/century
    """
    print("=" * 60)
    print("GRAVITY FACTOR FRAMEWORK: SELF-VERIFICATION")
    print("=" * 60)

    # 1. Schwarzschild radius
    r_s = schwarzschild_radius(M_SUN)
    print(f"\n1. Schwarzschild radius of the Sun:")
    print(f"   r_s = {r_s:.2f} m (expected ~2953 m)")
    assert 2950 < r_s < 2960, f"r_s out of range: {r_s}"

    # 2. Photon sphere
    r_ps = photon_sphere_radius(M_SUN)
    print(f"\n2. Photon sphere radius:")
    print(f"   r_ps = {r_ps:.2f} m = 1.5 * r_s = {1.5 * r_s:.2f} m")
    assert abs(r_ps - 1.5 * r_s) < 1e-6

    # 3. Gravity factors at known points
    print(f"\n3. Gravity factors at solar surface (r = R_sun):")
    alpha_r = gravity_factor_radial(R_SUN, M_SUN)
    alpha_t = gravity_factor_tangential(R_SUN, M_SUN)
    print(f"   alpha_r = {alpha_r:.10f}")
    print(f"   alpha_t = {alpha_t:.10f}")
    print(f"   (1 - alpha_r) = {1 - alpha_r:.2e} (should be ~4.24e-6)")
    assert abs((1 - alpha_r) - r_s / R_SUN) < 1e-15

    # Verify limiting cases
    alpha_rad = gravity_factor(R_SUN, M_SUN, 0)
    alpha_tan = gravity_factor(R_SUN, M_SUN, math.pi / 2)
    assert abs(alpha_rad - alpha_r) < 1e-12, "Radial limit mismatch"
    assert abs(alpha_tan - alpha_t) < 1e-12, "Tangential limit mismatch"
    print("   Unified formula matches radial and tangential limits: OK")

    # 4. Redshift
    print(f"\n4. Gravitational redshift (surface to infinity):")
    z_ratio = gravitational_redshift(R_SUN, 1e20, M_SUN)
    z = 1 - z_ratio
    print(f"   f_recv / f_emit = {z_ratio:.10f}")
    print(f"   Fractional redshift z = {z:.2e} (expected ~2.12e-6)")

    # 5. Deflection angle
    print(f"\n5. Light deflection by the Sun (grazing):")
    delta = deflection_angle(R_SUN, M_SUN)
    delta_arcsec = math.degrees(delta) * 3600
    print(f"   delta = {delta:.6e} rad = {delta_arcsec:.2f} arcseconds")
    print(f"   Expected: 1.75 arcseconds")
    assert 1.74 < delta_arcsec < 1.76, f"Deflection out of range: {delta_arcsec}"

    # 6. Shapiro delay
    print(f"\n6. Shapiro time delay (Earth-Sun-Mars, grazing):")
    r_earth = AU
    r_mars = 1.524 * AU
    dt = shapiro_delay(r_mars, r_earth, R_SUN, M_SUN)
    print(f"   One-way delay = {dt * 1e6:.1f} microseconds")
    print(f"   Round-trip    = {2 * dt * 1e6:.1f} microseconds")
    print(f"   Expected: ~120 us one-way, ~240 us round-trip")
    # Allow some margin since exact value depends on geometry
    assert 100e-6 < dt < 140e-6, f"Shapiro delay out of range: {dt}"

    # 7. Mercury precession
    print(f"\n7. Mercury orbital precession:")
    a_mercury = 5.791e10  # semi-major axis [m]
    e_mercury = 0.2056
    delta_phi = orbital_precession(a_mercury, e_mercury, M_SUN)
    period_mercury = 87.969  # days
    orbits_per_century = 100 * 365.25 / period_mercury
    arcsec_per_century = math.degrees(delta_phi) * 3600 * orbits_per_century
    print(f"   Precession per orbit = {delta_phi:.6e} rad")
    print(f"   Orbits per century   = {orbits_per_century:.1f}")
    print(f"   Precession per century = {arcsec_per_century:.2f} arcsec")
    print(f"   Expected: 42.98 arcsec/century")
    assert 42.5 < arcsec_per_century < 43.5, \
        f"Precession out of range: {arcsec_per_century}"

    # 8. Refractive index
    print(f"\n8. Refractive index at the photon sphere:")
    n_t = refractive_index(r_ps, M_SUN, math.pi / 2)
    print(f"   n_t(r_ps) = {n_t:.6f} (expected sqrt(3) = {math.sqrt(3):.6f})")
    assert abs(n_t - math.sqrt(3)) < 1e-6

    # 9. PPN consistency
    print(f"\n9. PPN consistency check (gamma = 1):")
    print(f"   Cassini confirmed gamma = 1.000021 +/- 0.000023")
    print(f"   Our framework uses gamma = 1 (exact GR)")
    r_test = 10 * R_SUN
    alpha_r_exact = gravity_factor_radial(r_test, M_SUN)
    alpha_r_weak = weak_field_gravity_factor(r_test, M_SUN)
    print(f"   At r = 10 R_sun:")
    print(f"     Exact alpha_r = {alpha_r_exact:.12f}")
    print(f"     Weak-field    = {alpha_r_weak:.12f}")
    print(f"     Difference    = {abs(alpha_r_exact - alpha_r_weak):.2e}")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _verify_all()
