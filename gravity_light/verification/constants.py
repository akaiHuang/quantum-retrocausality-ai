"""
Physical constants and solar system data for gravity factor verification.

All values are in SI units unless otherwise noted.
Sources: NIST CODATA 2018, IAU 2015 nominal values.
"""

import numpy as np

# =============================================================================
# Fundamental Constants
# =============================================================================

G = 6.67430e-11          # Gravitational constant [m^3 kg^-1 s^-2]
c = 299792458.0           # Speed of light in vacuum [m/s]
c2 = c * c               # c squared [m^2/s^2]
h = 6.62607015e-34        # Planck constant [J s]
hbar = h / (2 * np.pi)   # Reduced Planck constant [J s]
k_B = 1.380649e-23        # Boltzmann constant [J/K]

# =============================================================================
# Solar System Bodies
# =============================================================================

# Sun
M_sun = 1.989e30          # Solar mass [kg]
R_sun = 6.957e8           # Solar radius [m]
rs_sun = 2 * G * M_sun / c2  # Schwarzschild radius of Sun [m] ~ 2953 m

# Earth
M_earth = 5.972e24        # Earth mass [kg]
R_earth = 6.371e6         # Earth radius [m]
rs_earth = 2 * G * M_earth / c2  # Schwarzschild radius of Earth [m] ~ 8.87 mm

# Moon
M_moon = 7.342e22         # Moon mass [kg]
R_moon = 1.7374e6         # Moon radius [m]

# Jupiter
M_jupiter = 1.898e27      # Jupiter mass [kg]
R_jupiter = 6.9911e7      # Jupiter radius [m]

# =============================================================================
# Compact Objects
# =============================================================================

# Typical neutron star
M_ns = 1.4 * M_sun        # Neutron star mass [kg]
R_ns = 1.0e4               # Neutron star radius [m] (10 km)
rs_ns = 2 * G * M_ns / c2  # Schwarzschild radius of neutron star [m]

# Sagittarius A* (Milky Way central black hole)
M_sgra = 4.0e6 * M_sun    # Sgr A* mass [kg]
rs_sgra = 2 * G * M_sgra / c2  # Schwarzschild radius of Sgr A* [m]

# =============================================================================
# Orbital Parameters
# =============================================================================

# GPS satellite orbit
r_gps = 26_560_000.0       # GPS orbital radius [m] (from Earth center)
GPS_altitude = r_gps - R_earth  # GPS altitude above surface [m] ~ 20,189 km

# Mercury orbit
a_mercury = 5.791e10       # Mercury semi-major axis [m]
e_mercury = 0.2056         # Mercury orbital eccentricity
P_mercury = 87.969 * 86400 # Mercury orbital period [s]

# Earth-Sun distance
AU = 1.496e11              # Astronomical unit [m]

# =============================================================================
# Observational Reference Values
# =============================================================================

# Solar gravitational redshift (fractional wavelength shift)
solar_redshift_observed = 2.12e-6  # Dlambda/lambda = GM_sun/(R_sun * c^2)

# Solar limb deflection of light (General Relativity prediction, confirmed)
solar_deflection_arcsec = 1.75  # arcseconds

# Shapiro delay for signal grazing the Sun
shapiro_delay_sun_us = 240.0  # microseconds (approximate, for superior conjunction)

# Mercury perihelion precession due to GR
mercury_precession_arcsec_century = 42.98  # arcseconds per century

# Cassini PPN gamma measurement (2003)
cassini_gamma = 1.000021
cassini_gamma_uncertainty = 0.000023

# GPS gravitational time dilation: clocks run faster by ~45.85 us/day
# compared to ground clocks (gravitational effect only, before SR correction)
gps_gravitational_shift_us_per_day = 45.85  # microseconds per day

# Pound-Rebka experiment (1959): measured gravitational redshift
# over height h = 22.5 m at Earth surface
pound_rebka_height = 22.5  # meters
pound_rebka_fractional_shift = 2.46e-15  # Dlambda/lambda measured

# =============================================================================
# Derived Quantities
# =============================================================================

def schwarzschild_radius(M):
    """Compute Schwarzschild radius for a given mass.

    Parameters
    ----------
    M : float
        Mass in kg.

    Returns
    -------
    float
        Schwarzschild radius in meters.
    """
    return 2 * G * M / c2


def gravitational_potential(M, r):
    """Newtonian gravitational potential at distance r from mass M.

    Parameters
    ----------
    M : float
        Central mass in kg.
    r : float or ndarray
        Distance from center of mass in meters.

    Returns
    -------
    float or ndarray
        Gravitational potential Phi = -GM/r in m^2/s^2.
    """
    return -G * M / r


def c_radial(r, rs):
    """Coordinate speed of light for radial propagation in Schwarzschild metric.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate in meters.
    rs : float
        Schwarzschild radius in meters.

    Returns
    -------
    float or ndarray
        Radial coordinate speed of light c_r = c * (1 - rs/r).
    """
    return c * (1.0 - rs / r)


def c_tangential(r, rs):
    """Coordinate speed of light for tangential propagation in Schwarzschild metric.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate in meters.
    rs : float
        Schwarzschild radius in meters.

    Returns
    -------
    float or ndarray
        Tangential coordinate speed of light c_t = c * sqrt(1 - rs/r).
    """
    return c * np.sqrt(1.0 - rs / r)


def c_effective(r, rs, theta=0.0):
    """Effective coordinate speed of light at angle theta to radial direction.

    In the Schwarzschild metric, the coordinate speed of light depends on
    both the radial position and the propagation direction:
        c_eff(r, theta) = c * sqrt[(1 - rs/r)^2 cos^2(theta) + (1 - rs/r) sin^2(theta)]

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate in meters.
    rs : float
        Schwarzschild radius in meters.
    theta : float or ndarray, optional
        Angle between propagation direction and radial direction in radians.
        theta=0 is radial, theta=pi/2 is tangential. Default is 0 (radial).

    Returns
    -------
    float or ndarray
        Effective coordinate speed of light in m/s.
    """
    factor = 1.0 - rs / r
    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    return c * np.sqrt(factor ** 2 * cos2 + factor * sin2)


def refractive_index(r, rs, theta=0.0):
    """Effective refractive index of space in a gravitational field.

    n(r, theta) = c / c_eff(r, theta)

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate in meters.
    rs : float
        Schwarzschild radius in meters.
    theta : float or ndarray, optional
        Propagation angle relative to radial direction. Default is 0.

    Returns
    -------
    float or ndarray
        Refractive index (dimensionless, >= 1).
    """
    return c / c_effective(r, rs, theta)


def gravitational_redshift_ratio(r_emit, r_recv, rs):
    """Frequency ratio for gravitational redshift between two radii.

    f_recv / f_emit = sqrt((1 - rs/r_emit) / (1 - rs/r_recv))

    Parameters
    ----------
    r_emit : float
        Emission radius in meters.
    r_recv : float
        Reception radius in meters.
    rs : float
        Schwarzschild radius in meters.

    Returns
    -------
    float
        Frequency ratio f_recv/f_emit.
    """
    return np.sqrt((1.0 - rs / r_emit) / (1.0 - rs / r_recv))


def deflection_angle(M, b):
    """Gravitational deflection angle for light with impact parameter b.

    In the weak field limit: delta = 4GM / (b * c^2) radians.

    Parameters
    ----------
    M : float
        Mass of deflecting body in kg.
    b : float
        Impact parameter (closest approach distance) in meters.

    Returns
    -------
    float
        Deflection angle in radians.
    """
    return 4.0 * G * M / (b * c2)


def shapiro_delay(M, r_emit, r_recv, b):
    """Shapiro time delay for a signal passing near a massive body.

    The extra round-trip time delay is:
        dt = (4GM/c^3) * [ln(4*r_emit*r_recv / b^2) + 1]

    This is the one-way excess delay (multiply by 2 for round-trip).

    Parameters
    ----------
    M : float
        Mass of the body in kg.
    r_emit : float
        Distance of emitter from the massive body in meters.
    r_recv : float
        Distance of receiver from the massive body in meters.
    b : float
        Impact parameter (closest approach to body) in meters.

    Returns
    -------
    float
        One-way Shapiro time delay in seconds.
    """
    return (4.0 * G * M / c ** 3) * (np.log(4.0 * r_emit * r_recv / b ** 2) + 1.0)


def photon_sphere_radius(rs):
    """Radius of the photon sphere (circular light orbits).

    r_photon = 1.5 * r_s = 3GM/c^2

    Parameters
    ----------
    rs : float
        Schwarzschild radius in meters.

    Returns
    -------
    float
        Photon sphere radius in meters.
    """
    return 1.5 * rs


def weak_field_c_eff(M, r):
    """Effective speed of light in the weak field approximation.

    c_eff ~ c * (1 + 2*Phi/c^2) where Phi = -GM/r

    Valid when rs << r.

    Parameters
    ----------
    M : float
        Central mass in kg.
    r : float or ndarray
        Distance from center of mass in meters.

    Returns
    -------
    float or ndarray
        Approximate effective speed of light in m/s.
    """
    Phi = gravitational_potential(M, r)
    return c * (1.0 + 2.0 * Phi / c2)
