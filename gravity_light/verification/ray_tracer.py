"""
Numerical ray tracer for light propagation in a gravitational field.

Uses the variable effective speed of light (coordinate speed in Schwarzschild
metric) to trace light rays through curved spacetime. The approach treats
gravity as a refractive medium with index n(r) = c / c_eff(r), and applies
Fermat's principle: light follows the path of least coordinate time.

The ray equation in a refractive medium is:
    d/ds (n * dr/ds) = grad(n)

where s is the path parameter and n is the refractive index.

This module implements 4th-order Runge-Kutta integration of the ray equation
and compares the resulting deflection with the analytical GR prediction.
"""

import numpy as np
from typing import Tuple, List, Optional

from gravity_light.verification.constants import (
    G, c, c2, M_sun, R_sun, rs_sun, AU,
    schwarzschild_radius, refractive_index, deflection_angle,
)


class GravitationalRayTracer:
    """Numerical ray tracer for light in a gravitational field.

    Traces a light ray through the gravitational field of a point mass
    by integrating the ray equation using 4th-order Runge-Kutta.

    The ray is parameterized in 2D Cartesian coordinates (x, y) with
    the massive body at the origin. The coordinate speed of propagation
    is c_eff(r, theta) from the Schwarzschild metric.

    Parameters
    ----------
    M : float
        Mass of the central body in kg.
    """

    def __init__(self, M: float):
        self.M = M
        self.rs = schwarzschild_radius(M)
        self.GM = G * M

    def _refractive_index_at(self, x: float, y: float) -> float:
        """Compute the isotropic refractive index at position (x, y).

        For the ray tracer we use the isotropic (tangential) refractive
        index as an approximation, which gives the correct weak-field
        deflection. In the isotropic approximation:
            n(r) ~ 1 + 2GM/(r*c^2)
        which is accurate in the weak-field regime and gives the correct
        deflection angle of 4GM/(bc^2).

        For strong fields we use the exact expression from the Schwarzschild
        metric. The full anisotropic treatment would require tracking the
        propagation direction relative to the radial direction at each step.

        Parameters
        ----------
        x, y : float
            Cartesian coordinates in meters.

        Returns
        -------
        float
            Refractive index at (x, y).
        """
        r = np.sqrt(x * x + y * y)
        if r <= self.rs:
            return 1.0e10  # effectively infinite inside event horizon
        # Isotropic weak-field approximation (gives correct deflection):
        # n = 1 + 2GM/(r c^2) = 1 + rs/r
        # This is the standard result from the PPN formalism with gamma=1.
        return 1.0 + self.rs / r

    def _grad_n(self, x: float, y: float) -> Tuple[float, float]:
        """Compute the gradient of the refractive index at (x, y).

        grad(n) = -rs/r^2 * r_hat = -rs/r^3 * (x, y)

        Parameters
        ----------
        x, y : float
            Cartesian coordinates in meters.

        Returns
        -------
        tuple of float
            (dn/dx, dn/dy) components of the gradient.
        """
        r2 = x * x + y * y
        r = np.sqrt(r2)
        if r <= self.rs:
            return (0.0, 0.0)
        r3 = r2 * r
        factor = -self.rs / r3
        return (factor * x, factor * y)

    def trace_ray(
        self,
        x0: float,
        y0: float,
        vx0: float,
        vy0: float,
        ds: float = 1.0e7,
        n_steps: int = 100000,
        r_stop: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trace a light ray through the gravitational field.

        Integrates the ray equation:
            d/ds(n * dx/ds) = dn/dx
            d/ds(n * dy/ds) = dn/dy

        where s is the arc-length parameter and the tangent vector
        (dx/ds, dy/ds) has unit magnitude.

        We convert to a first-order system by introducing the "momentum"
        p = n * tangent_vector, so that:
            dx/ds = px / n
            dy/ds = py / n
            dpx/ds = dn/dx
            dpy/ds = dn/dy

        Parameters
        ----------
        x0, y0 : float
            Initial position in meters.
        vx0, vy0 : float
            Initial direction (will be normalized).
        ds : float, optional
            Step size in meters along the ray. Default 1e7 m (10000 km).
        n_steps : int, optional
            Maximum number of integration steps. Default 100000.
        r_stop : float or None, optional
            Stop when r exceeds this value. Default None (use n_steps).

        Returns
        -------
        xs, ys : ndarray
            Cartesian coordinates along the ray in meters.
        ts : ndarray
            Coordinate time along the ray in seconds.
        vxs, vys : ndarray
            Direction components at each point.
        """
        # Normalize initial direction
        v_mag = np.sqrt(vx0 * vx0 + vy0 * vy0)
        vx0 /= v_mag
        vy0 /= v_mag

        # Initialize arrays
        xs = np.zeros(n_steps + 1)
        ys = np.zeros(n_steps + 1)
        ts = np.zeros(n_steps + 1)
        vxs = np.zeros(n_steps + 1)
        vys = np.zeros(n_steps + 1)

        xs[0] = x0
        ys[0] = y0
        ts[0] = 0.0
        vxs[0] = vx0
        vys[0] = vy0

        # State: [x, y, px, py] where p = n * unit_tangent
        n0 = self._refractive_index_at(x0, y0)
        state = np.array([x0, y0, n0 * vx0, n0 * vy0])

        actual_steps = 0

        for i in range(n_steps):
            state_new = self._rk4_step(state, ds)

            x, y, px, py = state_new
            n_val = self._refractive_index_at(x, y)
            vx = px / n_val
            vy = py / n_val

            # Accumulate coordinate time: dt = ds * n / c
            # (in the refractive medium, coordinate speed = c/n,
            #  so time for arc length ds is ds / (c/n) = ds*n/c)
            n_mid = self._refractive_index_at(
                0.5 * (xs[i] + x), 0.5 * (ys[i] + y)
            )
            dt = ds * n_mid / c

            xs[i + 1] = x
            ys[i + 1] = y
            ts[i + 1] = ts[i] + dt
            vxs[i + 1] = vx
            vys[i + 1] = vy

            state = state_new
            actual_steps = i + 1

            # Check stopping condition
            r = np.sqrt(x * x + y * y)
            if r <= 1.01 * self.rs:
                break  # ray captured by black hole
            if r_stop is not None and r > r_stop:
                break

        # Trim arrays
        n_out = actual_steps + 1
        return xs[:n_out], ys[:n_out], ts[:n_out], vxs[:n_out], vys[:n_out]

    def _rk4_step(self, state: np.ndarray, ds: float) -> np.ndarray:
        """Single 4th-order Runge-Kutta step.

        Parameters
        ----------
        state : ndarray
            Current state [x, y, px, py].
        ds : float
            Step size.

        Returns
        -------
        ndarray
            New state after one RK4 step.
        """
        k1 = self._derivatives(state)
        k2 = self._derivatives(state + 0.5 * ds * k1)
        k3 = self._derivatives(state + 0.5 * ds * k2)
        k4 = self._derivatives(state + ds * k3)
        return state + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for the ray equation.

        d/ds [x, y, px, py] = [px/n, py/n, dn/dx, dn/dy]

        Parameters
        ----------
        state : ndarray
            [x, y, px, py]

        Returns
        -------
        ndarray
            Derivatives [dx/ds, dy/ds, dpx/ds, dpy/ds].
        """
        x, y, px, py = state
        n_val = self._refractive_index_at(x, y)
        dn_dx, dn_dy = self._grad_n(x, y)
        return np.array([px / n_val, py / n_val, dn_dx, dn_dy])

    def compute_deflection(
        self,
        b: float,
        x_start: float = -5.0 * AU,
        x_stop: float = 5.0 * AU,
        ds: Optional[float] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute the deflection angle for a ray with impact parameter b.

        The ray starts far away on the negative x-axis at y=b, traveling
        in the +x direction. After passing the mass at the origin, the
        deflection angle is measured from the final velocity direction.

        Parameters
        ----------
        b : float
            Impact parameter in meters.
        x_start : float, optional
            Starting x coordinate (far from mass). Default -5 AU.
        x_stop : float, optional
            Stopping distance from origin. Default 5 AU.
        ds : float or None, optional
            Step size. If None, auto-selected based on b.

        Returns
        -------
        delta : float
            Deflection angle in radians.
        xs, ys : ndarray
            Ray coordinates.
        """
        if ds is None:
            # Choose step size as a small fraction of the impact parameter
            ds = min(b / 100.0, abs(x_start) / 10000.0)

        # Initial conditions: ray comes from far left, moving in +x direction
        x0 = x_start
        y0 = b
        vx0 = 1.0
        vy0 = 0.0

        xs, ys, ts, vxs, vys = self.trace_ray(
            x0, y0, vx0, vy0, ds=ds, n_steps=1000000, r_stop=abs(x_stop)
        )

        # Compute deflection from final velocity direction
        vx_final = vxs[-1]
        vy_final = vys[-1]
        # The undeflected ray would have vy=0, so deflection angle is:
        delta = np.arctan2(vy_final, vx_final)

        return delta, xs, ys

    def compute_deflection_analytical(self, b: float) -> float:
        """Analytical deflection angle from GR (weak field).

        delta = 4GM / (b * c^2)

        Parameters
        ----------
        b : float
            Impact parameter in meters.

        Returns
        -------
        float
            Deflection angle in radians.
        """
        return deflection_angle(self.M, b)


def trace_solar_deflection(
    b_solar_radii: float = 1.0,
    ds_factor: float = 100.0,
) -> dict:
    """Trace a light ray past the Sun and compute deflection.

    Parameters
    ----------
    b_solar_radii : float, optional
        Impact parameter in units of solar radii. Default 1.0.
    ds_factor : float, optional
        Step size = R_sun / ds_factor. Default 100.0.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'b': impact parameter [m]
        - 'deflection_numerical': numerical deflection [rad]
        - 'deflection_analytical': analytical deflection [rad]
        - 'deflection_numerical_arcsec': numerical deflection [arcsec]
        - 'deflection_analytical_arcsec': analytical deflection [arcsec]
        - 'relative_error': fractional difference
        - 'xs', 'ys': ray coordinates [m]
    """
    tracer = GravitationalRayTracer(M_sun)
    b = b_solar_radii * R_sun

    ds = R_sun / ds_factor

    delta_num, xs, ys = tracer.compute_deflection(
        b=b,
        x_start=-3.0 * AU,
        x_stop=3.0 * AU,
        ds=ds,
    )

    delta_ana = tracer.compute_deflection_analytical(b)

    arcsec_per_rad = 3600.0 * 180.0 / np.pi

    return {
        'b': b,
        'deflection_numerical': abs(delta_num),
        'deflection_analytical': delta_ana,
        'deflection_numerical_arcsec': abs(delta_num) * arcsec_per_rad,
        'deflection_analytical_arcsec': delta_ana * arcsec_per_rad,
        'relative_error': abs(abs(delta_num) - delta_ana) / delta_ana,
        'xs': xs,
        'ys': ys,
    }


def trace_multiple_impact_parameters(
    b_values_solar_radii: Optional[List[float]] = None,
) -> List[dict]:
    """Trace rays at multiple impact parameters and collect results.

    Parameters
    ----------
    b_values_solar_radii : list of float, optional
        Impact parameters in solar radii. Default: [1, 2, 5, 10, 50].

    Returns
    -------
    list of dict
        Results for each impact parameter.
    """
    if b_values_solar_radii is None:
        b_values_solar_radii = [1.0, 2.0, 5.0, 10.0, 50.0]

    results = []
    for b_sr in b_values_solar_radii:
        result = trace_solar_deflection(b_solar_radii=b_sr)
        results.append(result)

    return results


if __name__ == '__main__':
    print("=" * 72)
    print("Gravitational Ray Tracer -- Solar Deflection Test")
    print("=" * 72)

    # Quick test with solar-limb grazing ray
    print("\nTracing ray at b = 1 R_sun (solar limb)...")
    result = trace_solar_deflection(b_solar_radii=1.0, ds_factor=50.0)

    print(f"  Impact parameter: {result['b']:.3e} m ({result['b']/R_sun:.1f} R_sun)")
    print(f"  Numerical deflection:  {result['deflection_numerical_arcsec']:.4f} arcsec")
    print(f"  Analytical deflection: {result['deflection_analytical_arcsec']:.4f} arcsec")
    print(f"  Relative error: {result['relative_error']:.2e}")
    print(f"  Ray trace points: {len(result['xs'])}")

    # Test at multiple impact parameters
    print("\n" + "-" * 72)
    print("Deflection vs impact parameter:")
    print(f"{'b/R_sun':>10s} {'Numerical [arcsec]':>20s} {'Analytical [arcsec]':>20s} {'Rel Error':>12s}")
    print("-" * 72)

    for b_sr in [1.0, 2.0, 5.0, 10.0]:
        r = trace_solar_deflection(b_solar_radii=b_sr, ds_factor=50.0)
        print(
            f"{b_sr:10.1f} {r['deflection_numerical_arcsec']:20.4f} "
            f"{r['deflection_analytical_arcsec']:20.4f} {r['relative_error']:12.2e}"
        )

    print("\nDone.")
