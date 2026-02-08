// physics.js - Gravitational Light Physics Module
// Based on the Schwarzschild metric from General Relativity

// ============================================================
// Physical Constants
// ============================================================
export const C = 299792458;           // Speed of light in vacuum (m/s)
export const G = 6.6743e-11;          // Gravitational constant (m^3 kg^-1 s^-2)
export const M_SUN = 1.989e30;        // Solar mass (kg)

// ============================================================
// Schwarzschild Geometry
// ============================================================

/**
 * Schwarzschild radius: r_s = 2GM/c^2
 * The radius of the event horizon for a non-rotating black hole.
 */
export function schwarzschildRadius(M) {
    return 2 * G * M / (C * C);
}

/**
 * Photon sphere radius: r_ps = 1.5 * r_s = 3GM/c^2
 * The radius at which photons orbit in unstable circular orbits.
 */
export function photonSphereRadius(M) {
    return 1.5 * schwarzschildRadius(M);
}

/**
 * ISCO (Innermost Stable Circular Orbit) radius: r_isco = 3 * r_s = 6GM/c^2
 */
export function iscoRadius(M) {
    return 3 * schwarzschildRadius(M);
}

// ============================================================
// Speed of Light in Schwarzschild Geometry
// ============================================================

/**
 * The gravity factor (1 - r_s/r), which appears throughout the metric.
 * Returns 0 at event horizon, negative inside (unphysical region).
 */
export function gravityFactor(r, M) {
    const rs = schwarzschildRadius(M);
    if (r <= 0) return 0;
    return 1 - rs / r;
}

/**
 * Effective coordinate speed of light as a function of distance.
 *
 * In Schwarzschild coordinates:
 *   Radial:     c_r = c * (1 - r_s/r)
 *   Tangential: c_t = c * sqrt(1 - r_s/r)
 *
 * For a ray at angle theta to the radial direction:
 *   1/c_eff^2 = cos^2(theta)/c_r^2 + sin^2(theta)/c_t^2
 *
 * For general ray tracing we use the isotropic approximation:
 *   c_eff(r) ~ c * (1 - r_s/r) for radial
 *
 * @param {number} r - Distance from center (m)
 * @param {number} M - Mass (kg)
 * @param {string} mode - 'radial', 'tangential', or 'average'
 * @returns {number} Effective speed of light (m/s)
 */
export function effectiveLightSpeed(r, M, mode = 'radial') {
    const rs = schwarzschildRadius(M);
    if (r <= rs) return 0;

    const factor = 1 - rs / r;

    switch (mode) {
        case 'radial':
            return C * factor;
        case 'tangential':
            return C * Math.sqrt(factor);
        case 'average':
            // Geometric mean of radial and tangential
            return C * Math.pow(factor, 0.75);
        default:
            return C * factor;
    }
}

/**
 * Refractive index of spacetime: n(r) = c / c_eff(r)
 * For the radial case: n(r) = 1 / (1 - r_s/r)
 *
 * For weak fields, this approximates to: n ~ 1 + r_s/r = 1 + 2GM/(rc^2)
 * This is the form used for ray tracing (works well for visualization).
 */
export function refractiveIndex(r, M) {
    const rs = schwarzschildRadius(M);
    if (r <= rs) return Infinity;
    const factor = 1 - rs / r;
    if (factor <= 0) return Infinity;
    return 1 / factor;
}

/**
 * Gradient of refractive index (radial component).
 *
 * n(r) = 1/(1 - r_s/r) = r/(r - r_s)
 * dn/dr = -r_s / (r - r_s)^2
 *
 * The gradient vector points radially inward (toward the mass),
 * meaning light bends toward the mass.
 */
export function refractiveIndexGradient(r, M) {
    const rs = schwarzschildRadius(M);
    if (r <= rs) return 0;
    const diff = r - rs;
    return -rs / (diff * diff);
}

// ============================================================
// Analytical Formulas for Light Deflection
// ============================================================

/**
 * Weak-field deflection angle (Einstein formula).
 * Valid for b >> r_s.
 *
 * delta_phi = 4GM / (bc^2) = 2 * r_s / b
 *
 * @param {number} b - Impact parameter (closest approach distance for straight ray)
 * @param {number} M - Mass (kg)
 * @returns {number} Deflection angle in radians
 */
export function deflectionAngleWeak(b, M) {
    const rs = schwarzschildRadius(M);
    if (b <= 0) return Infinity;
    return 2 * rs / b;
}

/**
 * Higher-order deflection angle (includes second-order correction).
 * delta_phi = 4GM/(bc^2) + 15*pi*(GM)^2 / (4*b^2*c^4)
 */
export function deflectionAngle(b, M) {
    const rs = schwarzschildRadius(M);
    if (b <= 0) return Infinity;

    const ratio = rs / b;
    // First order + second order correction
    return 2 * ratio + (15 * Math.PI / 16) * ratio * ratio;
}

/**
 * Shapiro time delay.
 *
 * The extra time for light to travel near a massive body compared
 * to the flat-space travel time.
 *
 * delta_t = (2GM/c^3) * ln((4*r1*r2)/(b^2))
 *
 * @param {number} r1 - Distance of source from mass (m)
 * @param {number} r2 - Distance of observer from mass (m)
 * @param {number} b - Closest approach distance (m)
 * @param {number} M - Mass (kg)
 * @returns {number} Extra time delay in seconds
 */
export function shapiroDelay(r1, r2, b, M) {
    const rs = schwarzschildRadius(M);
    if (b <= 0 || r1 <= 0 || r2 <= 0) return Infinity;

    // Classical Shapiro delay formula
    const delay = (rs / C) * Math.log((4 * r1 * r2) / (b * b));
    return Math.max(0, delay);
}

// ============================================================
// Numerical Ray Tracing
// ============================================================

/**
 * Trace a light ray through the Schwarzschild spacetime.
 *
 * Uses the eikonal equation / Fermat's principle:
 *   d/ds (n * v_hat) = grad(n)
 *
 * where:
 *   n(r) is the refractive index
 *   v_hat is the unit direction of the ray
 *   s is the path parameter
 *
 * This is equivalent to saying light follows the path of least
 * optical path length, and bends toward regions of higher n
 * (lower effective c), just like in a glass lens.
 *
 * We work in 2D (x-y plane) for visualization.
 *
 * @param {Object} startPos - {x, y} start position
 * @param {Object} direction - {x, y} initial direction (will be normalized)
 * @param {number} M - Mass (kg)
 * @param {number} maxSteps - Maximum integration steps
 * @param {number} dt - Time step size (in scaled units)
 * @param {number} scale - Spatial scale factor (simulation units per meter)
 * @returns {Object} Ray trace result with path points and metadata
 */
export function traceRay(startPos, direction, M, maxSteps = 10000, dt = 0.005, scale = 1.0) {
    const rs = schwarzschildRadius(M);

    // Normalize direction
    const dirLen = Math.sqrt(direction.x * direction.x + direction.y * direction.y);
    let vx = direction.x / dirLen;
    let vy = direction.y / dirLen;

    // Current position (in meters, physical units)
    let x = startPos.x / scale;
    let y = startPos.y / scale;

    // Step size in physical units
    const stepSize = dt / scale;

    // Results
    const points = [];
    const speeds = [];
    const distances = [];
    let totalTime = 0;
    let straightLineDistance = 0;
    let minDistance = Infinity;
    let captured = false;
    let escaped = false;

    // Bounding radius for escape detection
    const startR = Math.sqrt(x * x + y * y);
    const escapeRadius = startR * 3;

    for (let i = 0; i < maxSteps; i++) {
        const r = Math.sqrt(x * x + y * y);

        // Store point
        points.push({ x: x * scale, y: y * scale });
        distances.push(r);

        // Track minimum approach distance
        if (r < minDistance) {
            minDistance = r;
        }

        // Check if ray hit the event horizon
        if (r <= rs * 1.01) {
            captured = true;
            speeds.push(0);
            break;
        }

        // Check if ray has escaped far enough
        if (r > escapeRadius && i > 100) {
            escaped = true;
            speeds.push(C);
            break;
        }

        // Compute refractive index and its gradient at current position
        const factor = 1 - rs / r;
        if (factor <= 0) {
            captured = true;
            speeds.push(0);
            break;
        }

        const n = 1 / factor;
        const cEff = C * factor;
        speeds.push(cEff);

        // Gradient of n in Cartesian coordinates
        // dn/dr = -r_s / (r - r_s)^2
        const dndR = -rs / ((r - rs) * (r - rs));

        // r_hat = (x/r, y/r)
        const rx = x / r;
        const ry = y / r;

        // grad(n) = (dn/dr) * r_hat
        const gradNx = dndR * rx;
        const gradNy = dndR * ry;

        // Eikonal equation: d/ds(n * v_hat) = grad(n)
        // => n * dv/ds + (dn/ds) * v = grad(n)
        // => dv/ds = (grad(n) - (dn/ds) * v) / n
        // where dn/ds = grad(n) . v

        const dnds = gradNx * vx + gradNy * vy;

        const dvx = (gradNx - dnds * vx) / n;
        const dvy = (gradNy - dnds * vy) / n;

        // Adaptive step size - smaller steps closer to the mass
        let adaptiveDt = stepSize;
        const rRatio = r / rs;
        if (rRatio < 5) {
            adaptiveDt = stepSize * Math.max(0.05, (rRatio - 1) / 4);
        }

        // Update velocity (direction)
        vx += dvx * adaptiveDt;
        vy += dvy * adaptiveDt;

        // Re-normalize velocity direction
        const vLen = Math.sqrt(vx * vx + vy * vy);
        vx /= vLen;
        vy /= vLen;

        // Move photon
        const moveScale = cEff * adaptiveDt;
        x += vx * moveScale;
        y += vy * moveScale;

        // Accumulate time
        totalTime += adaptiveDt;
    }

    // Compute deflection angle
    let deflection = 0;
    if (points.length >= 2 && !captured) {
        const startDir = {
            x: direction.x / dirLen,
            y: direction.y / dirLen
        };
        const endDir = { x: vx, y: vy };

        // Angle between start and end direction
        const dot = startDir.x * endDir.x + startDir.y * endDir.y;
        const cross = startDir.x * endDir.y - startDir.y * endDir.x;
        deflection = Math.atan2(Math.abs(cross), dot);
    }

    // Compute straight-line travel time for Shapiro delay comparison
    if (points.length >= 2) {
        const dx = points[points.length - 1].x - points[0].x;
        const dy = points[points.length - 1].y - points[0].y;
        straightLineDistance = Math.sqrt(dx * dx + dy * dy) / scale;
    }
    const straightLineTime = straightLineDistance / C;
    const shapiroDelayValue = totalTime - straightLineTime;

    // Compute impact parameter (perpendicular distance from mass center to initial ray line)
    const impactParameter = Math.abs(startPos.x * direction.y - startPos.y * direction.x) /
                           (dirLen * scale);

    return {
        points,
        speeds,
        distances,
        totalTime,
        straightLineTime,
        shapiroDelay: shapiroDelayValue,
        deflection,
        minDistance,
        impactParameter,
        captured,
        escaped,
        rs
    };
}

// ============================================================
// Utility Functions
// ============================================================

/**
 * Convert solar masses to kg
 */
export function solarMassesToKg(solarMasses) {
    return solarMasses * M_SUN;
}

/**
 * Format a number in scientific notation
 */
export function formatSci(value, digits = 3) {
    if (!isFinite(value)) return 'Infinity';
    if (value === 0) return '0';
    return value.toExponential(digits);
}

/**
 * Convert radians to arcseconds
 */
export function radiansToArcseconds(rad) {
    return rad * (180 / Math.PI) * 3600;
}

/**
 * Convert radians to degrees
 */
export function radiansToDegrees(rad) {
    return rad * (180 / Math.PI);
}

/**
 * Compute c_eff / c ratio at a given distance
 */
export function speedRatio(r, M) {
    const rs = schwarzschildRadius(M);
    if (r <= rs) return 0;
    return 1 - rs / r;
}

/**
 * Get body properties for preset objects
 */
export function getPresetProperties(preset) {
    switch (preset) {
        case 'sun':
            return {
                mass: 1.0,
                name: 'Sun',
                color: 0xffaa00,
                emissiveColor: 0xff8800,
                description: 'Our Sun - very weak gravitational lensing'
            };
        case 'neutron_star':
            return {
                mass: 2.0,
                name: 'Neutron Star',
                color: 0x88bbff,
                emissiveColor: 0x4488ff,
                description: 'Neutron star (2 M_sun) - moderate lensing'
            };
        case 'black_hole':
            return {
                mass: 10.0,
                name: 'Black Hole (10 M_sun)',
                color: 0x111111,
                emissiveColor: 0x220000,
                description: 'Stellar black hole - strong lensing, event horizon visible'
            };
        default:
            return {
                mass: 1.0,
                name: 'Custom',
                color: 0xffaa00,
                emissiveColor: 0xff8800,
                description: ''
            };
    }
}
