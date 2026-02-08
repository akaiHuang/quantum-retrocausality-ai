"""Statistical analysis tools for retrocausality experiments.

Provides chi-squared tests, fringe visibility, mutual information,
correlation coefficients, and bootstrap confidence intervals.
"""

import numpy as np
from scipy import stats


def chi_squared_uniformity(data: np.ndarray, n_bins: int = 50) -> tuple[float, float]:
    """Chi-squared test for uniformity of a distribution.

    Used to verify that the total D0 pattern is featureless (no fringes).
    We test against a smoothed envelope rather than strict uniformity,
    since the diffraction envelope is Gaussian-like.

    Args:
        data: 1-D array of counts per bin (histogram).
        n_bins: Not used if data is already binned.

    Returns:
        (chi2_statistic, p_value). High p-value means consistent with smooth envelope.
    """
    data = np.asarray(data, dtype=float)
    if data.sum() == 0:
        return 0.0, 1.0

    # Smooth the data to get expected envelope (moving average)
    kernel_size = max(5, len(data) // 10)
    kernel = np.ones(kernel_size) / kernel_size
    expected = np.convolve(data, kernel, mode="same")
    expected = np.maximum(expected, 1e-10)  # avoid division by zero

    # Scale expected to match total counts
    expected *= data.sum() / expected.sum()

    # Chi-squared test
    mask = expected > 5  # standard requirement for chi-squared validity
    if mask.sum() < 3:
        return 0.0, 1.0

    chi2 = float(np.sum((data[mask] - expected[mask]) ** 2 / expected[mask]))
    dof = int(mask.sum() - 1)
    p_value = float(1.0 - stats.chi2.cdf(chi2, dof))

    return chi2, p_value


def fringe_visibility(pattern: np.ndarray) -> float:
    """Compute fringe visibility of an interference pattern.

    V = (I_max - I_min) / (I_max + I_min)

    For the total D0 pattern: V should be ~0 (no fringes).
    For coincidence-selected subsets: V should be high (clear fringes).

    Uses the central region of the pattern to avoid edge effects.

    Args:
        pattern: 1-D array of intensity/count values.

    Returns:
        Visibility in [0, 1]. 0 = no fringes, 1 = perfect fringes.
    """
    pattern = np.asarray(pattern, dtype=float)
    if len(pattern) < 3 or pattern.max() == 0:
        return 0.0

    # Use central 60% to avoid edge falloff
    n = len(pattern)
    start = int(n * 0.2)
    end = int(n * 0.8)
    central = pattern[start:end]

    if len(central) < 3 or central.max() == 0:
        return 0.0

    # Remove smooth envelope by dividing by moving average
    kernel_size = max(3, len(central) // 5)
    kernel = np.ones(kernel_size) / kernel_size
    envelope = np.convolve(central, kernel, mode="same")
    envelope = np.maximum(envelope, 1e-10)
    detrended = central / envelope

    i_max = float(np.max(detrended))
    i_min = float(np.min(detrended))

    if i_max + i_min == 0:
        return 0.0

    return (i_max - i_min) / (i_max + i_min)


def mutual_information(x: np.ndarray, y: np.ndarray,
                       n_bins: int = 20) -> float:
    """Estimate mutual information I(X; Y) from samples.

    Used for no-signaling verification: I(Alice_outcomes; Bob_settings) = 0.

    Args:
        x: Array of values for variable X.
        y: Array of values for variable Y.
        n_bins: Number of bins for discretization.

    Returns:
        Mutual information in nats (>= 0).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # 2D histogram for joint distribution
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist_2d = hist_2d / hist_2d.sum()

    # Marginals
    p_x = hist_2d.sum(axis=1)
    p_y = hist_2d.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if hist_2d[i, j] > 1e-15 and p_x[i] > 1e-15 and p_y[j] > 1e-15:
                mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_x[i] * p_y[j]))

    return max(0.0, mi)


def correlation_coefficient(x: np.ndarray, y: np.ndarray
                            ) -> tuple[float, float, tuple[float, float]]:
    """Pearson correlation with p-value and 95% confidence interval.

    Args:
        x: Array of values.
        y: Array of values.

    Returns:
        (r, p_value, (ci_lower, ci_upper)).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    r, p = stats.pearsonr(x, y)

    # Fisher z-transform for CI
    n = len(x)
    if n < 4:
        return float(r), float(p), (-1.0, 1.0)

    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_lo = z - 1.96 * se
    z_hi = z + 1.96 * se
    ci = (float(np.tanh(z_lo)), float(np.tanh(z_hi)))

    return float(r), float(p), ci


def bootstrap_confidence_interval(data: np.ndarray, statistic: callable,
                                   n_bootstrap: int = 10000,
                                   ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap confidence interval for any statistic.

    Args:
        data: Input data array.
        statistic: Function that takes an array and returns a scalar.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.

    Returns:
        (point_estimate, ci_lower, ci_upper).
    """
    data = np.asarray(data)
    point = float(statistic(data))

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = data[np.random.randint(0, len(data), size=len(data))]
        boot_stats.append(float(statistic(sample)))

    boot_stats = np.array(boot_stats)
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return point, lo, hi
