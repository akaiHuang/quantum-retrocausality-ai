"""Visualization for TSVF weak values and related quantities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_weak_value_complex_plane(weak_values: dict[str, complex],
                                  eigenvalues: list[float],
                                  title: str = "Weak Values in Complex Plane",
                                  save_path: str | None = None):
    """Plot weak values as points in the complex plane.

    Marks the eigenvalue range on the real axis. Anomalous values
    appear outside this range -- the signature of TSVF.

    Args:
        weak_values: Dict of {observable_name: weak_value}.
        eigenvalues: List of eigenvalues of the observable.
        title: Plot title.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Eigenvalue range on real axis
    ev_min, ev_max = min(eigenvalues), max(eigenvalues)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.fill_betweenx([-0.3, 0.3], ev_min, ev_max, alpha=0.2, color="green",
                      label=f"Eigenvalue range [{ev_min:.1f}, {ev_max:.1f}]")

    # Plot eigenvalues as diamonds
    for ev in eigenvalues:
        ax.plot(ev, 0, "gD", markersize=10, zorder=5)

    # Plot weak values
    colors = plt.cm.tab10(np.linspace(0, 1, len(weak_values)))
    for (name, wv), color in zip(weak_values.items(), colors):
        anomalous = wv.real < ev_min - 1e-10 or wv.real > ev_max + 1e-10
        marker = "s" if anomalous else "o"
        label = f"{name}: {wv.real:.2f} + {wv.imag:.2f}i"
        if anomalous:
            label += " (ANOMALOUS)"
        ax.plot(wv.real, wv.imag, marker, color=color, markersize=12,
                label=label, zorder=10)

    ax.set_xlabel("Re(A_w)")
    ax.set_ylabel("Im(A_w)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pointer_distribution(result, save_path: str | None = None):
    """Plot the pointer readout distribution from a weak measurement.

    Shows the pointer shifted by Re(A_w) * g, which can be outside
    the eigenvalue range (anomalous weak value).

    Args:
        result: WeakMeasurementResult from WeakValueCalculator.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pointer distribution
    ax.hist(result.pointer_readings, bins=50, density=True, alpha=0.7,
            color="steelblue", label="Pointer readings (post-selected)")

    # Theoretical prediction
    ax.axvline(result.pointer_shift_theory, color="red", linewidth=2,
               linestyle="--",
               label=f"Theory: Re(A_w)*g = {result.pointer_shift_theory:.3f}")

    # Measured mean
    ax.axvline(result.pointer_shift_measured, color="orange", linewidth=2,
               linestyle=":",
               label=f"Measured mean = {result.pointer_shift_measured:.3f}")

    # Eigenvalue positions (scaled by coupling)
    for ev in result.eigenvalues:
        ax.axvline(ev * result.coupling_strength, color="green", linewidth=1,
                   alpha=0.5, linestyle="-.")

    if result.is_anomalous:
        ax.set_title("Weak Measurement: ANOMALOUS Weak Value\n"
                     "Pointer shifted BEYOND eigenvalue range",
                     fontweight="bold", color="darkred")
    else:
        ax.set_title("Weak Measurement: Normal Weak Value")

    ax.set_xlabel("Pointer position")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_abl_vs_born(comparison, save_path: str | None = None):
    """Bar chart comparing ABL and Born rule probabilities.

    Highlights the retrocausal character: knowledge of the FUTURE
    (post-selection) changes intermediate measurement predictions.

    Args:
        comparison: ABLComparisonResult from ABLRule.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(comparison.eigenvalues))
    width = 0.35

    bars1 = ax.bar(x - width / 2, comparison.born_probabilities, width,
                   label="Born rule (past only)", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, comparison.abl_probabilities, width,
                   label="ABL rule (past + future)", color="coral", alpha=0.8)

    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Probability")
    ax.set_title(f"ABL vs Born Rule\n"
                 f"Time-symmetric: {comparison.time_symmetric} | "
                 f"Max difference: {comparison.max_difference:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in comparison.eigenvalues])
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_weak_value_sweep(sweep_df, observable_name: str = "A",
                          eigenvalue_range: tuple[float, float] | None = None,
                          save_path: str | None = None):
    """Plot weak value as a function of post-selection angle.

    Shows how the weak value varies continuously with the future measurement
    choice, demonstrating the retrocausal dependence.

    Args:
        sweep_df: DataFrame from PrePostSelectionEnsemble.sweep_post_selection_angle.
        observable_name: Name of the observable for labeling.
        eigenvalue_range: Optional (min, max) to shade the normal range.
        save_path: Optional save path.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Weak value real and imaginary parts
    ax1.plot(sweep_df["theta"], sweep_df["weak_value_real"],
             "b-", linewidth=2, label=f"Re({observable_name}_w)")
    ax1.plot(sweep_df["theta"], sweep_df["weak_value_imag"],
             "r--", linewidth=2, label=f"Im({observable_name}_w)")

    if eigenvalue_range is not None:
        ax1.axhspan(eigenvalue_range[0], eigenvalue_range[1],
                    alpha=0.1, color="green", label="Eigenvalue range")

    ax1.set_ylabel(f"Weak value of {observable_name}")
    ax1.set_title(f"Weak Value vs Post-Selection Angle\n"
                  f"(Retrocausal dependence on future measurement)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Post-selection rate
    ax2.plot(sweep_df["theta"], sweep_df["selection_rate"],
             "k-", linewidth=2)
    ax2.set_xlabel("Post-selection angle theta (rad)")
    ax2.set_ylabel("Post-selection probability")
    ax2.set_title("Post-selection rate")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
