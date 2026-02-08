"""Visualization for quantum eraser experiments.

Provides the key educational figures that demonstrate why the quantum
eraser does NOT demonstrate retrocausality.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_d0_total_vs_coincidence(eraser_result, n_bins: int = 100,
                                 save_path: str | None = None):
    """Side-by-side plot: total D0 pattern vs coincidence subsets.

    Left: Total D0 pattern (MUST be featureless -- no interference).
    Right: D0|D1, D0|D2, D0|D3, D0|D4 coincidence subsets.

    This is THE key educational figure: it shows that interference
    only appears in post-selected subsets, never in the total.

    Args:
        eraser_result: EraserResult from KimQuantumEraser.
        n_bins: Number of histogram bins.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Total D0 pattern
    centers, total = eraser_result.total_d0_pattern(n_bins)
    axes[0].bar(centers * 1e3, total, width=(centers[1] - centers[0]) * 1e3,
                color="steelblue", alpha=0.7)
    axes[0].set_xlabel("D0 position (mm)")
    axes[0].set_ylabel("Counts")
    axes[0].set_title("Total D0 Pattern\n(No interference -- no-signaling)")

    # Right: Coincidence subsets
    colors = {"D1": "red", "D2": "blue", "D3": "green", "D4": "orange"}
    labels = {
        "D1": "D0|D1 (erased, fringes)",
        "D2": "D0|D2 (erased, anti-fringes)",
        "D3": "D0|D3 (preserved, no fringes)",
        "D4": "D0|D4 (preserved, no fringes)",
    }
    for det in ["D1", "D2", "D3", "D4"]:
        c, counts = eraser_result.coincidence_pattern(det, n_bins)
        if len(counts) > 0 and counts.sum() > 0:
            # Normalize for comparison
            counts_norm = counts / max(counts.max(), 1)
            axes[1].plot(c * 1e3, counts_norm, color=colors[det],
                        label=labels[det], alpha=0.8)
    axes[1].set_xlabel("D0 position (mm)")
    axes[1].set_ylabel("Normalized counts")
    axes[1].set_title("Coincidence-Selected Subsets\n(Interference appears only here)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_no_signaling_comparison(eraser_result, n_bins: int = 100,
                                  save_path: str | None = None):
    """Show that D1+D2 fringes cancel out, proving no-signaling.

    The D1 and D2 patterns are complementary (anti-phase):
    D0|D1 ~ cos^2, D0|D2 ~ sin^2, so D0|D1 + D0|D2 = constant.
    Adding D3 and D4 (already featureless) gives the total: featureless.

    Args:
        eraser_result: EraserResult from KimQuantumEraser.
        n_bins: Number of histogram bins.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    detectors = ["D1", "D2", "D3", "D4"]
    titles = [
        "D0|D1 (erased -- interference fringes)",
        "D0|D2 (erased -- anti-fringes)",
        "D0|D3 (preserved -- no fringes)",
        "D0|D4 (preserved -- no fringes)",
    ]
    colors = ["red", "blue", "green", "orange"]

    all_counts = {}
    for ax, det, title, color in zip(axes.flat, detectors, titles, colors):
        c, counts = eraser_result.coincidence_pattern(det, n_bins)
        all_counts[det] = counts
        ax.bar(c * 1e3, counts, width=(c[1] - c[0]) * 1e3,
               color=color, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("D0 position (mm)")
        ax.set_ylabel("Counts")

    plt.suptitle("Why the quantum eraser does NOT violate no-signaling:\n"
                 "D1 and D2 fringes are anti-phase and cancel in the total",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_fringe_cancellation(eraser_result, n_bins: int = 100,
                              save_path: str | None = None):
    """Explicitly show how D1+D2 cancellation removes fringes.

    Args:
        eraser_result: EraserResult from KimQuantumEraser.
        n_bins: Number of histogram bins.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    c1, d1 = eraser_result.coincidence_pattern("D1", n_bins)
    c2, d2 = eraser_result.coincidence_pattern("D2", n_bins)

    # D1 fringes
    axes[0].plot(c1 * 1e3, d1, "r-", alpha=0.8)
    axes[0].set_title("D0|D1 (fringes)")
    axes[0].set_xlabel("Position (mm)")

    # D2 anti-fringes
    axes[1].plot(c2 * 1e3, d2, "b-", alpha=0.8)
    axes[1].set_title("D0|D2 (anti-fringes)")
    axes[1].set_xlabel("Position (mm)")

    # Sum: cancellation
    combined = d1 + d2
    axes[2].plot(c1 * 1e3, combined, "purple", alpha=0.8, linewidth=2)
    axes[2].set_title("D0|D1 + D0|D2 (fringes CANCEL)")
    axes[2].set_xlabel("Position (mm)")

    for ax in axes:
        ax.set_ylabel("Counts")

    plt.suptitle("Fringe Cancellation: Why no information goes backward",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
