"""Visualization for Bell test results and retrocausal model comparisons."""

import numpy as np
import matplotlib.pyplot as plt


def plot_correlation_curves(comparison_data, save_path: str | None = None):
    """Plot E(0, theta) for all models on the same axes.

    QM and retrocausal models trace out -cos(theta).
    Classical models trace out a linear curve.

    Args:
        comparison_data: DataFrame from BellTestComparator.run_chsh_sweep().
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "QM (analytical)": ("k-", 3),
        "ZigZag Retrocausal": ("ro", 1),
        "Boundary-Value": ("bs", 1),
        "Classical Local": ("g^", 1),
    }

    for model_name in comparison_data["model"].unique():
        subset = comparison_data[comparison_data["model"] == model_name]
        style, lw = styles.get(model_name, (".-", 1))

        if "o" in style or "s" in style or "^" in style:
            ax.plot(subset["angle"], subset["correlation"],
                    style, label=model_name, markersize=4, alpha=0.7)
        else:
            ax.plot(subset["angle"], subset["correlation"],
                    style, label=model_name, linewidth=lw)

    ax.set_xlabel("Angle theta (rad)")
    ax.set_ylabel("E(0, theta)")
    ax.set_title("Bell Correlation Curves: Retrocausal vs Classical vs QM\n"
                 "Retrocausal models match QM; classical models deviate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_chsh_comparison(chsh_values: dict[str, float],
                          save_path: str | None = None):
    """Bar chart comparing CHSH values across models.

    Shows the classical bound (2) and Tsirelson bound (2*sqrt(2)).

    Args:
        chsh_values: Dict of {model_name: S_value}.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(chsh_values.keys())
    values = [chsh_values[n] for n in names]
    colors = []
    for v in values:
        if abs(v) > 2:
            colors.append("coral")  # violates classical
        else:
            colors.append("steelblue")  # respects classical

    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor="black")

    # Reference lines
    ax.axhline(y=2.0, color="green", linewidth=2, linestyle="--",
               label="Classical bound (S=2)")
    ax.axhline(y=2 * np.sqrt(2), color="red", linewidth=2, linestyle="--",
               label=f"Tsirelson bound (S=2sqrt(2)={2*np.sqrt(2):.3f})")

    ax.set_ylabel("CHSH Value S")
    ax.set_title("CHSH Inequality Test: Which Models Violate Bell's Theorem?")
    ax.legend()

    # Rotate x labels
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_no_signaling_audit(audit_results: dict, save_path: str | None = None):
    """Visualize no-signaling audit results.

    For each model, shows the marginal P(A=+1 | a) as a function of a,
    for different Bob settings. Lines should overlap (no-signaling).

    Args:
        audit_results: Dict of {name: AuditResult}.
        save_path: Optional save path.
    """
    n_models = len(audit_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    for idx, (name, result) in enumerate(audit_results.items()):
        ax = axes[0][idx]
        df = result.details

        # For each Alice setting, plot P(A=+1) vs Bob setting
        alice_settings = df["alice_setting"].unique()
        for a in alice_settings[::max(1, len(alice_settings) // 5)]:
            subset = df[df["alice_setting"] == a]
            ax.plot(subset["bob_setting"], subset["p_alice_plus"],
                    ".-", alpha=0.5, label=f"a={a:.2f}")

        status = "PASS" if result.passed else "FAIL"
        ax.set_title(f"{name}\n[{status}] max dev={result.max_marginal_deviation:.4f}")
        ax.set_xlabel("Bob's setting b")
        ax.set_ylabel("P(A=+1 | a, b)")
        ax.set_ylim(0.3, 0.7)
        ax.axhline(y=0.5, color="red", linewidth=2, linestyle="--",
                   label="Expected (0.5)")

    plt.suptitle("No-Signaling Audit: P(A=+1) must be independent of Bob's setting",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
