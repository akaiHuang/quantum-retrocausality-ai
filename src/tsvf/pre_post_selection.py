"""Pre- and post-selection ensemble management.

In practice, TSVF applies to experiments where:
1. A state is prepared (pre-selection)
2. Some intermediate measurement/interaction occurs
3. A final measurement is performed and we KEEP only specific outcomes (post-selection)

The post-selection rate is |<phi|U|psi>|^2, which can be very small.
This module handles the bookkeeping of generating these ensembles.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .two_state_vector import TwoStateVector
from .weak_values import WeakValueCalculator


@dataclass
class EnsembleStats:
    """Statistics about a pre-and-post-selected ensemble."""
    n_total: int
    n_selected: int
    selection_rate: float
    post_state_label: str


class PrePostSelectionEnsemble:
    """Manage ensembles of pre-and-post-selected quantum systems.

    This class provides tools for:
    - Computing post-selection rates
    - Analyzing how post-selection affects statistics (collider bias)
    - Sweeping over post-selection states to show retrocausal dependence
    """

    def __init__(self, pre_state: np.ndarray,
                 hamiltonian: np.ndarray | None = None):
        """
        Args:
            pre_state: The prepared (pre-selected) state.
            hamiltonian: Time evolution Hamiltonian (optional).
        """
        self.pre = np.asarray(pre_state, dtype=complex).ravel()
        self.pre = self.pre / np.linalg.norm(self.pre)
        self.H = hamiltonian

    def post_selection_rate(self, post_state: np.ndarray,
                            t_f: float = 1.0) -> float:
        """Probability that post-selection on |phi> succeeds.

        P = |<phi|U(t_f)|psi>|^2

        Args:
            post_state: Post-selection state.
            t_f: Time of post-selection.

        Returns:
            Probability in [0, 1].
        """
        tsv = TwoStateVector(self.pre, post_state, self.H)
        return tsv.post_selection_probability(t_f=t_f)

    def generate_ensemble(self, post_state: np.ndarray,
                          n_total: int = 10000,
                          t_f: float = 1.0) -> EnsembleStats:
        """Simulate n_total trials and post-select.

        Args:
            post_state: Post-selection state.
            n_total: Total number of trials.
            t_f: Post-selection time.

        Returns:
            EnsembleStats with selection rate.
        """
        rate = self.post_selection_rate(post_state, t_f)
        n_selected = np.random.binomial(n_total, rate)

        return EnsembleStats(
            n_total=n_total,
            n_selected=n_selected,
            selection_rate=rate,
            post_state_label=f"|phi> (dim={len(post_state)})",
        )

    def post_selection_bias_analysis(self, observable: np.ndarray,
                                      post_states: dict[str, np.ndarray],
                                      t: float = 0.0,
                                      t_f: float = 1.0) -> pd.DataFrame:
        """Analyze how different post-selections change weak values.

        This is crucial for understanding why the quantum eraser is not retrocausal:
        the "effect" is entirely due to post-selection (collider bias in causal language).

        Collider bias: When you condition on a common effect of two causes,
        you induce a spurious correlation between the causes. Post-selection
        acts as a collider, creating apparent retrocausality.

        Args:
            observable: The observable to compute weak values for.
            post_states: Dict of {label: state_vector} for different post-selections.
            t: Intermediate time.
            t_f: Post-selection time.

        Returns:
            DataFrame with columns [post_label, weak_value_real, weak_value_imag,
            is_anomalous, selection_rate].
        """
        rows = []
        for label, post in post_states.items():
            tsv = TwoStateVector(self.pre, post, self.H)
            calc = WeakValueCalculator(tsv)
            try:
                wv = calc.compute(observable, t, t_f=t_f)
                anomalous = calc.is_anomalous(observable, t, t_f=t_f)
            except ValueError:
                wv = complex(float("inf"), float("inf"))
                anomalous = True

            rate = self.post_selection_rate(post, t_f)
            rows.append({
                "post_label": label,
                "weak_value_real": wv.real,
                "weak_value_imag": wv.imag,
                "is_anomalous": anomalous,
                "selection_rate": rate,
            })

        return pd.DataFrame(rows)

    def sweep_post_selection_angle(self, observable: np.ndarray,
                                    n_angles: int = 50,
                                    t: float = 0.0,
                                    t_f: float = 1.0) -> pd.DataFrame:
        """Sweep post-selection state on the Bloch sphere (for 2-level systems).

        Post-selects on cos(theta/2)|0> + sin(theta/2)|1> for theta in [0, pi].

        Shows how the weak value varies continuously with the future measurement choice.

        Args:
            observable: 2x2 observable matrix.
            n_angles: Number of angles to sweep.
            t: Intermediate time.
            t_f: Post-selection time.

        Returns:
            DataFrame with [theta, weak_value_real, weak_value_imag, selection_rate].
        """
        if len(self.pre) != 2:
            raise ValueError("Angle sweep only implemented for 2-level systems")

        thetas = np.linspace(0.01, np.pi - 0.01, n_angles)
        rows = []
        for theta in thetas:
            post = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
            tsv = TwoStateVector(self.pre, post, self.H)
            calc = WeakValueCalculator(tsv)
            try:
                wv = calc.compute(observable, t, t_f=t_f)
            except ValueError:
                wv = complex(float("nan"), float("nan"))

            rate = self.post_selection_rate(post, t_f)
            rows.append({
                "theta": float(theta),
                "weak_value_real": wv.real,
                "weak_value_imag": wv.imag,
                "selection_rate": rate,
            })

        return pd.DataFrame(rows)
