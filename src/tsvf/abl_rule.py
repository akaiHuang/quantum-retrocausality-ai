"""ABL (Aharonov-Bergmann-Lebowitz) rule implementation.

Reference: Aharonov, Bergmann, Lebowitz (1964), Phys. Rev. B 134, 1410-1416.

For a system pre-selected in |psi> and post-selected in |phi>,
the probability of finding eigenvalue a_j of observable A in a
STRONG intermediate measurement is:

    P(a_j) = |<phi|Pi_j|psi>|^2 / sum_k |<phi|Pi_k|psi>|^2

where Pi_j = |a_j><a_j| is the projector onto eigenstate |a_j>.

This differs from the standard Born rule:
- Born rule: P(a_j) = |<a_j|psi>|^2 (only depends on past)
- ABL rule: P(a_j) = |<phi|a_j><a_j|psi>|^2 / Z (depends on BOTH past and future)

The ABL rule is the TIME-SYMMETRIC generalization of the Born rule.
"""

import numpy as np
from dataclasses import dataclass

from .two_state_vector import TwoStateVector


@dataclass
class ABLComparisonResult:
    """Comparison between ABL and Born rule probabilities."""
    eigenvalues: np.ndarray
    abl_probabilities: np.ndarray     # with post-selection (time-symmetric)
    born_probabilities: np.ndarray    # without post-selection (standard)
    max_difference: float
    time_symmetric: bool              # ABL is invariant under time reversal


class ABLRule:
    """Compute ABL probabilities for intermediate measurements
    in pre-and-post-selected systems.

    The ABL rule is the operational prediction of the TSVF for STRONG
    intermediate measurements. It gives probabilities that depend on
    BOTH the preparation (past) and the post-selection (future).
    """

    def __init__(self, two_state_vector: TwoStateVector):
        """
        Args:
            two_state_vector: The pre/post-selected system.
        """
        self.tsv = two_state_vector

    def probability(self, observable: np.ndarray, eigenvalue_index: int,
                    t: float = 0.0, t_i: float = 0.0,
                    t_f: float = 1.0) -> float:
        """Compute ABL probability for finding the i-th eigenvalue.

        P(a_j) = |<phi(t)|Pi_j|psi(t)>|^2 / sum_k |<phi(t)|Pi_k|psi(t)>|^2

        Args:
            observable: Hermitian matrix A.
            eigenvalue_index: Index j of the eigenvalue to query.
            t: Intermediate measurement time.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            Probability in [0, 1].
        """
        probs = self.all_probabilities(observable, t, t_i, t_f)
        eigenvalues = sorted(probs.keys())
        return probs[eigenvalues[eigenvalue_index]]

    def all_probabilities(self, observable: np.ndarray, t: float = 0.0,
                          t_i: float = 0.0,
                          t_f: float = 1.0) -> dict[float, float]:
        """Compute ABL probabilities for all eigenvalues.

        Args:
            observable: Hermitian matrix A.
            t: Intermediate measurement time.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            Dict of {eigenvalue: probability}. Probabilities sum to 1.
        """
        A = np.asarray(observable, dtype=complex)
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        snapshot = self.tsv.at_time(t, t_i, t_f)
        psi_t = snapshot.forward_state
        phi_t = snapshot.backward_state

        # |<phi(t)|a_j><a_j|psi(t)>|^2 for each eigenvalue
        unnormalized = {}
        for j, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            amplitude = np.vdot(phi_t, vec) * np.vdot(vec, psi_t)
            unnormalized[float(val)] = float(abs(amplitude) ** 2)

        # Normalize
        total = sum(unnormalized.values())
        if total < 1e-30:
            # Degenerate case: return uniform
            n = len(eigenvalues)
            return {float(v): 1.0 / n for v in eigenvalues}

        return {val: p / total for val, p in unnormalized.items()}

    def compare_with_born_rule(self, observable: np.ndarray,
                                t: float = 0.0, t_i: float = 0.0,
                                t_f: float = 1.0) -> ABLComparisonResult:
        """Compare ABL probabilities with standard Born rule.

        The difference highlights the retrocausal character of TSVF:
        knowledge of the FUTURE (post-selection) changes what we predict
        for intermediate measurements.

        Args:
            observable: Hermitian matrix A.
            t: Intermediate measurement time.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            ABLComparisonResult with both probability sets.
        """
        A = np.asarray(observable, dtype=complex)
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # ABL probabilities
        abl_probs = self.all_probabilities(A, t, t_i, t_f)

        # Born rule probabilities (no post-selection)
        psi_t = self.tsv.forward_evolve(t, t_i)
        born_probs = {}
        for val, vec in zip(eigenvalues, eigenvectors.T):
            born_probs[float(val)] = float(abs(np.vdot(vec, psi_t)) ** 2)

        # Arrays for comparison
        evals = np.array(sorted(abl_probs.keys()))
        abl_arr = np.array([abl_probs[v] for v in evals])
        born_arr = np.array([born_probs[v] for v in evals])

        # Time symmetry check
        # Swap pre/post and negate Hamiltonian
        tsv_reversed = TwoStateVector(self.tsv.post, self.tsv.pre, -self.tsv.H)
        abl_reversed = ABLRule(tsv_reversed)
        reversed_probs = abl_reversed.all_probabilities(A, t_f - t, t_i, t_f)
        reversed_arr = np.array([reversed_probs.get(v, 0.0) for v in evals])
        time_symmetric = bool(np.allclose(abl_arr, reversed_arr, atol=1e-10))

        return ABLComparisonResult(
            eigenvalues=evals,
            abl_probabilities=abl_arr,
            born_probabilities=born_arr,
            max_difference=float(np.max(np.abs(abl_arr - born_arr))),
            time_symmetric=time_symmetric,
        )

    def time_symmetry_demonstration(self, observable: np.ndarray,
                                     t: float = 0.5, t_i: float = 0.0,
                                     t_f: float = 1.0) -> dict:
        """Demonstrate ABL time symmetry.

        The ABL probabilities are unchanged if we:
        1. Swap pre/post states: |psi> <-> |phi>
        2. Reverse the Hamiltonian: H -> -H
        3. Mirror the time: t -> t_f - t

        This is the core time-symmetry of TSVF and the foundation
        of the retrocausal interpretation.

        Returns:
            Dict with original and reversed probabilities and verification.
        """
        comparison = self.compare_with_born_rule(observable, t, t_i, t_f)
        return {
            "original_abl": dict(zip(comparison.eigenvalues.tolist(),
                                     comparison.abl_probabilities.tolist())),
            "born_rule": dict(zip(comparison.eigenvalues.tolist(),
                                  comparison.born_probabilities.tolist())),
            "time_symmetric": comparison.time_symmetric,
            "max_abl_born_difference": comparison.max_difference,
            "explanation": (
                "ABL probabilities depend on BOTH past preparation and future "
                "post-selection. Born rule only depends on the past. When they "
                "differ, the future measurement is 'retrocausally' influencing "
                "intermediate predictions."
            ),
        }
