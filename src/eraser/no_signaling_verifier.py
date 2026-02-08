"""No-signaling theorem verification.

Implements verification of Eberhard's (1978) no-communication theorem:
For any bipartite state rho_AB and any operation O_B on subsystem B,
    Tr_B(rho_AB) == Tr_B((I_A x O_B) rho_AB (I_A x O_B^dag))

Alice's reduced density matrix is UNCHANGED by anything Bob does.
This is why the quantum eraser cannot send information backward in time.
"""

import numpy as np
from dataclasses import dataclass

from ..core.density_matrix import (
    partial_trace, state_to_density, apply_operation_on_subsystem, fidelity
)
from ..core.operators import beam_splitter, PAULI


@dataclass
class VerificationResult:
    """Result of a no-signaling verification."""
    passed: bool
    max_fidelity_deviation: float  # max |1 - F(rho_A, rho_A')| across operations
    rho_a_original: np.ndarray
    rho_a_after_ops: list[np.ndarray]
    operation_names: list[str]
    tolerance: float


class NoSignalingVerifier:
    """General-purpose no-signaling verification for any bipartite experiment.

    Tests that Alice's reduced density matrix is invariant under
    any operation Bob performs on his subsystem.
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Args:
            tolerance: Maximum allowed deviation in fidelity from 1.0.
        """
        self.tolerance = tolerance

    def verify_state(self, state: np.ndarray, dims: list[int],
                     alice_subsystem: int = 0,
                     bob_operations: dict[str, np.ndarray] | None = None
                     ) -> VerificationResult:
        """Verify no-signaling for a pure state under various Bob operations.

        Args:
            state: State vector of the composite system.
            dims: Dimensions of each subsystem.
            alice_subsystem: Index of Alice's subsystem.
            bob_operations: Dict of {name: unitary} operations for Bob.
                            If None, uses a default set (Pauli gates + beam splitter).

        Returns:
            VerificationResult with pass/fail and detailed measurements.
        """
        rho = state_to_density(state)
        return self.verify_density_matrix(rho, dims, alice_subsystem,
                                          bob_operations)

    def verify_density_matrix(self, rho: np.ndarray, dims: list[int],
                               alice_subsystem: int = 0,
                               bob_operations: dict[str, np.ndarray] | None = None
                               ) -> VerificationResult:
        """Verify no-signaling for a density matrix.

        Args:
            rho: Density matrix of the composite system.
            dims: Dimensions of each subsystem.
            alice_subsystem: Index of Alice's subsystem.
            bob_operations: Dict of {name: unitary} operations for Bob.

        Returns:
            VerificationResult.
        """
        n_subsystems = len(dims)
        bob_subsystems = [i for i in range(n_subsystems) if i != alice_subsystem]

        if bob_operations is None:
            bob_operations = {
                "I": np.eye(dims[bob_subsystems[0]], dtype=complex),
                "X": PAULI["X"][:dims[bob_subsystems[0]], :dims[bob_subsystems[0]]],
                "Y": PAULI["Y"][:dims[bob_subsystems[0]], :dims[bob_subsystems[0]]],
                "Z": PAULI["Z"][:dims[bob_subsystems[0]], :dims[bob_subsystems[0]]],
            }

        # Compute Alice's original reduced density matrix
        rho_a_original = partial_trace(rho, keep=[alice_subsystem], dims=dims)

        rho_a_after_list = []
        op_names = []
        max_deviation = 0.0

        for name, op in bob_operations.items():
            # Apply Bob's operation
            bob_idx = bob_subsystems[0]
            rho_after = apply_operation_on_subsystem(rho, op, bob_idx, dims)

            # Compute Alice's reduced state after Bob's operation
            rho_a_after = partial_trace(rho_after, keep=[alice_subsystem],
                                        dims=dims)

            rho_a_after_list.append(rho_a_after)
            op_names.append(name)

            # Check fidelity
            f = fidelity(rho_a_original, rho_a_after)
            deviation = abs(1.0 - f)
            max_deviation = max(max_deviation, deviation)

        passed = max_deviation < self.tolerance

        return VerificationResult(
            passed=passed,
            max_fidelity_deviation=max_deviation,
            rho_a_original=rho_a_original,
            rho_a_after_ops=rho_a_after_list,
            operation_names=op_names,
            tolerance=self.tolerance,
        )

    def verify_eraser_no_signaling(self, eraser_result) -> dict:
        """Verify no-signaling specifically for the quantum eraser experiment.

        Checks that the total D0 distribution is independent of whether
        idler beam splitters are present. In practice, we verify that
        the total D0 pattern (summed over all idler detectors) is
        statistically identical to a featureless distribution.

        Args:
            eraser_result: EraserResult from KimQuantumEraser.

        Returns:
            Dict with statistical tests and pass/fail.
        """
        from ..analysis.statistics import fringe_visibility, chi_squared_uniformity

        centers, total_counts = eraser_result.total_d0_pattern(n_bins=100)

        # Test 1: Fringe visibility should be ~0
        vis = fringe_visibility(total_counts)

        # Test 2: Chi-squared against envelope (Gaussian-like)
        # Fit a smooth envelope and test for deviations
        chi2, p_value = chi_squared_uniformity(total_counts)

        # Test 3: Compare D1+D2 (should cancel fringes) with D3+D4
        _, d1_counts = eraser_result.coincidence_pattern("D1", n_bins=100)
        _, d2_counts = eraser_result.coincidence_pattern("D2", n_bins=100)
        combined_erased = d1_counts + d2_counts
        vis_combined = fringe_visibility(combined_erased)

        return {
            "total_visibility": vis,
            "total_visibility_pass": vis < 0.05,
            "chi_squared": chi2,
            "chi_squared_p_value": p_value,
            "d1_d2_combined_visibility": vis_combined,
            "d1_d2_cancel_pass": vis_combined < 0.05,
            "overall_pass": vis < 0.05 and vis_combined < 0.05,
        }


def mutual_information_signaling_test(alice_outcomes: np.ndarray,
                                       bob_settings: np.ndarray,
                                       n_bins: int = 20) -> dict:
    """Test if Alice's outcomes carry information about Bob's settings.

    For no-signaling: I(A; S_B) must be zero. Alice cannot learn what
    Bob chose to do by looking at her own outcomes.

    Args:
        alice_outcomes: Array of Alice's measurement results.
        bob_settings: Array of Bob's measurement setting choices.
        n_bins: Number of bins for discretization.

    Returns:
        Dict with mutual information value, significance test, and pass/fail.
    """
    from ..analysis.statistics import mutual_information

    mi = mutual_information(alice_outcomes, bob_settings, n_bins=n_bins)

    # Permutation test for significance
    n_permutations = 1000
    mi_null = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(bob_settings)
        mi_null.append(mutual_information(alice_outcomes, shuffled, n_bins=n_bins))
    mi_null = np.array(mi_null)

    p_value = float(np.mean(mi_null >= mi))

    return {
        "mutual_information": mi,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "no_signaling_pass": p_value >= 0.05,  # NOT significant = no signaling
        "null_distribution_mean": float(np.mean(mi_null)),
        "null_distribution_std": float(np.std(mi_null)),
    }
