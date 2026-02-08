"""Multipartite entanglement retrocausal analysis.

GHZ and W states present different challenges for retrocausal models:
- GHZ: all-or-nothing nonlocality (Mermin inequality)
- W: distributed entanglement, robust against particle loss
"""

import numpy as np
from ..core.states import ghz_state, w_state
from ..core.operators import PAULI, multi_qubit_operator
from ..core.density_matrix import state_to_density, partial_trace


def ghz_mermin_test(n_qubits: int = 3, n_samples: int = 50000) -> dict:
    """Run Mermin inequality test on an n-qubit GHZ state.

    For the 3-qubit GHZ state |GHZ> = (|000> + |111>)/sqrt(2):
    M = E(XYY) + E(YXY) + E(YYX) - E(XXX)

    QM prediction: M = 4 (maximum algebraic violation)
    Classical bound: |M| <= 2

    Args:
        n_qubits: Number of qubits (currently supports 3).
        n_samples: Number of measurement samples.

    Returns:
        Dict with Mermin value and individual correlators.
    """
    if n_qubits != 3:
        raise NotImplementedError("Mermin test currently supports 3 qubits only")

    state = ghz_state(3)
    rho = state_to_density(state)

    def expectation_3qubit(bases: str) -> float:
        """Compute <O1 x O2 x O3> for basis string like 'XYY'."""
        op = np.eye(1, dtype=complex)
        for b in bases:
            op = np.kron(op, PAULI[b])
        return float(np.real(np.trace(rho @ op)))

    E_XYY = expectation_3qubit("XYY")
    E_YXY = expectation_3qubit("YXY")
    E_YYX = expectation_3qubit("YYX")
    E_XXX = expectation_3qubit("XXX")

    M = E_XYY + E_YXY + E_YYX - E_XXX

    return {
        "n_qubits": n_qubits,
        "mermin_value": M,
        "classical_bound": 2.0,
        "qm_prediction": 4.0,
        "violates": abs(M) > 2.0,
        "correlators": {
            "E(XYY)": E_XYY,
            "E(YXY)": E_YXY,
            "E(YYX)": E_YYX,
            "E(XXX)": E_XXX,
        },
        "explanation": (
            "GHZ states show ALL-OR-NOTHING nonlocality: the Mermin value is "
            "exactly 4 (algebraic maximum), not just statistically above 2. "
            "A retrocausal model must explain this perfect violation."
        ),
    }


def w_state_entanglement_analysis(n_qubits: int = 3) -> dict:
    """Analyze the entanglement structure of a W state.

    W states have distributed entanglement: tracing out one qubit still
    leaves the remaining qubits entangled. This is fundamentally different
    from GHZ states (where tracing one qubit destroys all entanglement).

    Args:
        n_qubits: Number of qubits.

    Returns:
        Dict with entanglement measures.
    """
    state = w_state(n_qubits)
    rho = state_to_density(state)
    dims = [2] * n_qubits

    # Reduced density matrices after tracing one qubit
    results = {}
    for k in range(n_qubits):
        keep = [i for i in range(n_qubits) if i != k]
        rho_reduced = partial_trace(rho, keep, dims)

        from ..core.density_matrix import purity, von_neumann_entropy
        results[f"trace_out_qubit_{k}"] = {
            "purity": purity(rho_reduced),
            "entropy": von_neumann_entropy(rho_reduced),
            "still_entangled": purity(rho_reduced) < 1.0 - 1e-10,
        }

    return {
        "n_qubits": n_qubits,
        "subsystem_analysis": results,
        "w_state_robust": all(r["still_entangled"]
                              for r in results.values()),
        "explanation": (
            "W states have 'distributed' entanglement: removing one particle "
            "preserves entanglement among the rest. This is robust against "
            "particle loss, unlike GHZ states. For retrocausal models, this "
            "means the future-input dependence must be distributed across "
            "all particles, not concentrated in any single pair."
        ),
    }
