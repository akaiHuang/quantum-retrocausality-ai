"""Density matrix utilities.

Partial trace is the mathematical core of every no-signaling proof:
rho_A = Tr_B(rho_AB) must be independent of operations on B.
"""

import numpy as np
from scipy.linalg import sqrtm, logm


def state_to_density(state: np.ndarray) -> np.ndarray:
    """Convert a pure state vector to a density matrix.

    Args:
        state: 1-D complex array (ket).

    Returns:
        |psi><psi| density matrix.
    """
    v = np.asarray(state, dtype=complex).ravel()
    return np.outer(v, v.conj())


def partial_trace(rho: np.ndarray, keep: list[int],
                  dims: list[int]) -> np.ndarray:
    """Compute the partial trace of a density matrix.

    This is the KEY function for no-signaling verification:
    rho_A = Tr_B(rho_AB) must be independent of operations on B.

    Args:
        rho: Density matrix of the composite system.
        keep: List of subsystem indices to KEEP (0-indexed).
        dims: List of dimensions of each subsystem. Product must equal rho.shape[0].

    Returns:
        Reduced density matrix over the kept subsystems.

    Example:
        For a 2-qubit system: partial_trace(rho, keep=[0], dims=[2, 2])
        returns the reduced density matrix of qubit 0.
    """
    n = len(dims)
    rho = np.asarray(rho, dtype=complex)

    # Reshape into tensor with indices for each subsystem (ket and bra)
    rho_tensor = rho.reshape(dims + dims)

    # Determine which subsystems to trace out
    trace_out = sorted(set(range(n)) - set(keep))

    # Trace out subsystems from highest index to lowest (to preserve indexing)
    for i in sorted(trace_out, reverse=True):
        # Trace axis i (ket) with axis i+n_remaining (bra)
        n_remaining = rho_tensor.ndim // 2
        rho_tensor = np.trace(rho_tensor, axis1=i, axis2=i + n_remaining)

    # Reshape back to 2D matrix
    kept_dim = int(np.prod([dims[k] for k in keep]))
    return rho_tensor.reshape(kept_dim, kept_dim)


def purity(rho: np.ndarray) -> float:
    """Compute Tr(rho^2).

    Pure state: purity = 1. Maximally mixed: purity = 1/d.

    Args:
        rho: Density matrix.

    Returns:
        Purity value in [1/d, 1].
    """
    rho = np.asarray(rho, dtype=complex)
    return np.real(np.trace(rho @ rho))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute the von Neumann entropy S(rho) = -Tr(rho * log(rho)).

    For pure states S = 0. For bipartite pure states, S of a subsystem
    measures entanglement.

    Args:
        rho: Density matrix.

    Returns:
        Entropy in nats (natural log).
    """
    rho = np.asarray(rho, dtype=complex)
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out zero/negative eigenvalues from numerical noise
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Quantum fidelity between two density matrices.

    F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2

    F = 1 iff rho == sigma.

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Fidelity in [0, 1].
    """
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    sqrt_rho = sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = sqrtm(product)
    return float(np.real(np.trace(sqrt_product)) ** 2)


def apply_operation_on_subsystem(rho: np.ndarray, operation: np.ndarray,
                                  subsystem: int,
                                  dims: list[int]) -> np.ndarray:
    """Apply a unitary or channel operation on one subsystem of a composite state.

    For unitary U on subsystem k:
        rho' = (I x...x U x...x I) rho (I x...x U^dag x...x I)

    This is used to verify no-signaling: after this operation,
    the partial trace over the OTHER subsystems should be unchanged.

    Args:
        rho: Density matrix of composite system.
        operation: Unitary matrix acting on the specified subsystem.
        subsystem: Index of the subsystem (0-indexed).
        dims: Dimensions of each subsystem.

    Returns:
        Transformed density matrix.
    """
    n = len(dims)
    op = np.asarray(operation, dtype=complex)

    # Build full operator: I x ... x U x ... x I
    ops = [np.eye(d, dtype=complex) for d in dims]
    ops[subsystem] = op
    full_op = ops[0]
    for o in ops[1:]:
        full_op = np.kron(full_op, o)

    return full_op @ rho @ full_op.conj().T
