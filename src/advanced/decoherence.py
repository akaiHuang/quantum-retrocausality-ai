"""Decoherence effects on retrocausal correlations.

Studies how noise destroys quantum correlations and at what point
Bell inequality violations disappear.
"""

import numpy as np
import pandas as pd
from ..core.states import bell_state
from ..core.density_matrix import state_to_density, partial_trace, von_neumann_entropy
from ..core.operators import PAULI
from ..analysis.bell_inequality import qm_singlet_correlation


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply depolarizing noise to a density matrix.

    rho' = (1-p) * rho + p * I/d

    Args:
        rho: Input density matrix.
        p: Error rate in [0, 1]. p=0: no noise, p=1: maximally mixed.

    Returns:
        Noisy density matrix.
    """
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d, dtype=complex) / d


def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply single-qubit amplitude damping to a 2-qubit state on qubit 1.

    Models energy relaxation (T1 decay).

    Args:
        rho: 4x4 density matrix of 2-qubit system.
        gamma: Damping rate in [0, 1].

    Returns:
        Damped density matrix.
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

    # Apply to qubit 1: K_i x I
    I2 = np.eye(2, dtype=complex)
    rho_out = np.zeros_like(rho)
    for K in [K0, K1]:
        full_K = np.kron(K, I2)
        rho_out += full_K @ rho @ full_K.conj().T
    return rho_out


def bell_violation_vs_noise(noise_type: str = "depolarizing",
                             n_points: int = 30) -> pd.DataFrame:
    """Compute CHSH violation as a function of noise level.

    The Bell violation disappears at a critical noise threshold.
    For depolarizing noise on the singlet: violation vanishes at p ~ 0.293.

    Args:
        noise_type: "depolarizing" or "amplitude_damping".
        n_points: Number of noise levels to evaluate.

    Returns:
        DataFrame with columns [noise_level, chsh_value, violates].
    """
    psi = bell_state("psi-")  # singlet state
    rho_pure = state_to_density(psi)

    noise_levels = np.linspace(0, 1, n_points)
    rows = []

    for p in noise_levels:
        if noise_type == "depolarizing":
            rho_noisy = depolarizing_channel(rho_pure, p)
        elif noise_type == "amplitude_damping":
            rho_noisy = amplitude_damping_channel(rho_pure, p)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Compute CHSH value analytically from the noisy state
        # For the singlet: E(a,b) = -Tr(rho * (sigma_a x sigma_b))
        # Optimal settings: a=0, a'=pi/2, b=pi/4, b'=3pi/4
        def noisy_correlation(a: float, b: float) -> float:
            # sigma_a = cos(a)*Z + sin(a)*X
            sigma_a = np.cos(a) * PAULI["Z"] + np.sin(a) * PAULI["X"]
            sigma_b = np.cos(b) * PAULI["Z"] + np.sin(b) * PAULI["X"]
            op = np.kron(sigma_a, sigma_b)
            return float(np.real(np.trace(rho_noisy @ op)))

        a, b = 0.0, np.pi / 4
        a_p, b_p = np.pi / 2, 3 * np.pi / 4

        S = (noisy_correlation(a, b) - noisy_correlation(a, b_p) +
             noisy_correlation(a_p, b) + noisy_correlation(a_p, b_p))

        rows.append({
            "noise_level": float(p),
            "chsh_value": float(S),
            "violates": abs(S) > 2.0,
        })

    return pd.DataFrame(rows)


def concurrence(rho: np.ndarray) -> float:
    """Compute the concurrence of a 2-qubit density matrix.

    Concurrence measures entanglement: C=0 means separable, C=1 means
    maximally entangled. Used to find "entanglement sudden death" threshold.

    Args:
        rho: 4x4 density matrix.

    Returns:
        Concurrence in [0, 1].
    """
    sigma_y = PAULI["Y"]
    rho_tilde = np.kron(sigma_y, sigma_y) @ rho.conj() @ np.kron(sigma_y, sigma_y)
    R = rho @ rho_tilde
    eigenvalues = np.sqrt(np.maximum(np.linalg.eigvals(R).real, 0))
    eigenvalues = np.sort(eigenvalues)[::-1]
    return float(max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3]))


def entanglement_vs_noise(noise_type: str = "depolarizing",
                           n_points: int = 50) -> pd.DataFrame:
    """Track entanglement (concurrence) as noise increases.

    Identifies the "sudden death" threshold where entanglement vanishes.

    Args:
        noise_type: "depolarizing" or "amplitude_damping".
        n_points: Number of noise levels.

    Returns:
        DataFrame with [noise_level, concurrence, entropy].
    """
    psi = bell_state("psi-")
    rho_pure = state_to_density(psi)

    noise_levels = np.linspace(0, 1, n_points)
    rows = []

    for p in noise_levels:
        if noise_type == "depolarizing":
            rho_noisy = depolarizing_channel(rho_pure, p)
        else:
            rho_noisy = amplitude_damping_channel(rho_pure, p)

        c = concurrence(rho_noisy)
        rho_a = partial_trace(rho_noisy, [0], [2, 2])
        s = von_neumann_entropy(rho_a)

        rows.append({
            "noise_level": float(p),
            "concurrence": c,
            "subsystem_entropy": s,
        })

    return pd.DataFrame(rows)
