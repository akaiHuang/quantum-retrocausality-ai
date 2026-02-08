"""Tlalpan Interpretation phase-transition simulation.

Reference: "Time Symmetry, Retrocausality, and Emergent Collapse" (2025),
arXiv:2508.19301.

The Tlalpan Interpretation proposes that wavefunction collapse is a
PHASE TRANSITION: as amplification (measurement coupling) increases,
the system undergoes a sharp transition from quantum superposition
to classical definite outcomes.

Predictions:
- Below threshold: interference persists (weak measurement regime)
- At threshold: sharp transition (phase transition)
- Above threshold: classical outcomes (strong measurement)

This module simulates the transition and looks for the predicted
sharpness of the critical threshold.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm
from ..core.density_matrix import purity


def measurement_amplification_model(system_dim: int = 2,
                                      apparatus_dim: int = 20,
                                      coupling: float = 0.1,
                                      n_stages: int = 1) -> dict:
    """Model measurement as system-apparatus coupling with amplification.

    The system (qubit) couples to a measurement apparatus (harmonic oscillator)
    via a von Neumann interaction: H_int = g * sigma_z x p_apparatus.

    After n_stages of amplification, the apparatus state becomes increasingly
    correlated with the system state, leading to decoherence.

    Args:
        system_dim: Dimension of quantum system (2 for qubit).
        apparatus_dim: Dimension of apparatus Hilbert space.
        coupling: Coupling strength g.
        n_stages: Number of amplification stages.

    Returns:
        Dict with system state after measurement, coherence, and purity.
    """
    # System in superposition: (|0> + |1>)/sqrt(2)
    psi_system = np.array([1, 1], dtype=complex) / np.sqrt(2)

    # Apparatus in ground state
    psi_apparatus = np.zeros(apparatus_dim, dtype=complex)
    psi_apparatus[0] = 1.0

    # Total initial state
    psi_total = np.kron(psi_system, psi_apparatus)
    total_dim = system_dim * apparatus_dim

    # Interaction Hamiltonian: sigma_z x p
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    # Momentum operator for apparatus (truncated)
    p_app = np.zeros((apparatus_dim, apparatus_dim), dtype=complex)
    for i in range(apparatus_dim - 1):
        p_app[i, i + 1] = -1j * np.sqrt((i + 1) / 2)
        p_app[i + 1, i] = 1j * np.sqrt((i + 1) / 2)

    H_int = coupling * np.kron(sigma_z, p_app)

    # Apply interaction for each stage
    for _ in range(n_stages):
        U = expm(-1j * H_int)
        psi_total = U @ psi_total

    # Compute reduced system density matrix
    rho_total = np.outer(psi_total, psi_total.conj())
    rho_system = np.zeros((system_dim, system_dim), dtype=complex)
    for i in range(system_dim):
        for j in range(system_dim):
            for k in range(apparatus_dim):
                rho_system[i, j] += rho_total[
                    i * apparatus_dim + k, j * apparatus_dim + k]

    # Coherence: |rho_01| (off-diagonal element)
    coherence = float(abs(rho_system[0, 1]))
    p = purity(rho_system)

    return {
        "coupling": coupling,
        "n_stages": n_stages,
        "apparatus_dim": apparatus_dim,
        "coherence": coherence,
        "purity": p,
        "rho_system": rho_system,
    }


def interference_visibility_vs_amplification(
        coupling_range: np.ndarray | None = None,
        n_stages_list: list[int] | None = None,
        apparatus_dim: int = 30) -> pd.DataFrame:
    """Sweep coupling strength and amplification stages.

    Track interference visibility (coherence) at each point.
    The Tlalpan Interpretation predicts a SHARP drop (phase transition)
    rather than a gradual decay.

    Args:
        coupling_range: Array of coupling strengths.
        n_stages_list: List of amplification stage counts.
        apparatus_dim: Dimension of apparatus.

    Returns:
        DataFrame with [coupling, n_stages, coherence, purity].
    """
    if coupling_range is None:
        coupling_range = np.linspace(0.01, 2.0, 30)
    if n_stages_list is None:
        n_stages_list = [1, 2, 3, 5]

    rows = []
    for n_stages in n_stages_list:
        for g in coupling_range:
            result = measurement_amplification_model(
                coupling=g, n_stages=n_stages, apparatus_dim=apparatus_dim)
            rows.append({
                "coupling": float(g),
                "n_stages": n_stages,
                "coherence": result["coherence"],
                "purity": result["purity"],
            })

    return pd.DataFrame(rows)


def finite_size_scaling(apparatus_dims: list[int] | None = None,
                         coupling_range: np.ndarray | None = None) -> pd.DataFrame:
    """Standard finite-size scaling analysis.

    If collapse is truly a phase transition, the critical coupling
    should converge as apparatus_dim -> infinity.

    Args:
        apparatus_dims: List of apparatus dimensions to test.
        coupling_range: Range of coupling strengths.

    Returns:
        DataFrame with scaling data.
    """
    if apparatus_dims is None:
        apparatus_dims = [10, 20, 40, 60]
    if coupling_range is None:
        coupling_range = np.linspace(0.01, 2.0, 30)

    rows = []
    for dim in apparatus_dims:
        for g in coupling_range:
            result = measurement_amplification_model(
                coupling=g, n_stages=1, apparatus_dim=dim)
            rows.append({
                "apparatus_dim": dim,
                "coupling": float(g),
                "coherence": result["coherence"],
                "purity": result["purity"],
            })

    return pd.DataFrame(rows)
