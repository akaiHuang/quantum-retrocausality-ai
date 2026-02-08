"""Quantum state preparation functions.

Provides factory functions for common entangled states used
in retrocausality experiments: Bell states, SPDC pairs, GHZ, and W states.
"""

import numpy as np


def bell_state(label: str = "phi+") -> np.ndarray:
    """Create one of the four Bell states.

    Args:
        label: One of "phi+", "phi-", "psi+", "psi-".

    Returns:
        4-element complex state vector in the computational basis {|00>, |01>, |10>, |11>}.
    """
    s = 1.0 / np.sqrt(2)
    states = {
        "phi+": np.array([s, 0, 0, s], dtype=complex),       # (|00> + |11>) / sqrt(2)
        "phi-": np.array([s, 0, 0, -s], dtype=complex),      # (|00> - |11>) / sqrt(2)
        "psi+": np.array([0, s, s, 0], dtype=complex),       # (|01> + |10>) / sqrt(2)
        "psi-": np.array([0, s, -s, 0], dtype=complex),      # (|01> - |10>) / sqrt(2)
    }
    if label not in states:
        raise ValueError(f"Unknown Bell state label '{label}'. Use: {list(states)}")
    return states[label]


def spdc_entangled_pair(phase: float = 0.0) -> np.ndarray:
    """Simulate SPDC (spontaneous parametric down-conversion) output.

    Creates a path-entangled signal-idler pair:
        |psi> = (|upper,lower> + e^{i*phase} |lower,upper>) / sqrt(2)

    In the computational basis {|UU>, |UL>, |LU>, |LL>}:
        |psi> = (|01> + e^{i*phase} |10>) / sqrt(2)

    This is the photon-pair state used in the Kim et al. (1999) quantum eraser.

    Args:
        phase: Relative phase between the two paths.

    Returns:
        4-element complex state vector.
    """
    s = 1.0 / np.sqrt(2)
    return np.array([0, s, s * np.exp(1j * phase), 0], dtype=complex)


def ghz_state(n_qubits: int = 3) -> np.ndarray:
    """Create an n-qubit GHZ (Greenberger-Horne-Zeilinger) state.

    |GHZ> = (|00...0> + |11...1>) / sqrt(2)

    GHZ states exhibit all-or-nothing nonlocality (Mermin inequality),
    which poses a stronger challenge for retrocausal models than Bell states.

    Args:
        n_qubits: Number of qubits (>= 2).

    Returns:
        Complex state vector of dimension 2^n.
    """
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)       # |00...0>
    state[-1] = 1.0 / np.sqrt(2)      # |11...1>
    return state


def w_state(n_qubits: int = 3) -> np.ndarray:
    """Create an n-qubit W state.

    |W> = (|100...0> + |010...0> + ... + |000...1>) / sqrt(n)

    W states have distributed entanglement robust against particle loss,
    posing different challenges for retrocausal models than GHZ states.

    Args:
        n_qubits: Number of qubits (>= 2).

    Returns:
        Complex state vector of dimension 2^n.
    """
    if n_qubits < 2:
        raise ValueError("W state requires at least 2 qubits")
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    s = 1.0 / np.sqrt(n_qubits)
    for k in range(n_qubits):
        index = 1 << (n_qubits - 1 - k)  # single 1-bit at position k
        state[index] = s
    return state


# --- Single-qubit basis states ---

ZERO = np.array([1, 0], dtype=complex)  # |0>
ONE = np.array([0, 1], dtype=complex)   # |1>
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)   # |+>
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)  # |->
