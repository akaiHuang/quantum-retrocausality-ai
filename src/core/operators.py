"""Quantum operators, projectors, and optical elements.

Provides measurement operators, Pauli matrices, beam splitters, and
phase shifters used across all simulation modules.
"""

import numpy as np

# --- Pauli matrices ---

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

PAULI = {"I": IDENTITY_2, "X": SIGMA_X, "Y": SIGMA_Y, "Z": SIGMA_Z}


def projector(state_vec: np.ndarray) -> np.ndarray:
    """Construct the projector |psi><psi| from a state vector.

    Args:
        state_vec: 1-D complex array (ket).

    Returns:
        Outer product matrix |psi><psi|.
    """
    v = np.asarray(state_vec, dtype=complex).ravel()
    return np.outer(v, v.conj())


def measurement_operator(basis: str, outcome: int, qubit_index: int,
                         n_qubits: int) -> np.ndarray:
    """Construct a measurement projector for a single qubit in a multi-qubit system.

    Args:
        basis: "Z", "X", or "Y".
        outcome: 0 or 1 (eigenvalue index).
        qubit_index: Which qubit to measure (0-indexed).
        n_qubits: Total number of qubits.

    Returns:
        2^n x 2^n projector matrix.
    """
    if basis == "Z":
        eigenstates = [np.array([1, 0], dtype=complex),
                       np.array([0, 1], dtype=complex)]
    elif basis == "X":
        s = 1.0 / np.sqrt(2)
        eigenstates = [np.array([s, s], dtype=complex),
                       np.array([s, -s], dtype=complex)]
    elif basis == "Y":
        s = 1.0 / np.sqrt(2)
        eigenstates = [np.array([s, 1j * s], dtype=complex),
                       np.array([s, -1j * s], dtype=complex)]
    else:
        raise ValueError(f"Unknown basis '{basis}'. Use Z, X, or Y.")

    proj_single = projector(eigenstates[outcome])

    # Tensor product: I x ... x proj x ... x I
    ops = [IDENTITY_2] * n_qubits
    ops[qubit_index] = proj_single
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def multi_qubit_operator(single_op: np.ndarray, qubit_index: int,
                         n_qubits: int) -> np.ndarray:
    """Embed a single-qubit operator into a multi-qubit Hilbert space.

    Args:
        single_op: 2x2 matrix.
        qubit_index: Which qubit it acts on (0-indexed).
        n_qubits: Total number of qubits.

    Returns:
        2^n x 2^n matrix: I x ... x op x ... x I.
    """
    ops = [IDENTITY_2] * n_qubits
    ops[qubit_index] = np.asarray(single_op, dtype=complex)
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def beam_splitter(theta: float = np.pi / 4) -> np.ndarray:
    """Beam splitter unitary matrix.

    BS(theta) = [[cos(theta), i*sin(theta)],
                 [i*sin(theta), cos(theta)]]

    theta = pi/4 gives a 50:50 beam splitter.

    Args:
        theta: Reflectivity angle.

    Returns:
        2x2 unitary matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 1j * s], [1j * s, c]], dtype=complex)


def phase_shifter(phi: float) -> np.ndarray:
    """Phase shift operator.

    PS(phi) = [[1, 0], [0, e^{i*phi}]]

    Args:
        phi: Phase angle.

    Returns:
        2x2 unitary matrix.
    """
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
