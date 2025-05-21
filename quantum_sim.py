import numpy as np

"""Simple random-based quantum experiment simulation.

This module provides a lightweight replacement for the previous Qiskit-based
implementation. The function `run_quantum_experiment` generates synthetic data
without requiring any external quantum computing libraries.
"""


def run_quantum_experiment(b_measurement_basis: str):
    """Simulate a single trial of a two-qubit experiment.

    Parameters
    ----------
    b_measurement_basis : str
        Measurement basis for qubit B. Supported values are ``"Z"`` and ``"X"``.

    Returns
    -------
    tuple[int, int]
        A tuple ``(a, b)`` representing the measurement outcomes for qubits
        A and B respectively.
    """
    # Randomly choose a result for qubit A
    a = np.random.randint(2)

    if b_measurement_basis == "Z":
        # Correlated with A in Z basis
        b = a
    elif b_measurement_basis == "X":
        # Anti-correlated with A in X basis
        b = 1 - a
    else:
        # Fallback to independent random result
        b = np.random.randint(2)

    return a, b
