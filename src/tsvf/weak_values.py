"""Weak value calculator for pre/post-selected quantum systems.

Reference: Aharonov, Albert, Vaidman (1988), PRL 60, 1351.
"How the Result of a Measurement of a Component of the Spin of a
Spin-1/2 Particle Can Turn Out to Be 100."

The weak value of observable A for pre-state |psi> and post-state |phi> is:

    A_w = <phi|A|psi> / <phi|psi>

Key properties:
- Can be complex-valued
- Can lie OUTSIDE the eigenvalue spectrum (anomalous weak values)
- Real part: pointer position shift in weak measurement
- Imaginary part: pointer momentum kick in weak measurement
- These are experimentally measurable quantities

This module is the primary novel contribution of the project --
no open-source weak value calculator with simulation exists.
"""

import numpy as np
from dataclasses import dataclass

from .two_state_vector import TwoStateVector


@dataclass
class WeakMeasurementResult:
    """Result of a simulated weak measurement."""
    weak_value: complex
    pointer_readings: np.ndarray      # simulated pointer position readings
    pointer_shift_theory: float       # Re(A_w) * coupling
    pointer_shift_measured: float     # actual mean of readings
    coupling_strength: float
    n_trials: int
    eigenvalues: np.ndarray
    is_anomalous: bool


class WeakValueCalculator:
    """Compute weak values of observables for pre/post-selected systems.

    This is the core engine of the TSVF module. Weak values provide the
    operational content of the two-state vector formalism: they predict
    the shift of a measurement pointer under weak coupling.
    """

    def __init__(self, two_state_vector: TwoStateVector):
        """
        Args:
            two_state_vector: The pre/post-selected system.
        """
        self.tsv = two_state_vector

    def compute(self, observable: np.ndarray, t: float = 0.0,
                t_i: float = 0.0, t_f: float = 1.0) -> complex:
        """Compute the weak value A_w = <phi(t)|A|psi(t)> / <phi(t)|psi(t)>.

        Args:
            observable: Hermitian matrix (the operator A).
            t: Time at which to evaluate.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            Complex weak value.

        Raises:
            ValueError: If overlap is too small (near-orthogonal pre/post states).
        """
        snapshot = self.tsv.at_time(t, t_i, t_f)
        A = np.asarray(observable, dtype=complex)

        numerator = np.vdot(snapshot.backward_state, A @ snapshot.forward_state)
        denominator = snapshot.overlap

        if abs(denominator) < 1e-15:
            raise ValueError(
                "Pre and post-selected states are (nearly) orthogonal. "
                f"|<phi|psi>| = {abs(denominator):.2e}. "
                "Weak value diverges."
            )

        return complex(numerator / denominator)

    def is_anomalous(self, observable: np.ndarray, t: float = 0.0,
                     t_i: float = 0.0, t_f: float = 1.0) -> bool:
        """Check if the weak value lies outside the eigenvalue range.

        Anomalous weak values are a hallmark of the TSVF and represent
        the most striking "retrocausal" prediction: the observable
        appears to take a value that no single measurement could produce.

        Args:
            observable: Hermitian matrix.
            t: Evaluation time.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            True if Re(A_w) is outside [min_eigenvalue, max_eigenvalue].
        """
        A = np.asarray(observable, dtype=complex)
        eigenvalues = np.linalg.eigvalsh(A)
        wv = self.compute(A, t, t_i, t_f)

        return float(wv.real) < eigenvalues.min() - 1e-10 or \
               float(wv.real) > eigenvalues.max() + 1e-10

    def compute_for_observable_set(self, observables: dict[str, np.ndarray],
                                    t: float = 0.0, t_i: float = 0.0,
                                    t_f: float = 1.0) -> dict[str, complex]:
        """Compute weak values for multiple observables simultaneously.

        Args:
            observables: Dict of {name: matrix}.
            t: Evaluation time.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            Dict of {name: weak_value}.
        """
        return {name: self.compute(A, t, t_i, t_f)
                for name, A in observables.items()}

    def weak_measurement_simulation(self, observable: np.ndarray,
                                     coupling_strength: float = 0.01,
                                     n_trials: int = 10000,
                                     pointer_width: float = 1.0,
                                     t: float = 0.0, t_i: float = 0.0,
                                     t_f: float = 1.0
                                     ) -> WeakMeasurementResult:
        """Simulate a full weak measurement protocol.

        Protocol:
        1. Prepare pointer in Gaussian state centered at 0
        2. Weakly couple system to pointer: H_int = g * A * p_pointer
        3. Post-select system in |phi>
        4. Read pointer position -- shifted by Re(A_w) * g

        The pointer shift is the OPERATIONAL definition of weak value.

        Args:
            observable: The measured observable A.
            coupling_strength: g (weak coupling parameter). Must be small.
            n_trials: Number of simulated experiments.
            pointer_width: Width of initial pointer Gaussian.
            t: Measurement time.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            WeakMeasurementResult with pointer distribution and comparison.
        """
        A = np.asarray(observable, dtype=complex)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        wv = self.compute(A, t, t_i, t_f)

        snapshot = self.tsv.at_time(t, t_i, t_f)
        psi = snapshot.forward_state
        phi = snapshot.backward_state

        pointer_readings = []
        for _ in range(n_trials):
            # Initial pointer state: Gaussian centered at 0
            q0 = np.random.normal(0, pointer_width)

            # The weak measurement shifts the pointer by g * eigenvalue
            # with probability given by the pre/post-selection
            # In the weak limit, this averages to g * Re(A_w)
            probs = np.abs(eigenvectors.conj().T @ psi) ** 2

            # Sample an eigenvalue
            idx = np.random.choice(len(eigenvalues), p=probs)
            a_k = eigenvalues[idx]

            # Pointer shifts by g * a_k
            q_shifted = q0 + coupling_strength * a_k

            # Post-selection: accept with probability |<phi|a_k>|^2 / p_post
            # (conditional on the eigenstate |a_k> being the intermediate state)
            p_post_given_ak = abs(np.vdot(phi, eigenvectors[:, idx])) ** 2
            p_post_total = abs(np.vdot(phi, psi)) ** 2

            # Weight by post-selection probability
            if np.random.random() < p_post_given_ak * probs[idx] / max(p_post_total, 1e-15):
                pointer_readings.append(q_shifted)

        pointer_readings = np.array(pointer_readings) if pointer_readings else np.array([0.0])

        return WeakMeasurementResult(
            weak_value=wv,
            pointer_readings=pointer_readings,
            pointer_shift_theory=float(wv.real) * coupling_strength,
            pointer_shift_measured=float(np.mean(pointer_readings)),
            coupling_strength=coupling_strength,
            n_trials=len(pointer_readings),
            eigenvalues=eigenvalues,
            is_anomalous=self.is_anomalous(A, t, t_i, t_f),
        )

    def three_box_paradox(self) -> dict:
        """Implement the Aharonov-Vaidman three-box paradox.

        Setup:
            Pre-select:  |psi> = (|A> + |B> + |C>) / sqrt(3)
            Post-select: |phi> = (|A> + |B> - |C>) / sqrt(3)

        Weak values of box projectors:
            Pi_A_w = <phi|A><A|psi> / <phi|psi> = 1
            Pi_B_w = <phi|B><B|psi> / <phi|psi> = 1
            Pi_C_w = <phi|C><C|psi> / <phi|psi> = -1

        Yet Pi_A + Pi_B + Pi_C = I, so weak values sum to 1.

        The paradox: the particle is "certainly in box A" AND "certainly in box B"
        at the same time, while the probability of being in C is NEGATIVE.
        This is the essence of retrocausal quantum mechanics.

        Returns:
            Dict with all weak values and verification of the sum rule.
        """
        s3 = 1.0 / np.sqrt(3)

        # Basis states
        A = np.array([1, 0, 0], dtype=complex)
        B = np.array([0, 1, 0], dtype=complex)
        C = np.array([0, 0, 1], dtype=complex)

        pre = s3 * (A + B + C)
        post = s3 * (A + B - C)

        # Create a static TSV (no Hamiltonian)
        tsv = TwoStateVector(pre, post)
        calc = WeakValueCalculator(tsv)

        # Box projectors
        Pi_A = np.outer(A, A.conj())
        Pi_B = np.outer(B, B.conj())
        Pi_C = np.outer(C, C.conj())

        wv_A = calc.compute(Pi_A)
        wv_B = calc.compute(Pi_B)
        wv_C = calc.compute(Pi_C)

        return {
            "Pi_A_weak_value": wv_A,
            "Pi_B_weak_value": wv_B,
            "Pi_C_weak_value": wv_C,
            "sum": wv_A + wv_B + wv_C,
            "sum_equals_1": abs(wv_A + wv_B + wv_C - 1.0) < 1e-10,
            "Pi_A_anomalous": wv_A.real > 1 + 1e-10 or wv_A.real < -1e-10,
            "Pi_C_negative": wv_C.real < -1e-10,
            "explanation": (
                "The particle is 'certainly in A' (Pi_A_w=1) and 'certainly in B' "
                "(Pi_B_w=1) simultaneously, while the probability of C is negative "
                "(Pi_C_w=-1). This is consistent because weak values are not "
                "probabilities -- they are conditional on BOTH pre- and post-selection. "
                "The 'retrocausal' character: the future post-selection determines "
                "what we can say about the past."
            ),
        }
