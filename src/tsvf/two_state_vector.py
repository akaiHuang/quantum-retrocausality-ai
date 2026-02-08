"""Two-State Vector Formalism (TSVF) core implementation.

Reference: Aharonov, Bergmann, Lebowitz (1964), Phys. Rev. B 134, 1410-1416.
Updated review: Aharonov & Vaidman, arXiv:quant-ph/0105101 (2001).

In the TSVF, a quantum system at time t is described by BOTH:
- A forward-evolving state |psi(t)> prepared at t_i (pre-selection)
- A backward-evolving state <phi(t)| post-selected at t_f (post-selection)

The two-state vector is denoted <phi| ... |psi>. This formalism:
- Is empirically equivalent to standard QM
- Provides a time-symmetric description
- Naturally produces weak values and the ABL rule
- Gives a retrocausal interpretation: the future measurement affects
  what we can say about the system at intermediate times
"""

import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass


@dataclass
class TwoStateVectorSnapshot:
    """A two-state vector at a specific time."""
    time: float
    forward_state: np.ndarray   # |psi(t)>
    backward_state: np.ndarray  # |phi(t)> (as a ket; bra is conjugate)
    overlap: complex            # <phi(t)|psi(t)>


class TwoStateVector:
    """Represents a quantum system described by both a pre-selected and
    a post-selected state.

    The two-state vector <phi| ... |psi> fully describes a pre-and-post-selected
    quantum system. At intermediate time t:
        |psi(t)> = U(t, t_i) |psi>
        <phi(t)| = <phi| U(t_f, t)

    where U(t2, t1) = exp(-i H (t2 - t1) / hbar).
    """

    def __init__(self, pre_state: np.ndarray, post_state: np.ndarray,
                 hamiltonian: np.ndarray | None = None, hbar: float = 1.0):
        """
        Args:
            pre_state: |psi> -- the prepared (pre-selected) state (ket).
            post_state: |phi> -- the post-selected state (ket).
            hamiltonian: H for time evolution U(t) = exp(-iHt/hbar).
                         If None, the system is static (H = 0).
            hbar: Reduced Planck constant (default 1 for natural units).
        """
        self.pre = np.asarray(pre_state, dtype=complex).ravel()
        self.post = np.asarray(post_state, dtype=complex).ravel()
        self.dim = len(self.pre)
        self.H = (np.asarray(hamiltonian, dtype=complex)
                  if hamiltonian is not None
                  else np.zeros((self.dim, self.dim), dtype=complex))
        self.hbar = hbar

        # Normalize
        self.pre = self.pre / np.linalg.norm(self.pre)
        self.post = self.post / np.linalg.norm(self.post)

    def _unitary(self, dt: float) -> np.ndarray:
        """Time evolution operator U(dt) = exp(-i H dt / hbar)."""
        return expm(-1j * self.H * dt / self.hbar)

    def forward_evolve(self, t: float, t_i: float = 0.0) -> np.ndarray:
        """Evolve |psi> forward from t_i to time t.

        |psi(t)> = U(t - t_i) |psi>

        Args:
            t: Target time.
            t_i: Preparation time (default 0).

        Returns:
            State vector |psi(t)>.
        """
        U = self._unitary(t - t_i)
        return U @ self.pre

    def backward_evolve(self, t: float, t_f: float = 1.0) -> np.ndarray:
        """Evolve |phi> backward from t_f to time t.

        <phi(t)| = <phi| U(t_f - t), so |phi(t)> = U^dag(t_f - t) |phi>

        Args:
            t: Target time.
            t_f: Post-selection time (default 1).

        Returns:
            State vector |phi(t)> (as a ket).
        """
        U_dag = self._unitary(t_f - t).conj().T
        return U_dag @ self.post

    def at_time(self, t: float, t_i: float = 0.0,
                t_f: float = 1.0) -> TwoStateVectorSnapshot:
        """Get the full two-state vector at an intermediate time.

        Args:
            t: Intermediate time.
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            TwoStateVectorSnapshot with both states and their overlap.
        """
        fwd = self.forward_evolve(t, t_i)
        bwd = self.backward_evolve(t, t_f)
        overlap = np.vdot(bwd, fwd)  # <phi(t)|psi(t)>
        return TwoStateVectorSnapshot(
            time=t,
            forward_state=fwd,
            backward_state=bwd,
            overlap=overlap,
        )

    def overlap(self, t_i: float = 0.0, t_f: float = 1.0) -> complex:
        """Compute <phi|U(t_f - t_i)|psi>.

        |<phi|U|psi>|^2 is the probability that this pre/post-selection pair
        occurs in an experiment.

        Args:
            t_i: Pre-selection time.
            t_f: Post-selection time.

        Returns:
            Complex amplitude.
        """
        U = self._unitary(t_f - t_i)
        return complex(np.vdot(self.post, U @ self.pre))

    def post_selection_probability(self, t_i: float = 0.0,
                                     t_f: float = 1.0) -> float:
        """Probability that the post-selection succeeds.

        P = |<phi|U(t_f - t_i)|psi>|^2

        Returns:
            Probability in [0, 1].
        """
        return float(abs(self.overlap(t_i, t_f)) ** 2)

    def time_evolution_data(self, n_times: int = 100,
                            t_i: float = 0.0,
                            t_f: float = 1.0) -> list[TwoStateVectorSnapshot]:
        """Generate data for animating both state vectors evolving in time.

        Forward state moves from t_i to t_f.
        Backward state moves from t_f to t_i.

        Args:
            n_times: Number of time points.
            t_i: Start time.
            t_f: End time.

        Returns:
            List of TwoStateVectorSnapshot at each time.
        """
        times = np.linspace(t_i, t_f, n_times)
        return [self.at_time(t, t_i, t_f) for t in times]
