"""Tests for core quantum primitives."""

import numpy as np
import pytest
from src.core.states import bell_state, ghz_state, w_state, spdc_entangled_pair
from src.core.operators import projector, measurement_operator, beam_splitter, PAULI
from src.core.density_matrix import (
    partial_trace, state_to_density, purity, von_neumann_entropy,
    fidelity, apply_operation_on_subsystem,
)


class TestStates:
    def test_bell_states_normalized(self):
        for label in ["phi+", "phi-", "psi+", "psi-"]:
            s = bell_state(label)
            assert abs(np.linalg.norm(s) - 1.0) < 1e-10

    def test_bell_states_orthogonal(self):
        states = [bell_state(l) for l in ["phi+", "phi-", "psi+", "psi-"]]
        for i in range(4):
            for j in range(4):
                overlap = abs(np.vdot(states[i], states[j]))
                if i == j:
                    assert abs(overlap - 1.0) < 1e-10
                else:
                    assert overlap < 1e-10

    def test_ghz_state_normalized(self):
        for n in [2, 3, 4, 5]:
            s = ghz_state(n)
            assert abs(np.linalg.norm(s) - 1.0) < 1e-10

    def test_ghz_state_components(self):
        s = ghz_state(3)
        assert abs(s[0]) > 0.5  # |000>
        assert abs(s[7]) > 0.5  # |111>
        assert abs(np.sum(np.abs(s[1:7]) ** 2)) < 1e-10

    def test_w_state_normalized(self):
        for n in [2, 3, 4]:
            s = w_state(n)
            assert abs(np.linalg.norm(s) - 1.0) < 1e-10

    def test_w_state_components(self):
        s = w_state(3)
        # Should have |100>, |010>, |001> each with amplitude 1/sqrt(3)
        expected_amp = 1 / np.sqrt(3)
        assert abs(abs(s[4]) - expected_amp) < 1e-10  # |100>
        assert abs(abs(s[2]) - expected_amp) < 1e-10  # |010>
        assert abs(abs(s[1]) - expected_amp) < 1e-10  # |001>

    def test_spdc_pair_normalized(self):
        s = spdc_entangled_pair()
        assert abs(np.linalg.norm(s) - 1.0) < 1e-10


class TestOperators:
    def test_projector_idempotent(self):
        v = np.array([1, 0], dtype=complex)
        P = projector(v)
        assert np.allclose(P @ P, P)

    def test_beam_splitter_unitary(self):
        BS = beam_splitter()
        assert np.allclose(BS @ BS.conj().T, np.eye(2), atol=1e-10)

    def test_measurement_operator_projector(self):
        P = measurement_operator("Z", 0, 0, 2)
        assert np.allclose(P @ P, P, atol=1e-10)


class TestDensityMatrix:
    def test_partial_trace_bell_state(self):
        """Tracing out one qubit of a Bell state gives maximally mixed state."""
        psi = bell_state("phi+")
        rho = state_to_density(psi)
        rho_a = partial_trace(rho, keep=[0], dims=[2, 2])
        expected = np.eye(2, dtype=complex) / 2
        assert np.allclose(rho_a, expected, atol=1e-10)

    def test_partial_trace_product_state(self):
        """Tracing out one qubit of a product state gives the other qubit."""
        psi = np.kron(np.array([1, 0]), np.array([0, 1])).astype(complex)
        rho = state_to_density(psi)
        rho_a = partial_trace(rho, keep=[0], dims=[2, 2])
        expected = state_to_density(np.array([1, 0], dtype=complex))
        assert np.allclose(rho_a, expected, atol=1e-10)

    def test_purity_pure_state(self):
        rho = state_to_density(np.array([1, 0], dtype=complex))
        assert abs(purity(rho) - 1.0) < 1e-10

    def test_purity_maximally_mixed(self):
        rho = np.eye(2, dtype=complex) / 2
        assert abs(purity(rho) - 0.5) < 1e-10

    def test_entropy_pure_state(self):
        rho = state_to_density(np.array([1, 0], dtype=complex))
        assert abs(von_neumann_entropy(rho)) < 1e-10

    def test_entropy_maximally_mixed(self):
        rho = np.eye(2, dtype=complex) / 2
        assert abs(von_neumann_entropy(rho) - np.log(2)) < 1e-10

    def test_fidelity_same_state(self):
        rho = state_to_density(np.array([1, 0], dtype=complex))
        assert abs(fidelity(rho, rho) - 1.0) < 1e-10

    def test_no_signaling_bell_state(self):
        """No-signaling: rho_A unchanged by operations on B."""
        psi = bell_state("psi-")
        rho = state_to_density(psi)

        rho_a_original = partial_trace(rho, keep=[0], dims=[2, 2])

        for name, op in PAULI.items():
            rho_after = apply_operation_on_subsystem(rho, op, 1, [2, 2])
            rho_a_after = partial_trace(rho_after, keep=[0], dims=[2, 2])
            assert np.allclose(rho_a_original, rho_a_after, atol=1e-10), \
                f"No-signaling violated for operation {name}"
