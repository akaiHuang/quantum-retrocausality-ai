"""Tests for TSVF engine."""

import numpy as np
import pytest
from src.tsvf.two_state_vector import TwoStateVector
from src.tsvf.weak_values import WeakValueCalculator
from src.tsvf.abl_rule import ABLRule
from src.core.operators import PAULI


class TestTwoStateVector:
    def test_overlap_same_state(self):
        psi = np.array([1, 0], dtype=complex)
        tsv = TwoStateVector(psi, psi)
        assert abs(abs(tsv.overlap()) - 1.0) < 1e-10

    def test_overlap_orthogonal(self):
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([0, 1], dtype=complex)
        tsv = TwoStateVector(psi, phi)
        assert abs(tsv.overlap()) < 1e-10

    def test_post_selection_probability(self):
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        tsv = TwoStateVector(psi, phi)
        assert abs(tsv.post_selection_probability() - 0.5) < 1e-10

    def test_forward_evolve_no_hamiltonian(self):
        psi = np.array([1, 0], dtype=complex)
        tsv = TwoStateVector(psi, psi)
        evolved = tsv.forward_evolve(0.5)
        assert np.allclose(evolved, psi / np.linalg.norm(psi), atol=1e-10)


class TestWeakValues:
    def test_weak_value_pre_equals_post(self):
        """When pre = post, weak value = expectation value."""
        psi = np.array([1, 0], dtype=complex)
        tsv = TwoStateVector(psi, psi)
        calc = WeakValueCalculator(tsv)
        wv = calc.compute(PAULI["Z"])
        # <0|Z|0> = 1
        assert abs(wv - 1.0) < 1e-10

    def test_three_box_paradox(self):
        """The three-box paradox: Pi_A_w = 1, Pi_B_w = 1, Pi_C_w = -1."""
        calc = WeakValueCalculator.__new__(WeakValueCalculator)
        result = calc.three_box_paradox()

        assert abs(result["Pi_A_weak_value"].real - 1.0) < 1e-10
        assert abs(result["Pi_B_weak_value"].real - 1.0) < 1e-10
        assert abs(result["Pi_C_weak_value"].real - (-1.0)) < 1e-10
        assert result["sum_equals_1"]

    def test_weak_value_anomalous_detection(self):
        """Anomalous weak value should be detected."""
        s3 = 1 / np.sqrt(3)
        pre = np.array([s3, s3, s3], dtype=complex)
        post = np.array([s3, s3, -s3], dtype=complex)
        tsv = TwoStateVector(pre, post)
        calc = WeakValueCalculator(tsv)

        # Projector onto |C>
        Pi_C = np.zeros((3, 3), dtype=complex)
        Pi_C[2, 2] = 1.0

        assert calc.is_anomalous(Pi_C)

    def test_weak_value_sum_identity(self):
        """Weak values of a complete set of projectors must sum to 1."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        phi = np.array([np.cos(0.3), np.sin(0.3)], dtype=complex)
        tsv = TwoStateVector(psi, phi)
        calc = WeakValueCalculator(tsv)

        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)

        wv0 = calc.compute(P0)
        wv1 = calc.compute(P1)
        assert abs(wv0 + wv1 - 1.0) < 1e-10


class TestABLRule:
    def test_abl_probabilities_sum_to_one(self):
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        phi = np.array([1, 0], dtype=complex)
        tsv = TwoStateVector(psi, phi)
        abl = ABLRule(tsv)

        probs = abl.all_probabilities(PAULI["Z"])
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-10

    def test_abl_equals_born_when_pre_equals_post(self):
        """ABL = Born rule when there is no post-selection effect."""
        psi = np.array([1, 0], dtype=complex)
        tsv = TwoStateVector(psi, psi)
        abl = ABLRule(tsv)

        comparison = abl.compare_with_born_rule(PAULI["Z"])
        assert comparison.max_difference < 1e-10

    def test_time_symmetry(self):
        """ABL probabilities should be time-symmetric."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        phi = np.array([np.cos(0.3), np.sin(0.3)], dtype=complex)
        tsv = TwoStateVector(psi, phi)
        abl = ABLRule(tsv)

        comparison = abl.compare_with_born_rule(PAULI["X"])
        assert comparison.time_symmetric
