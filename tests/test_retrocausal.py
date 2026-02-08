"""Tests for retrocausal models."""

import numpy as np
import pytest
from src.retrocausal.zigzag_model import ZigZagModel, ClassicalLocalModel
from src.retrocausal.boundary_value import BoundaryValueModel
from src.retrocausal.bell_test import BellTestComparator
from src.analysis.bell_inequality import qm_singlet_correlation


class TestZigZagModel:
    @pytest.fixture
    def model(self):
        return ZigZagModel(n_trials=50000)

    def test_correlation_matches_qm(self, model):
        """E(a,b) should approximate -cos(a-b)."""
        for a, b in [(0, np.pi/4), (0, np.pi/2), (np.pi/3, np.pi/6)]:
            corr = model.correlation(a, b)
            expected = qm_singlet_correlation(a, b)
            assert abs(corr - expected) < 0.1, \
                f"E({a:.2f},{b:.2f}) = {corr:.3f}, expected {expected:.3f}"

    def test_bell_violation(self, model):
        """CHSH value should exceed 2 (violate Bell inequality)."""
        result = model.run_bell_test()
        assert abs(result.chsh_value) > 2.0, \
            f"CHSH S = {result.chsh_value:.3f}, expected > 2"

    def test_locality(self, model):
        """Verify locality property."""
        result = model.verify_locality()
        assert result["locality_verified"]


class TestClassicalModel:
    def test_bell_respects_bound(self):
        """Classical model should NOT violate Bell inequality."""
        model = ClassicalLocalModel(n_trials=50000)
        result = model.run_bell_test()
        assert abs(result.chsh_value) < 2.2, \
            f"Classical CHSH S = {result.chsh_value:.3f}, expected <= 2"


class TestBoundaryValueModel:
    def test_correlation_matches_qm(self):
        model = BoundaryValueModel(n_paths=30000)
        for b in [np.pi/4, np.pi/2, np.pi]:
            corr = model.singlet_correlation(0, b)
            expected = qm_singlet_correlation(0, b)
            assert abs(corr - expected) < 0.15, \
                f"E(0,{b:.2f}) = {corr:.3f}, expected {expected:.3f}"


class TestBellComparator:
    def test_chsh_values(self):
        comparator = BellTestComparator(n_trials=20000)
        chsh = comparator.compute_chsh_values()

        # QM should give ~2*sqrt(2) in absolute value
        assert abs(abs(chsh["QM (analytical)"]) - 2 * np.sqrt(2)) < 0.01

        # Classical should be <= 2
        assert abs(chsh["Classical Local"]) < 2.3

        # Retrocausal should be > 2
        assert abs(chsh["ZigZag Retrocausal"]) > 1.8
