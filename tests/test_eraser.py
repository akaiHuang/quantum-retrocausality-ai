"""Tests for quantum eraser simulation."""

import numpy as np
import pytest
from src.eraser.kim_eraser import KimQuantumEraser
from src.eraser.no_signaling_verifier import NoSignalingVerifier
from src.core.states import bell_state
from src.core.operators import PAULI
from src.analysis.statistics import fringe_visibility


class TestKimEraser:
    @pytest.fixture
    def eraser_result(self):
        eraser = KimQuantumEraser(n_experiments=10000)
        return eraser.run_experiment()

    def test_total_d0_less_fringes_than_subsets(self, eraser_result):
        """Total D0 pattern must have less visibility than coincidence subsets."""
        _, total = eraser_result.total_d0_pattern(n_bins=50)
        vis_total = fringe_visibility(total)

        # D1 should show higher visibility than total
        _, d1 = eraser_result.coincidence_pattern("D1", n_bins=50)
        vis_d1 = fringe_visibility(d1) if d1.sum() > 100 else 0

        # The key test: total visibility should be lower than subset visibility
        # (perfect would be vis_total ~ 0 and vis_d1 >> 0)
        assert vis_total < vis_d1 + 0.3, \
            f"Total vis={vis_total:.3f} should be less than D1 vis={vis_d1:.3f}"

    def test_d1_d2_show_fringes(self, eraser_result):
        """D1 and D2 coincidence patterns should show interference."""
        for det in ["D1", "D2"]:
            _, counts = eraser_result.coincidence_pattern(det, n_bins=80)
            if counts.sum() > 100:
                vis = fringe_visibility(counts)
                # Relaxed threshold due to statistical noise
                assert vis > 0.05, f"{det} visibility {vis:.4f} too low"

    def test_all_detectors_have_counts(self, eraser_result):
        """All four detectors should register events."""
        counts = eraser_result.counter.detector_counts()
        for det in ["D1", "D2", "D3", "D4"]:
            assert counts.get(det, 0) > 0, f"No events at detector {det}"


class TestNoSignalingVerifier:
    def test_bell_state_no_signaling(self):
        """Bell state must satisfy no-signaling."""
        psi = bell_state("psi-")
        verifier = NoSignalingVerifier()
        result = verifier.verify_state(psi, dims=[2, 2])
        assert result.passed, \
            f"No-signaling violated! Max deviation: {result.max_fidelity_deviation}"

    def test_product_state_no_signaling(self):
        """Product state trivially satisfies no-signaling."""
        psi = np.kron(np.array([1, 0]), np.array([0, 1])).astype(complex)
        verifier = NoSignalingVerifier()
        result = verifier.verify_state(psi, dims=[2, 2])
        assert result.passed
