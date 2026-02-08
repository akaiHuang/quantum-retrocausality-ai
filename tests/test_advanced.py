"""Tests for advanced experiments."""

import numpy as np
import pytest


class TestMultipartite:
    def test_ghz_mermin_violation(self):
        from src.advanced.multipartite import ghz_mermin_test
        result = ghz_mermin_test()
        assert abs(abs(result["mermin_value"]) - 4.0) < 0.01, \
            f"|Mermin value| = {abs(result['mermin_value']):.4f}, expected 4.0"
        assert result["violates"]

    def test_w_state_robustness(self):
        from src.advanced.multipartite import w_state_entanglement_analysis
        result = w_state_entanglement_analysis(3)
        assert result["w_state_robust"], "W state should remain entangled after tracing"


class TestDecoherence:
    def test_no_noise_violates_bell(self):
        from src.advanced.decoherence import bell_violation_vs_noise
        data = bell_violation_vs_noise(n_points=3)
        # First row (p=0) should violate
        assert data.iloc[0]["violates"]

    def test_full_noise_no_violation(self):
        from src.advanced.decoherence import bell_violation_vs_noise
        data = bell_violation_vs_noise(n_points=3)
        # Last row (p=1) should not violate
        assert not data.iloc[-1]["violates"]

    def test_concurrence_bell_state(self):
        from src.advanced.decoherence import concurrence
        from src.core.states import bell_state
        from src.core.density_matrix import state_to_density
        rho = state_to_density(bell_state("psi-"))
        c = concurrence(rho)
        assert abs(c - 1.0) < 0.1, f"Concurrence = {c}, expected ~1.0"


class TestSpeedupAnalysis:
    def test_deutsch_jozsa_correct(self):
        from src.advanced.speedup_analysis import deutsch_jozsa_tsvf_analysis
        for oracle in ["constant_0", "constant_1", "balanced_id", "balanced_not"]:
            result = deutsch_jozsa_tsvf_analysis(oracle)
            assert result["correct"], f"DJ failed for oracle {oracle}"

    def test_grover_finds_item(self):
        from src.advanced.speedup_analysis import grover_tsvf_analysis
        for item in [0, 1, 2, 3]:
            result = grover_tsvf_analysis(marked_item=item)
            assert result["final_probability"] > 0.9, \
                f"Grover failed for item {item}: p={result['final_probability']}"
