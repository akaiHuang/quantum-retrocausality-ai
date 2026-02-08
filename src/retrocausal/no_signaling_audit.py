"""No-signaling audit for retrocausal models.

This is CRITICAL: retrocausal models MUST respect no-signaling.
If any model violates it, the model is WRONG (or has a bug).

Tests:
1. P(A=+1 | a) independent of b for all a, b
2. P(B=+1 | b) independent of a for all a, b
3. I(A; S_B) = 0 (mutual information between Alice's outcomes and Bob's settings)
4. I(B; S_A) = 0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


@dataclass
class AuditResult:
    """Result of a no-signaling audit for a single model."""
    model_name: str
    passed: bool
    max_marginal_deviation: float  # max |P(A|a,b1) - P(A|a,b2)| over all a,b1,b2
    mutual_info_a_sb: float        # I(Alice outcomes; Bob settings)
    mutual_info_b_sa: float        # I(Bob outcomes; Alice settings)
    details: pd.DataFrame          # per-setting-pair results
    tolerance: float


class NoSignalingAudit:
    """Comprehensive audit that every retrocausal model respects no-signaling.

    This ensures scientific honesty: if a model violates no-signaling,
    it would allow backward-in-time communication, which is forbidden.
    """

    def __init__(self, n_settings: int = 12, n_trials_per_setting: int = 20000,
                 tolerance: float = 0.03):
        """
        Args:
            n_settings: Number of setting angles to test.
            n_trials_per_setting: Trials per setting pair.
            tolerance: Maximum allowed deviation from uniform marginals.
        """
        self.n_settings = n_settings
        self.n_trials = n_trials_per_setting
        self.tolerance = tolerance

    def audit_model(self, model, model_name: str = "Unknown") -> AuditResult:
        """Full no-signaling audit for a given model.

        For each Alice setting a, sweep Bob's settings and check that
        P(A=+1 | a) remains constant regardless of b (and vice versa).

        Args:
            model: Object with methods: _sample_hidden_variables(a, b, n),
                   alice_outcome(lam, a), bob_outcome(lam, b).
                   Or any object with a .correlation(a, b) method (less detailed audit).
            model_name: Display name.

        Returns:
            AuditResult with pass/fail and detailed statistics.
        """
        alice_settings = np.linspace(0, np.pi, self.n_settings)
        bob_settings = np.linspace(0, np.pi, self.n_settings)

        rows = []
        all_alice_outcomes = []
        all_bob_settings_for_alice = []
        all_bob_outcomes = []
        all_alice_settings_for_bob = []

        max_dev = 0.0

        for a in alice_settings:
            p_alice_plus_given_b = []

            for b in bob_settings:
                # Generate data
                if hasattr(model, '_sample_hidden_variables'):
                    lam = model._sample_hidden_variables(a, b, self.n_trials)
                    A = model.alice_outcome(lam, a)
                    B = model.bob_outcome(lam, b)
                    mask = (A != 0) & (B != 0)
                    A, B = A[mask], B[mask]
                else:
                    # Fallback: simulate with random +/-1 weighted by correlation
                    corr = model.correlation(a, b)
                    n = self.n_trials
                    A = np.random.choice([-1, 1], size=n)
                    # Generate B correlated with A
                    p_same = (1 - corr) / 2  # P(B = -A) for anti-correlation
                    flip = np.random.random(n) < p_same
                    B = np.where(flip, -A, A)

                p_a_plus = float(np.mean(A == 1))
                p_b_plus = float(np.mean(B == 1))
                p_alice_plus_given_b.append(p_a_plus)

                all_alice_outcomes.extend(A.tolist())
                all_bob_settings_for_alice.extend([b] * len(A))
                all_bob_outcomes.extend(B.tolist())
                all_alice_settings_for_bob.extend([a] * len(B))

                rows.append({
                    "alice_setting": float(a),
                    "bob_setting": float(b),
                    "p_alice_plus": p_a_plus,
                    "p_bob_plus": p_b_plus,
                })

            # Check marginal consistency for this Alice setting
            p_arr = np.array(p_alice_plus_given_b)
            dev = float(np.max(p_arr) - np.min(p_arr))
            max_dev = max(max_dev, dev)

        details = pd.DataFrame(rows)

        # Mutual information tests
        alice_outcomes = np.array(all_alice_outcomes)
        bob_settings = np.array(all_bob_settings_for_alice)
        bob_outcomes = np.array(all_bob_outcomes)
        alice_settings_arr = np.array(all_alice_settings_for_bob)

        mi_a_sb = self._estimate_mutual_information(alice_outcomes, bob_settings)
        mi_b_sa = self._estimate_mutual_information(bob_outcomes, alice_settings_arr)

        passed = max_dev < self.tolerance and mi_a_sb < 0.01 and mi_b_sa < 0.01

        return AuditResult(
            model_name=model_name,
            passed=passed,
            max_marginal_deviation=max_dev,
            mutual_info_a_sb=mi_a_sb,
            mutual_info_b_sa=mi_b_sa,
            details=details,
            tolerance=self.tolerance,
        )

    def _estimate_mutual_information(self, outcomes: np.ndarray,
                                      settings: np.ndarray) -> float:
        """Estimate mutual information I(outcomes; settings).

        For no-signaling, this must be ~0.

        Args:
            outcomes: Array of +1/-1 outcomes.
            settings: Array of setting values.

        Returns:
            Mutual information estimate.
        """
        # Discretize settings into bins
        n_bins = min(self.n_settings, 10)
        setting_bins = np.digitize(settings, np.linspace(
            settings.min(), settings.max() + 1e-10, n_bins + 1))

        # Compute MI
        outcome_vals = np.unique(outcomes)
        setting_vals = np.unique(setting_bins)

        n_total = len(outcomes)
        mi = 0.0

        for o in outcome_vals:
            p_o = np.mean(outcomes == o)
            for s in setting_vals:
                mask_s = setting_bins == s
                p_s = mask_s.mean()
                if p_s == 0:
                    continue
                p_os = np.mean((outcomes == o) & mask_s)
                if p_os > 0 and p_o > 0 and p_s > 0:
                    mi += p_os * np.log(p_os / (p_o * p_s))

        return max(0.0, mi)

    def audit_all_models(self, models: dict[str, object]) -> dict[str, AuditResult]:
        """Audit every model and produce consolidated results.

        Args:
            models: Dict of {name: model_object}.

        Returns:
            Dict of {name: AuditResult}.
        """
        results = {}
        for name, model in models.items():
            results[name] = self.audit_model(model, name)
        return results

    def print_audit_report(self, results: dict[str, AuditResult]):
        """Print a formatted audit report."""
        print("=" * 60)
        print("NO-SIGNALING AUDIT REPORT")
        print("=" * 60)
        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"\n  {name}: [{status}]")
            print(f"    Max marginal deviation: {result.max_marginal_deviation:.4f} "
                  f"(threshold: {result.tolerance})")
            print(f"    I(A; S_B) = {result.mutual_info_a_sb:.6f}")
            print(f"    I(B; S_A) = {result.mutual_info_b_sa:.6f}")
        print("\n" + "=" * 60)
