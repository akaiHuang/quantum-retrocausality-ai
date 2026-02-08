"""Bell test comparator across different models.

Compares Bell test results from:
1. Standard quantum mechanics (analytical)
2. ZigZag retrocausal model
3. Boundary-value model
4. Classical local hidden variable model

All retrocausal models should match QM predictions.
Classical model should NOT violate Bell inequality.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .zigzag_model import ZigZagModel, ClassicalLocalModel, BellTestResult
from .boundary_value import BoundaryValueModel


@dataclass
class ComparisonReport:
    """Full comparison report across all models."""
    correlation_data: pd.DataFrame
    chsh_values: dict[str, float]
    qm_prediction: callable
    summary: str


class BellTestComparator:
    """Compare Bell test results across different models.

    This is the central demonstration of Phase 3:
    - Retrocausal models violate Bell inequality (S = 2*sqrt(2))
    - Classical local models respect Bell bound (S <= 2)
    - All models match in correlation structure, differ in mechanism
    """

    def __init__(self, n_trials: int = 50000):
        self.n_trials = n_trials
        self.models: dict[str, object] = {}
        self._register_default_models()

    def _register_default_models(self):
        """Register the three default models."""
        self.models["QM (analytical)"] = "analytical"
        self.models["ZigZag Retrocausal"] = ZigZagModel(self.n_trials)
        self.models["Boundary-Value"] = BoundaryValueModel(self.n_trials)
        self.models["Classical Local"] = ClassicalLocalModel(self.n_trials)

    def register_model(self, name: str, model: object):
        """Register an additional model for comparison.

        Args:
            name: Display name.
            model: Object with a .correlation(a, b) method.
        """
        self.models[name] = model

    @staticmethod
    def qm_singlet_correlation(a: float, b: float) -> float:
        """Analytical QM prediction for singlet state: E(a,b) = -cos(a-b)."""
        return -np.cos(a - b)

    def run_chsh_sweep(self, n_angles: int = 36) -> pd.DataFrame:
        """Sweep through angles and compute E(0, theta) for all models.

        This produces the correlation curves that should overlap
        (for retrocausal models + QM) or deviate (for classical).

        Args:
            n_angles: Number of angles to sweep.

        Returns:
            DataFrame with columns [model, angle, correlation].
        """
        angles = np.linspace(0, 2 * np.pi, n_angles)
        rows = []

        for name, model in self.models.items():
            for theta in angles:
                if model == "analytical":
                    corr = self.qm_singlet_correlation(0.0, theta)
                else:
                    corr = model.correlation(0.0, theta)
                rows.append({
                    "model": name,
                    "angle": float(theta),
                    "correlation": corr,
                })

        return pd.DataFrame(rows)

    def compute_chsh_values(self, a: float = 0.0,
                             a_prime: float = np.pi / 2,
                             b: float = np.pi / 4,
                             b_prime: float = 3 * np.pi / 4) -> dict[str, float]:
        """Compute CHSH value S for each model.

        S = E(a,b) - E(a,b') + E(a',b) + E(a',b')

        QM / Retrocausal: S = 2*sqrt(2) ~ 2.828
        Classical: S <= 2

        Args:
            a, a_prime: Alice's two settings.
            b, b_prime: Bob's two settings.

        Returns:
            Dict of {model_name: S_value}.
        """
        results = {}
        for name, model in self.models.items():
            if model == "analytical":
                E = lambda x, y: self.qm_singlet_correlation(x, y)
            else:
                E = model.correlation

            S = (E(a, b) - E(a, b_prime) +
                 E(a_prime, b) + E(a_prime, b_prime))
            results[name] = float(S)

        return results

    def generate_comparison_report(self) -> ComparisonReport:
        """Generate a full comparison report.

        Returns:
            ComparisonReport with all data and summary.
        """
        correlation_data = self.run_chsh_sweep()
        chsh_values = self.compute_chsh_values()

        # Generate summary
        lines = ["=" * 60]
        lines.append("BELL TEST COMPARISON REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append("CHSH Values (classical bound = 2, Tsirelson bound = 2*sqrt(2) ~ 2.828):")
        lines.append("-" * 40)

        for name, S in chsh_values.items():
            violation = "VIOLATES Bell" if abs(S) > 2 else "Respects Bell"
            lines.append(f"  {name}: S = {S:.4f}  [{violation}]")

        lines.append("")
        lines.append("KEY INSIGHT:")
        lines.append("-" * 40)
        lines.append("  - Retrocausal models violate Bell inequality (S > 2)")
        lines.append("  - But they are LOCAL: outcomes depend only on (lambda, local_setting)")
        lines.append("  - The trick: hidden variable lambda depends on BOTH future settings")
        lines.append("  - Classical models without future-input dependence CANNOT violate Bell")
        lines.append("  - No-signaling is always respected: P(A|a) is independent of b")

        summary = "\n".join(lines)

        return ComparisonReport(
            correlation_data=correlation_data,
            chsh_values=chsh_values,
            qm_prediction=self.qm_singlet_correlation,
            summary=summary,
        )
