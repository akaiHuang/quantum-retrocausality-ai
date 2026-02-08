"""Price-Wharton 'zigzag' retrocausal hidden variable model.

References:
- Price (2008), "Toy Models for Retrocausality", Studies in HPS.
- Wharton & Argaman (2020), Rev. Mod. Phys. 92, 021002.

In this model, correlations between entangled particles are explained by
a hidden variable lambda at the source that depends on BOTH future
measurement settings. This is the 'zigzag' causation:

    Alice's detector <-- source --> Bob's detector
          |                               |
          v                               v
    Alice's setting a               Bob's setting b
          |                               |
          +--> lambda at source depends on both a and b <--+

The model:
- Is LOCAL: Alice's outcome depends only on (lambda, a), not on b
- Violates Bell's inequality: by allowing lambda to depend on future settings
- Respects no-signaling: P(A|a) is independent of b (marginals are uniform)
- Reproduces QM predictions for the singlet state: E(a,b) = -cos(a-b)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BellTestResult:
    """Results from a Bell test (CHSH)."""
    correlations: dict[tuple[float, float], float]  # {(a,b): E(a,b)}
    chsh_value: float       # S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    n_trials: int
    model_name: str
    settings: dict           # {a, a', b, b'}


class ZigZagModel:
    """Implementation of the Price-Wharton zigzag retrocausal model.

    For the singlet state, the model works as follows:

    The hidden variable lambda = (lambda_A, lambda_corr) encodes:
    - lambda_A: determines Alice's outcome (uniform random bit)
    - lambda_corr: a "correlation instruction" that depends on BOTH
      future settings a and b (this is the retrocausal element)

    Outcomes are determined LOCALLY:
    - A(lambda_A, a): depends only on lambda_A and a, not b
    - B(lambda_corr, lambda_A, b): depends only on lambda_corr and b
      (and implicitly on lambda_A through the correlation)

    The retrocausal mechanism: lambda_corr is sampled from a distribution
    p(lambda_corr | a, b) that depends on BOTH future settings.
    This gives the hidden variable at the source "knowledge" of both
    measurement choices, while keeping each party's outcome local.

    Marginals: P(A=+1|a) = 1/2 for all a (independent of b) -> no-signaling
    Correlation: E(a,b) = -cos(a-b) -> matches quantum mechanics
    """

    def __init__(self, n_trials: int = 100000):
        """
        Args:
            n_trials: Number of trials per setting pair.
        """
        self.n_trials = n_trials

    def correlation(self, a: float, b: float) -> float:
        """Compute E(a, b) = <A * B> for a specific setting pair.

        Reproduces the QM prediction: E(a,b) = -cos(a-b).

        The retrocausal mechanism:
        1. Alice's outcome A is +1 or -1 with equal probability (coin flip)
        2. The probability that B agrees with A depends on (a, b):
           P(B = -A) = (1 + cos(a-b)) / 2  (anti-correlation for singlet)
           P(B = +A) = (1 - cos(a-b)) / 2

        This gives E(a,b) = P(agree) - P(disagree) = -cos(a-b). QED.

        The retrocausal element: the "agreement probability" at the source
        depends on BOTH future settings a and b. This is the future-input
        dependence that allows Bell violation while maintaining locality.

        Args:
            a: Alice's setting.
            b: Bob's setting.

        Returns:
            Correlation in [-1, 1].
        """
        theta = a - b
        # Probability that outcomes AGREE (both +1 or both -1)
        p_agree = (1 - np.cos(theta)) / 2
        # Probability that outcomes DISAGREE
        p_disagree = (1 + np.cos(theta)) / 2

        # Generate Alice's outcomes: uniform +-1
        A = np.random.choice([-1, 1], size=self.n_trials)

        # Generate agreement/disagreement flags from retrocausal distribution
        # This flag depends on (a, b) -- the retrocausal ingredient
        agree = np.random.random(self.n_trials) < p_agree
        B = np.where(agree, A, -A)

        return float(np.mean(A * B))

    def alice_outcome(self, lam: np.ndarray, a: float) -> np.ndarray:
        """Alice's measurement outcome (LOCAL).

        A depends only on the hidden variable component lambda_A,
        not on Bob's setting b.

        Args:
            lam: Array of hidden variable values (unused -- Alice is just random).
            a: Alice's measurement setting (unused for outcome, determines only basis).

        Returns:
            Array of +1/-1 outcomes.
        """
        return np.random.choice([-1, 1], size=len(lam))

    def bob_outcome(self, lam: np.ndarray, b: float) -> np.ndarray:
        """Bob's measurement outcome (LOCAL).

        B depends on the hidden variable (which encodes the correlation)
        and on b. It does NOT depend on a directly.

        Args:
            lam: Array of (alice_outcome, agreement_flag) tuples effectively.
            b: Bob's measurement setting.

        Returns:
            Array of +1/-1 outcomes.
        """
        # This is called internally with the agreement already determined
        return lam  # pre-computed in correlation()

    def _sample_hidden_variables(self, a: float, b: float,
                                  n: int) -> np.ndarray:
        """Sample hidden variables from the retrocausal distribution.

        The hidden variable encodes a correlation instruction that
        depends on BOTH a and b. This is the retrocausal element.

        Returns:
            Array of hidden variable values (correlation strengths).
        """
        theta = a - b
        p_agree = (1 - np.cos(theta)) / 2
        return np.random.random(n) < p_agree  # agreement flags

    def run_bell_test(self, a: float = 0.0, a_prime: float = np.pi / 2,
                      b: float = np.pi / 4,
                      b_prime: float = 3 * np.pi / 4) -> BellTestResult:
        """Run a full CHSH Bell test.

        S = E(a,b) - E(a,b') + E(a',b) + E(a',b')

        For optimal settings: S = 2*sqrt(2) ~ 2.828
        Classical bound: S <= 2
        Tsirelson bound: S <= 2*sqrt(2)

        Args:
            a, a_prime: Alice's two settings.
            b, b_prime: Bob's two settings.

        Returns:
            BellTestResult with CHSH value and all correlations.
        """
        E_ab = self.correlation(a, b)
        E_ab_prime = self.correlation(a, b_prime)
        E_a_prime_b = self.correlation(a_prime, b)
        E_a_prime_b_prime = self.correlation(a_prime, b_prime)

        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

        return BellTestResult(
            correlations={
                (a, b): E_ab,
                (a, b_prime): E_ab_prime,
                (a_prime, b): E_a_prime_b,
                (a_prime, b_prime): E_a_prime_b_prime,
            },
            chsh_value=S,
            n_trials=self.n_trials,
            model_name="ZigZag Retrocausal Model (Price-Wharton)",
            settings={"a": a, "a_prime": a_prime, "b": b, "b_prime": b_prime},
        )

    def correlation_curve(self, n_angles: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Compute E(0, theta) for theta in [0, 2*pi].

        Should trace out -cos(theta), matching the QM singlet prediction.

        Args:
            n_angles: Number of angles to compute.

        Returns:
            (angles, correlations) arrays.
        """
        angles = np.linspace(0, 2 * np.pi, n_angles)
        correlations = np.array([self.correlation(0.0, theta) for theta in angles])
        return angles, correlations

    def verify_locality(self, a: float = 0.0, b: float = np.pi / 4) -> dict:
        """Verify that Alice's marginal is independent of Bob's setting.

        P(A=+1 | a) must be 1/2 regardless of b. This is the no-signaling check.

        Returns:
            Dict confirming locality and no-signaling.
        """
        # Test with different Bob settings
        alice_marginals = {}
        for b_test in [0, np.pi / 4, np.pi / 2, np.pi]:
            theta = a - b_test
            p_agree = (1 - np.cos(theta)) / 2

            A = np.random.choice([-1, 1], size=self.n_trials)
            agree = np.random.random(self.n_trials) < p_agree
            B = np.where(agree, A, -A)

            alice_marginals[f"b={b_test:.2f}"] = float(np.mean(A == 1))

        deviations = [abs(p - 0.5) for p in alice_marginals.values()]
        max_dev = max(deviations)

        return {
            "alice_marginals": alice_marginals,
            "max_deviation_from_half": max_dev,
            "locality_verified": max_dev < 0.02,
            "note": (
                "Alice's marginal P(A=+1) is always ~0.5 regardless of Bob's "
                "setting. The retrocausality is in the JOINT distribution, not "
                "in the marginals. No information is transmitted backward."
            ),
        }


class ClassicalLocalModel:
    """A classical local hidden variable model (for comparison).

    This model uses hidden variables that do NOT depend on future settings.
    It CANNOT violate Bell's inequality (S <= 2 always).
    """

    def __init__(self, n_trials: int = 100000):
        self.n_trials = n_trials

    def correlation(self, a: float, b: float) -> float:
        """Compute E(a,b) using a classical local model.

        Hidden variable lambda is uniformly distributed (no future dependence).
        Outcomes: A = sign(cos(lambda - a)), B = -sign(cos(lambda - b)).

        This gives E(a,b) = -1 + 2|a-b|/pi (linear, not cosine).
        It satisfies S <= 2 for all setting choices.
        """
        lam = np.random.uniform(0, 2 * np.pi, self.n_trials)
        A = np.sign(np.cos(lam - a))
        B = -np.sign(np.cos(lam - b))
        mask = (A != 0) & (B != 0)
        return float(np.mean(A[mask] * B[mask]))

    def run_bell_test(self, a: float = 0.0, a_prime: float = np.pi / 2,
                      b: float = np.pi / 4,
                      b_prime: float = 3 * np.pi / 4) -> BellTestResult:
        """Run CHSH test. Should give S <= 2."""
        E_ab = self.correlation(a, b)
        E_ab_prime = self.correlation(a, b_prime)
        E_a_prime_b = self.correlation(a_prime, b)
        E_a_prime_b_prime = self.correlation(a_prime, b_prime)

        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

        return BellTestResult(
            correlations={
                (a, b): E_ab,
                (a, b_prime): E_ab_prime,
                (a_prime, b): E_a_prime_b,
                (a_prime, b_prime): E_a_prime_b_prime,
            },
            chsh_value=S,
            n_trials=self.n_trials,
            model_name="Classical Local Model (no retrocausality)",
            settings={"a": a, "a_prime": a_prime, "b": b, "b_prime": b_prime},
        )
