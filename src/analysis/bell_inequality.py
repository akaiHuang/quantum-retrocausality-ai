"""Bell inequality computations.

Provides CHSH inequality evaluation, Mermin inequality for GHZ states,
and tools for computing correlations from measurement data.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CHSHResult:
    """Result of a CHSH inequality evaluation."""
    S: float                    # CHSH value
    classical_bound: float      # 2.0
    tsirelson_bound: float      # 2*sqrt(2)
    violates_classical: bool    # |S| > 2
    violates_tsirelson: bool    # |S| > 2*sqrt(2) (should not happen for QM)
    correlations: dict


def chsh_value(E_ab: float, E_ab_prime: float,
               E_a_prime_b: float,
               E_a_prime_b_prime: float) -> CHSHResult:
    """Compute the CHSH value S from four correlations.

    S = E(a,b) - E(a,b') + E(a',b) + E(a',b')

    Classical bound: |S| <= 2
    Tsirelson bound: |S| <= 2*sqrt(2) ~ 2.828

    Args:
        E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime: Correlations.

    Returns:
        CHSHResult.
    """
    S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

    return CHSHResult(
        S=S,
        classical_bound=2.0,
        tsirelson_bound=2 * np.sqrt(2),
        violates_classical=abs(S) > 2.0,
        violates_tsirelson=abs(S) > 2 * np.sqrt(2) + 0.01,
        correlations={
            "E(a,b)": E_ab,
            "E(a,b')": E_ab_prime,
            "E(a',b)": E_a_prime_b,
            "E(a',b')": E_a_prime_b_prime,
        },
    )


def optimal_chsh_settings() -> dict:
    """Return the optimal CHSH settings for maximum violation.

    For the singlet state:
        a = 0, a' = pi/2, b = pi/4, b' = 3*pi/4
    gives S = 2*sqrt(2).

    Returns:
        Dict with optimal angle settings.
    """
    return {
        "a": 0.0,
        "a_prime": np.pi / 2,
        "b": np.pi / 4,
        "b_prime": 3 * np.pi / 4,
    }


def correlation_from_data(outcomes_a: np.ndarray,
                          outcomes_b: np.ndarray) -> float:
    """Compute correlation E = <A*B> from outcome arrays.

    Args:
        outcomes_a: Array of +1/-1 values.
        outcomes_b: Array of +1/-1 values.

    Returns:
        Correlation in [-1, 1].
    """
    a = np.asarray(outcomes_a, dtype=float)
    b = np.asarray(outcomes_b, dtype=float)
    return float(np.mean(a * b))


@dataclass
class MerminResult:
    """Result of a Mermin inequality evaluation."""
    M: float                     # Mermin value
    classical_bound: float       # 1.0 for 3-qubit
    qm_prediction: float         # 4.0 for 3-qubit GHZ
    n_qubits: int
    violates: bool


def mermin_inequality_3qubit(correlations: dict[str, float]) -> MerminResult:
    """Evaluate the 3-qubit Mermin inequality.

    M = E(X,Y,Y) + E(Y,X,Y) + E(Y,Y,X) - E(X,X,X)

    Classical bound: |M| <= 2
    QM prediction for GHZ: M = 4

    Args:
        correlations: Dict with keys "XYY", "YXY", "YYX", "XXX"
                      giving the 3-party correlator for those basis choices.

    Returns:
        MerminResult.
    """
    M = (correlations.get("XYY", 0) + correlations.get("YXY", 0) +
         correlations.get("YYX", 0) - correlations.get("XXX", 0))

    return MerminResult(
        M=M,
        classical_bound=2.0,
        qm_prediction=4.0,
        n_qubits=3,
        violates=abs(M) > 2.0,
    )


def qm_singlet_correlation(a: float, b: float) -> float:
    """Analytical QM prediction for singlet state: E(a,b) = -cos(a-b).

    Args:
        a: Alice's measurement angle.
        b: Bob's measurement angle.

    Returns:
        Correlation.
    """
    return -np.cos(a - b)
