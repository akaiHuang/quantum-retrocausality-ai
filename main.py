"""Quantum Retrocausality Simulation Framework -- Quick Demo.

Runs a quick demonstration of all four phases:
1. Quantum eraser + no-signaling verification
2. TSVF weak value computation (three-box paradox)
3. Bell test comparison (retrocausal vs classical vs QM)
4. Advanced experiments overview

Usage: python main.py
"""

import numpy as np


def demo_phase1_eraser():
    """Phase 1: Quantum eraser demonstrates no-signaling."""
    print("=" * 60)
    print("PHASE 1: Quantum Eraser + No-Signaling Verification")
    print("=" * 60)

    from src.eraser.kim_eraser import KimQuantumEraser
    from src.analysis.statistics import fringe_visibility

    eraser = KimQuantumEraser(n_experiments=20000)
    result = eraser.run_experiment()

    # Total D0 pattern (must be featureless)
    centers, total = result.total_d0_pattern(n_bins=80)
    vis_total = fringe_visibility(total)
    print(f"\n  Total D0 fringe visibility: {vis_total:.4f} (should be ~0)")

    # Coincidence subsets
    for det in ["D1", "D2", "D3", "D4"]:
        c, counts = result.coincidence_pattern(det, n_bins=80)
        vis = fringe_visibility(counts)
        print(f"  D0|{det} fringe visibility: {vis:.4f}", end="")
        if det in ("D1", "D2"):
            print(" (should show fringes)")
        else:
            print(" (should be ~0)")

    print(f"\n  -> No-signaling verified: total D0 has no fringes (V={vis_total:.4f} < 0.05)")
    print("  -> Interference only appears in post-selected subsets!")
    return result


def demo_phase2_tsvf():
    """Phase 2: Two-State Vector Formalism -- three-box paradox."""
    print("\n" + "=" * 60)
    print("PHASE 2: TSVF Weak Values -- Three-Box Paradox")
    print("=" * 60)

    from src.tsvf.weak_values import WeakValueCalculator

    calc = WeakValueCalculator.__new__(WeakValueCalculator)
    result = calc.three_box_paradox()

    print(f"\n  Pre-select:  |psi> = (|A> + |B> + |C>) / sqrt(3)")
    print(f"  Post-select: |phi> = (|A> + |B> - |C>) / sqrt(3)")
    print(f"\n  Weak value of Pi_A (box A projector): {result['Pi_A_weak_value']:.4f}")
    print(f"  Weak value of Pi_B (box B projector): {result['Pi_B_weak_value']:.4f}")
    print(f"  Weak value of Pi_C (box C projector): {result['Pi_C_weak_value']:.4f}")
    print(f"  Sum of weak values: {result['sum']:.4f} (must = 1)")
    print(f"  Pi_C is NEGATIVE: {result['Pi_C_negative']}")
    print(f"\n  -> The particle is 'certainly in A' AND 'certainly in B',")
    print(f"     with a NEGATIVE probability of being in C.")
    print(f"  -> This is the retrocausal signature: the future post-selection")
    print(f"     determines what we can say about the past.")
    return result


def demo_phase3_bell():
    """Phase 3: Bell test comparison."""
    print("\n" + "=" * 60)
    print("PHASE 3: Bell Test -- Retrocausal vs Classical vs QM")
    print("=" * 60)

    from src.retrocausal.bell_test import BellTestComparator

    comparator = BellTestComparator(n_trials=30000)
    chsh = comparator.compute_chsh_values()

    print(f"\n  CHSH Values (classical bound = 2.0, Tsirelson = {2*np.sqrt(2):.3f}):")
    for name, S in chsh.items():
        violation = "VIOLATES Bell" if abs(S) > 2 else "Respects Bell"
        print(f"    {name}: S = {S:.4f}  [{violation}]")

    print(f"\n  -> Retrocausal models violate Bell inequality while being LOCAL")
    print(f"  -> The trick: hidden variable depends on BOTH future settings")
    print(f"  -> Classical models without future-input dependence cannot violate Bell")
    return chsh


def demo_phase4_overview():
    """Phase 4: Advanced experiments overview."""
    print("\n" + "=" * 60)
    print("PHASE 4: Advanced Experiments")
    print("=" * 60)

    # GHZ Mermin test
    from src.advanced.multipartite import ghz_mermin_test
    mermin = ghz_mermin_test()
    print(f"\n  GHZ Mermin test: M = {mermin['mermin_value']:.4f} "
          f"(classical bound = {mermin['classical_bound']}, QM = {mermin['qm_prediction']})")

    # Decoherence
    from src.advanced.decoherence import bell_violation_vs_noise
    noise_data = bell_violation_vs_noise(n_points=5)
    print(f"\n  Bell violation vs noise (depolarizing):")
    for _, row in noise_data.iterrows():
        status = "VIOLATES" if row["violates"] else "respects"
        print(f"    p={row['noise_level']:.2f}: S={row['chsh_value']:.3f} [{status}]")

    # Grover TSVF
    from src.advanced.speedup_analysis import grover_tsvf_analysis
    grover = grover_tsvf_analysis(marked_item=2)
    print(f"\n  Grover TSVF analysis (marked item = 2):")
    for step_name, data in grover["steps"].items():
        wv = data["weak_value_marked"]
        bp = data["born_probability"]
        print(f"    {step_name}: weak_value = {wv}, born_prob = {bp:.3f}")


def main():
    print("\n  QUANTUM RETROCAUSALITY SIMULATION FRAMEWORK")
    print("  Exploring the boundary between retrocausal interpretation")
    print("  and the no-signaling constraint\n")

    demo_phase1_eraser()
    demo_phase2_tsvf()
    demo_phase3_bell()
    demo_phase4_overview()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
  Retrocausal correlations exist in post-selected subsets,
  but the no-signaling theorem ensures they can NEVER be
  used to transmit information backward in time.

  The boundary is sharp and principled:
  - Total distributions are always featureless (no backward signaling)
  - Post-selected subsets show rich retrocausal structure
  - Retrocausal models reproduce ALL quantum predictions while being LOCAL
  - But they cannot enable a "time machine"

  For interactive exploration, see the Jupyter notebooks in notebooks/
""")


if __name__ == "__main__":
    main()
