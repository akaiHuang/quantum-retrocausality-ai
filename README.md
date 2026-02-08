# quantum-retrocausality-ai

**Computational framework for exploring retrocausality in quantum mechanics.**

Can quantum mechanics let us send information backward in time? **No.** But the boundary between what retrocausality *means* and what it *cannot do* is fascinating -- and this project lets you explore it computationally.

## The Question

The delayed-choice quantum eraser (Kim et al., 1999) appears to show that a future measurement choice can retroactively affect a past result. Does this mean we can build a time machine?

## The Answer (in code)

This framework demonstrates three key results:

1. **The quantum eraser does NOT enable backward signaling.** The total D0 distribution is always featureless -- interference only appears in post-selected subsets via coincidence counting. (Phase 1)

2. **Retrocausal effects are real in post-selected ensembles.** The Two-State Vector Formalism (TSVF) produces anomalous weak values that lie outside the eigenvalue spectrum -- experimentally measurable "retrocausal" signatures. (Phase 2)

3. **Retrocausal models can be local AND violate Bell's inequality.** The Price-Wharton zigzag model reproduces all quantum predictions while being strictly local, at the cost of future-input dependence. But no-signaling is always respected. (Phase 3)

## Features

- **Quantum Eraser Simulation** -- Full Kim et al. (1999) setup with SPDC, coincidence counting, and no-signaling verification
- **TSVF Engine** -- First open-source Two-State Vector Formalism simulator with weak value calculator, ABL rule, and the three-box paradox
- **Retrocausal Toy Models** -- Executable implementations of the Price-Wharton zigzag model and Wharton-Argaman boundary-value approach
- **Bell Test Comparator** -- Side-by-side comparison of QM, retrocausal, and classical models
- **No-Signaling Audit** -- Rigorous verification that all models respect the no-communication theorem
- **Advanced Experiments** -- GHZ/W state analysis, decoherence effects, Tlalpan phase-transition simulation, Castagnoli speedup-retrocausality link

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

```
src/
  core/             # Quantum states, operators, density matrices
  eraser/           # Phase 1: Quantum eraser + no-signaling
  tsvf/             # Phase 2: Two-State Vector Formalism engine
  retrocausal/      # Phase 3: Retrocausal hidden variable models
  advanced/         # Phase 4: GHZ, decoherence, phase transitions
  analysis/         # Statistical tools, Bell inequalities
  visualization/    # Plotting functions
tests/              # Unit tests
notebooks/          # Interactive Jupyter notebooks
```

## Key References

| Paper | Year | Relevance |
|-------|------|-----------|
| Kim, Yu, Kulik, Shih, Scully -- Delayed-Choice Quantum Eraser | 1999 | Phase 1: the experiment we simulate |
| Aharonov, Bergmann, Lebowitz -- Time Symmetry in Quantum Measurement | 1964 | Phase 2: foundation of TSVF |
| Aharonov, Albert, Vaidman -- Weak Values | 1988 | Phase 2: anomalous weak values |
| Wharton & Argaman -- Bell's Theorem and Local Reformulations | 2020 | Phase 3: retrocausal Bell models |
| Leifer & Pusey -- Time Symmetry Without Retrocausality? | 2017 | Theoretical motivation |
| Castagnoli -- Quantum Speedup and Retrocausality | 2025 | Phase 4: speedup connection |

## Dependencies

- numpy, scipy, matplotlib, plotly, pandas
- pytest (for testing)

## License

MIT
