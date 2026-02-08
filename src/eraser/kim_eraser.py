"""Full simulation of the Kim et al. (1999) delayed-choice quantum eraser.

Reference: Kim, Yu, Kulik, Shih, Scully, PRL 84, 1-5 (2000).
arXiv: quant-ph/9903047

Optical layout:
- SPDC source produces entangled signal (s) + idler (i) pairs
- Signal photon goes to detector D0 via a double-slit arrangement
- Idler photon hits beam splitters and ends at one of D1, D2, D3, D4
- D1/D2: "which-path erased" detectors (coincidence with D0 shows interference)
- D3/D4: "which-path preserved" detectors (coincidence shows no interference)

KEY INSIGHT: The pattern at D0 *alone* (without coincidence counting) is ALWAYS
a featureless Gaussian -- no interference. You can only see fringes by
POST-SELECTING on specific idler detector outcomes. This is why the quantum
eraser does NOT demonstrate retrocausality or backward-in-time signaling.
"""

import numpy as np
from dataclasses import dataclass, field

from .coincidence_counter import CoincidenceCounter, DetectionEvent


@dataclass
class EraserResult:
    """Results from a quantum eraser experiment run."""
    n_experiments: int
    counter: CoincidenceCounter
    x_range: tuple[float, float]
    slit_separation: float
    wavelength: float

    def total_d0_pattern(self, n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Get the total D0 distribution (must be featureless).

        Returns:
            (bin_centers, counts) arrays.
        """
        df = self.counter.get_all_signal_data()
        counts, edges = np.histogram(df["signal_x"], bins=n_bins,
                                     range=self.x_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts.astype(float)

    def coincidence_pattern(self, detector: str,
                            n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Get D0 pattern conditioned on a specific idler detector.

        For D1/D2 (erased): shows interference fringes.
        For D3/D4 (preserved): shows no fringes.

        Args:
            detector: "D1", "D2", "D3", or "D4".

        Returns:
            (bin_centers, counts) arrays.
        """
        df = self.counter.get_coincidences(detector)
        if len(df) == 0:
            bins = np.linspace(self.x_range[0], self.x_range[1], n_bins)
            return bins, np.zeros(n_bins - 1)
        counts, edges = np.histogram(df["signal_x"], bins=n_bins,
                                     range=self.x_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts.astype(float)


class KimQuantumEraser:
    """Simulation of the Kim et al. delayed-choice quantum eraser.

    This uses a wave-optics model rather than a circuit model, since the
    original experiment is optical. The simulation follows the actual
    experimental setup:

    1. SPDC creates a signal-idler pair, entangled in path
    2. Signal photon passes through a double-slit arrangement
    3. Idler photon traverses a beam-splitter network
    4. The beam splitter network either erases or preserves which-path info

    The model computes detection probabilities analytically from the
    quantum state and samples from these distributions.
    """

    def __init__(self, n_experiments: int = 50000,
                 slit_separation: float = 0.4e-3,
                 slit_width: float = 0.1e-3,
                 wavelength: float = 702e-9,
                 screen_distance: float = 1.0,
                 x_range: tuple[float, float] = (-0.005, 0.005)):
        """
        Args:
            n_experiments: Number of photon pairs to simulate.
            slit_separation: Distance between slits (meters).
            slit_width: Width of each slit (meters).
            wavelength: Signal photon wavelength (meters). Kim used ~702nm (SPDC from 351nm).
            screen_distance: Distance from slits to D0 screen (meters).
            x_range: Range of D0 scanning positions (meters).
        """
        self.n_experiments = n_experiments
        self.d = slit_separation
        self.a = slit_width
        self.lam = wavelength
        self.L = screen_distance
        self.x_range = x_range

    def _double_slit_amplitude(self, x: float, slit: str) -> complex:
        """Amplitude for a photon from a given slit to arrive at position x on D0.

        Uses Fraunhofer diffraction:
            A_upper(x) = sinc(pi*a*x/(lam*L)) * exp(i*pi*d*x/(lam*L))
            A_lower(x) = sinc(pi*a*x/(lam*L)) * exp(-i*pi*d*x/(lam*L))

        Args:
            x: Position on the D0 screen.
            slit: "upper" or "lower".

        Returns:
            Complex amplitude.
        """
        envelope = np.sinc(self.a * x / (self.lam * self.L))
        phase = np.pi * self.d * x / (self.lam * self.L)
        if slit == "upper":
            return envelope * np.exp(1j * phase)
        else:
            return envelope * np.exp(-1j * phase)

    def _idler_detector_probabilities(self, slit: str) -> dict[str, float]:
        """Probability that the idler from a given slit hits each detector.

        In the Kim setup, the beam splitter network routes idlers as follows:
        - From upper slit: 50% to D3, 25% to D1, 25% to D2
        - From lower slit: 50% to D4, 25% to D1, 25% to D2

        D3 and D4 preserve which-path info (only receive from one slit).
        D1 and D2 erase which-path info (receive from both slits).
        """
        if slit == "upper":
            return {"D1": 0.25, "D2": 0.25, "D3": 0.50, "D4": 0.00}
        else:
            return {"D1": 0.25, "D2": 0.25, "D3": 0.00, "D4": 0.50}

    def run_experiment(self) -> EraserResult:
        """Run the full quantum eraser simulation.

        For each photon pair:
        1. The SPDC state is |psi> = (|upper>|upper_idler> + |lower>|lower_idler>)/sqrt(2)
        2. Signal photon: sample detection position x from the double-slit pattern
        3. Idler photon: determine which detector fires based on beam-splitter routing
        4. For D1/D2, the which-path info is erased because both slits contribute

        The KEY mechanism: when we condition on D1 or D2, the signal photon's
        position distribution is the INTERFERENCE pattern (because we've projected
        onto a state where both slits contribute coherently). When we condition on
        D3 or D4, it's a single-slit pattern (no interference, which-path known).

        But the TOTAL D0 distribution = D0|D1 + D0|D2 + D0|D3 + D0|D4
        always sums to a featureless envelope. The D1 and D2 fringes are
        complementary (anti-phase) and cancel out; D3 and D4 are already featureless.

        Returns:
            EraserResult containing all detection events and metadata.
        """
        counter = CoincidenceCounter()
        x_samples = np.linspace(self.x_range[0], self.x_range[1], 1000)

        for _ in range(self.n_experiments):
            # Step 1: Choose which slit (equal probability in SPDC)
            slit = np.random.choice(["upper", "lower"])

            # Step 2: Determine idler detector
            probs = self._idler_detector_probabilities(slit)
            detectors = list(probs.keys())
            detector_probs = [probs[d] for d in detectors]
            idler_det = np.random.choice(detectors, p=detector_probs)

            # Step 3: Sample signal position
            # The signal distribution depends on which idler detector fires
            # because we are conditioning on the joint state.
            #
            # For D3 (upper only): P(x) ~ |A_upper(x)|^2 (single slit, no fringes)
            # For D4 (lower only): P(x) ~ |A_lower(x)|^2 (single slit, no fringes)
            # For D1: P(x) ~ |A_upper(x) + A_lower(x)|^2 (double slit, fringes)
            # For D2: P(x) ~ |A_upper(x) - A_lower(x)|^2 (double slit, anti-fringes)

            if idler_det == "D3":
                amplitudes = np.array([self._double_slit_amplitude(x, "upper")
                                       for x in x_samples])
            elif idler_det == "D4":
                amplitudes = np.array([self._double_slit_amplitude(x, "lower")
                                       for x in x_samples])
            elif idler_det == "D1":
                amplitudes = np.array([
                    self._double_slit_amplitude(x, "upper") +
                    self._double_slit_amplitude(x, "lower")
                    for x in x_samples])
            elif idler_det == "D2":
                amplitudes = np.array([
                    self._double_slit_amplitude(x, "upper") -
                    self._double_slit_amplitude(x, "lower")
                    for x in x_samples])
            else:
                continue

            probs_x = np.abs(amplitudes) ** 2
            probs_x /= probs_x.sum()  # normalize

            x_pos = np.random.choice(x_samples, p=probs_x)

            # Step 4: Register the event
            t = np.random.uniform(0, 1)  # simulated time
            event = DetectionEvent(
                signal_x=x_pos,
                signal_time=t,
                idler_detector=idler_det,
                idler_time=t + np.random.normal(0, 1e-10),  # small jitter
                source_path=slit,
            )
            counter.register_event(event)

        return EraserResult(
            n_experiments=self.n_experiments,
            counter=counter,
            x_range=self.x_range,
            slit_separation=self.d,
            wavelength=self.lam,
        )


class WheelerDelayedChoice:
    """Simulation of Wheeler's delayed-choice experiment.

    A Mach-Zehnder interferometer where the second beam splitter
    can be inserted or removed AFTER the photon has entered.

    With BS2: interference (wave behavior)
    Without BS2: which-path (particle behavior)

    The choice is made after the photon enters, yet the behavior is
    as if it "knew" in advance whether to be a wave or particle.
    """

    def __init__(self, n_experiments: int = 10000, phase: float = 0.0):
        """
        Args:
            n_experiments: Number of photons to simulate.
            phase: Phase difference between the two arms.
        """
        self.n_experiments = n_experiments
        self.phase = phase

    def run(self, insert_bs2: bool) -> dict:
        """Run the experiment with or without BS2.

        Args:
            insert_bs2: True = BS2 present (interference); False = BS2 absent (which-path).

        Returns:
            Dict with detector counts and statistics.
        """
        from ..core.operators import beam_splitter, phase_shifter

        bs = beam_splitter()  # 50:50
        ps = phase_shifter(self.phase)

        # Initial state: photon enters from one port
        state = np.array([1, 0], dtype=complex)

        # BS1
        state = bs @ state

        # Phase shift in one arm
        state = ps @ state

        if insert_bs2:
            # BS2 present: interference
            state = bs @ state
            p0 = float(np.abs(state[0]) ** 2)
            p1 = float(np.abs(state[1]) ** 2)
        else:
            # BS2 absent: which-path
            p0 = float(np.abs(state[0]) ** 2)
            p1 = float(np.abs(state[1]) ** 2)

        # Sample outcomes
        outcomes = np.random.choice([0, 1], size=self.n_experiments, p=[p0, p1])
        return {
            "detector_0_count": int(np.sum(outcomes == 0)),
            "detector_1_count": int(np.sum(outcomes == 1)),
            "p0_theory": p0,
            "p1_theory": p1,
            "bs2_present": insert_bs2,
            "phase": self.phase,
        }

    def phase_sweep(self, n_phases: int = 50,
                    insert_bs2: bool = True) -> dict:
        """Sweep the phase and record detector probabilities.

        With BS2: should see sinusoidal interference pattern.
        Without BS2: should see flat 50:50 regardless of phase.

        Args:
            n_phases: Number of phase values to sweep.
            insert_bs2: Whether BS2 is present.

        Returns:
            Dict with phases, p0 values, and p1 values.
        """
        phases = np.linspace(0, 2 * np.pi, n_phases)
        p0_list = []
        p1_list = []
        for phi in phases:
            self.phase = phi
            result = self.run(insert_bs2)
            p0_list.append(result["detector_0_count"] / self.n_experiments)
            p1_list.append(result["detector_1_count"] / self.n_experiments)
        return {
            "phases": phases,
            "p0": np.array(p0_list),
            "p1": np.array(p1_list),
            "bs2_present": insert_bs2,
        }
