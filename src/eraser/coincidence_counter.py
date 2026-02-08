"""Coincidence counting and post-selection engine.

In the quantum eraser, interference only "reappears" when you post-select
on specific idler-detector outcomes via coincidence counting. This is NOT
a physical effect on the signal photon -- it is a SELECTION of data subsets.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DetectionEvent:
    """A single detection event in the quantum eraser experiment."""
    signal_x: float          # signal photon position at D0
    signal_time: float       # signal photon detection time
    idler_detector: str      # which idler detector fired ("D1", "D2", "D3", "D4")
    idler_time: float        # idler photon detection time
    source_path: str         # which slit the SPDC pair originated from ("upper", "lower")


class CoincidenceCounter:
    """Engine for coincidence counting and post-selection.

    This is the mechanism by which interference "reappears" in the quantum eraser.
    It is NOT a physical effect on the signal photon; it is a SELECTION of data subsets.

    The timing window simulates a real coincidence counter that only
    registers events where both signal and idler are detected within
    a narrow time window.
    """

    def __init__(self, timing_window: float = 1e-9):
        """
        Args:
            timing_window: Coincidence window in seconds (simulated).
        """
        self.timing_window = timing_window
        self.events: list[DetectionEvent] = []

    def register_event(self, event: DetectionEvent):
        """Register a detection event."""
        self.events.append(event)

    def register_batch(self, events: list[DetectionEvent]):
        """Register multiple events at once."""
        self.events.extend(events)

    def get_coincidences(self, idler_detector: str) -> pd.DataFrame:
        """Get signal photon data conditioned on a specific idler detector.

        This is the post-selection step. For D1/D2 (which-path erased),
        the signal distribution shows interference fringes. For D3/D4
        (which-path preserved), it does not.

        Args:
            idler_detector: One of "D1", "D2", "D3", "D4".

        Returns:
            DataFrame with columns [signal_x, signal_time, idler_time, source_path].
        """
        matching = [e for e in self.events
                    if e.idler_detector == idler_detector
                    and abs(e.signal_time - e.idler_time) < self.timing_window]
        return pd.DataFrame([{
            "signal_x": e.signal_x,
            "signal_time": e.signal_time,
            "idler_time": e.idler_time,
            "source_path": e.source_path,
        } for e in matching])

    def get_all_signal_data(self) -> pd.DataFrame:
        """Get ALL signal photon data, regardless of idler detector.

        This produces the TOTAL D0 distribution, which must be featureless
        (no interference). This is the no-signaling constraint in action.

        Returns:
            DataFrame with all signal positions.
        """
        return pd.DataFrame([{
            "signal_x": e.signal_x,
            "idler_detector": e.idler_detector,
            "source_path": e.source_path,
        } for e in self.events])

    def post_select(self, condition: callable) -> pd.DataFrame:
        """General post-selection: filter events by arbitrary condition.

        Args:
            condition: Function that takes a DetectionEvent and returns bool.

        Returns:
            Filtered DataFrame.
        """
        matching = [e for e in self.events if condition(e)]
        return pd.DataFrame([{
            "signal_x": e.signal_x,
            "signal_time": e.signal_time,
            "idler_detector": e.idler_detector,
            "idler_time": e.idler_time,
            "source_path": e.source_path,
        } for e in matching])

    @property
    def n_events(self) -> int:
        return len(self.events)

    def detector_counts(self) -> dict[str, int]:
        """Count events per idler detector."""
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.idler_detector] = counts.get(e.idler_detector, 0) + 1
        return counts
