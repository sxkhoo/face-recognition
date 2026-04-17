"""Liveness / anti-spoofing checker — stub implementation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pipeline.interfaces import LivenessChecker


class StubLivenessChecker(LivenessChecker):
    """Placeholder that always returns live=True.

    Replace with SilentFace, FAS, or other model later.
    """

    def check(self, aligned_face: np.ndarray) -> Tuple[bool, float]:
        return True, 1.0


class SimpleLivenessChecker(LivenessChecker):
    """Basic heuristic liveness using color variance.

    Not production-grade — just demonstrates the interface.
    Checks that the face has reasonable color variation
    (flat printouts/screens sometimes have lower variance).
    """

    def __init__(self, min_color_std: float = 15.0) -> None:
        self.min_color_std = min_color_std

    def check(self, aligned_face: np.ndarray) -> Tuple[bool, float]:
        std = float(np.std(aligned_face))
        # Normalize to [0, 1] range — higher std = more likely live
        score = min(std / 60.0, 1.0)
        is_live = std >= self.min_color_std
        return is_live, score
