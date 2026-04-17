"""Core data types used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """Single face detection result."""

    bbox: np.ndarray            # (4,) — x1, y1, x2, y2 in pixel coords
    landmarks_5: np.ndarray     # (5, 2) — left_eye, right_eye, nose, left_mouth, right_mouth
    score: float

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class QualityResult:
    """Face crop quality assessment."""

    passed: bool
    blur_score: float           # Laplacian variance (higher = sharper)
    detection_score: float
    face_size: float            # min(width, height) of bbox
    reason: str = ""


@dataclass
class MatchResult:
    """Watchlist match result."""

    identity_id: str
    name: str
    distance: float             # Cosine distance (lower = more similar)
    matched: bool               # True if distance < threshold

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@dataclass
class Alert:
    """Alert emitted when a watchlist match occurs."""

    identity_id: str
    name: str
    similarity: float
    bbox: np.ndarray
    timestamp: float            # time.time()
    camera_id: str
    frame_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full result for one face through the pipeline."""

    detection: Detection
    liveness_passed: Optional[bool] = None
    quality: Optional[QualityResult] = None
    embedding: Optional[np.ndarray] = None
    match: Optional[MatchResult] = None
    alert: Optional[Alert] = None
