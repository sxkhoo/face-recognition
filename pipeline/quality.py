"""Face crop quality assessment — blur, size, detection confidence."""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.types import Detection, QualityResult


def laplacian_blur_score(image: np.ndarray) -> float:
    """Compute Laplacian variance as a sharpness metric.

    Higher value = sharper image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_quality(
    aligned_face: np.ndarray,
    detection: Detection,
    blur_threshold: float = 100.0,
    min_detection_score: float = 0.6,
    min_face_size: float = 56.0,
) -> QualityResult:
    """Assess whether a face crop is good enough for embedding.

    Args:
        aligned_face: 112x112x3 BGR image.
        detection: Original detection (for score and bbox size).
        blur_threshold: Minimum Laplacian variance to pass.
        min_detection_score: Minimum detector confidence.
        min_face_size: Minimum bbox side length in original frame pixels.

    Returns:
        QualityResult with pass/fail and reason.
    """
    blur = laplacian_blur_score(aligned_face)
    face_size = min(detection.width, detection.height)

    reasons = []
    if blur < blur_threshold:
        reasons.append(f"blurry ({blur:.1f} < {blur_threshold})")
    if detection.score < min_detection_score:
        reasons.append(f"low det score ({detection.score:.2f} < {min_detection_score})")
    if face_size < min_face_size:
        reasons.append(f"small face ({face_size:.0f}px < {min_face_size})")

    passed = len(reasons) == 0

    return QualityResult(
        passed=passed,
        blur_score=blur,
        detection_score=detection.score,
        face_size=face_size,
        reason="; ".join(reasons) if reasons else "ok",
    )
