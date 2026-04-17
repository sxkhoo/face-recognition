"""CPU fallback detector and embedder for laptop testing without Hailo."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from pipeline.interfaces import FaceDetector, FaceEmbedder
from pipeline.types import Detection

logger = logging.getLogger(__name__)


def _resolve_haarcascade_path() -> str:
    """Resolve the bundled Haar cascade path across OpenCV packaging variants."""
    cascade_name = "haarcascade_frontalface_default.xml"

    candidates: list[Path] = []

    # Allow manual override for stripped-down board images.
    override = os.environ.get("OPENCV_HAAR_CASCADE")
    if override:
        candidates.append(Path(override))

    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidates.append(Path(data_dir) / cascade_name)

    cv2_file = getattr(cv2, "__file__", None)
    if cv2_file:
        cv2_dir = Path(cv2_file).resolve().parent
        candidates.extend(
            [
                cv2_dir / "data" / cascade_name,
                cv2_dir.parent / "share" / "opencv4" / "haarcascades" / cascade_name,
                cv2_dir.parent / "share" / "opencv" / "haarcascades" / cascade_name,
            ]
        )

    candidates.extend(
        [
            Path("/usr/share/opencv4/haarcascades") / cascade_name,
            Path("/usr/share/opencv/haarcascades") / cascade_name,
            Path("/usr/local/share/opencv4/haarcascades") / cascade_name,
            Path("/usr/local/share/opencv/haarcascades") / cascade_name,
            Path("/usr/share/OpenCV/haarcascades") / cascade_name,
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate OpenCV Haar cascade XML. "
        "Set OPENCV_HAAR_CASCADE to the full path of "
        f"{cascade_name}. Searched: {searched}"
    )


class DummyDetector(FaceDetector):
    """Returns empty detections. Useful for pipeline integration testing."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []


class OpenCVCascadeDetector(FaceDetector):
    """Simple Haar cascade detector for CPU-only laptop testing.

    Not accurate enough for production — just for verifying the pipeline
    runs end-to-end without Hailo hardware.
    Landmarks are approximated from bbox geometry.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        cascade_path = _resolve_haarcascade_path()
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(60, 60),
        )

        detections = []
        for x, y, w, h in rects:
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            landmarks = self._approximate_landmarks(x, y, w, h)
            detections.append(Detection(bbox=bbox, landmarks_5=landmarks, score=0.9))

        # Sort by area descending
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    @staticmethod
    def _approximate_landmarks(x: int, y: int, w: int, h: int) -> np.ndarray:
        """Approximate 5-point landmarks from bbox geometry.

        Good enough for alignment testing; not geometrically precise.
        """
        cx, cy = x + w / 2, y + h / 2
        return np.array(
            [
                [cx - w * 0.17, cy - h * 0.12],   # left eye
                [cx + w * 0.17, cy - h * 0.12],   # right eye
                [cx,            cy + h * 0.05],    # nose tip
                [cx - w * 0.14, cy + h * 0.22],   # left mouth
                [cx + w * 0.14, cy + h * 0.22],   # right mouth
            ],
            dtype=np.float32,
        )


class DummyEmbedder(FaceEmbedder):
    """Returns deterministic pseudo-random embeddings for testing.

    Uses a hash of the face image so the same input gives the same
    embedding — useful for debugging pipeline logic.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self._dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        # Deterministic seed from image content
        seed = int(np.sum(aligned_face.astype(np.int64)) % (2**31))
        rng = np.random.RandomState(seed)
        emb = rng.randn(self._dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        return emb
