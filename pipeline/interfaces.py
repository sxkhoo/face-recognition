"""Abstract interfaces for pluggable pipeline components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from pipeline.types import Detection


class FaceDetector(ABC):
    """Interface for face detection backends (Hailo, CPU, ONNX, etc.)."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in a BGR frame.

        Args:
            frame: HxWx3 BGR uint8 image.

        Returns:
            List of Detection objects sorted by score descending.
        """
        ...

    def warmup(self) -> None:
        """Optional warmup call (e.g., run dummy inference to init device)."""

    def release(self) -> None:
        """Optional cleanup (e.g., release Hailo device)."""


class FaceEmbedder(ABC):
    """Interface for face embedding backends."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings."""
        ...

    @abstractmethod
    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Compute L2-normalized embedding for an aligned 112x112 face.

        Args:
            aligned_face: 112x112x3 BGR uint8 image.

        Returns:
            (embedding_dim,) float32 array, L2-normalized.
        """
        ...

    def embed_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """Embed multiple faces. Default: loop over embed().

        Args:
            faces: List of 112x112x3 BGR uint8 images.

        Returns:
            (N, embedding_dim) float32 array, each row L2-normalized.
        """
        embeddings = np.stack([self.embed(f) for f in faces])
        return embeddings

    def warmup(self) -> None:
        """Optional warmup."""

    def release(self) -> None:
        """Optional cleanup."""


class LivenessChecker(ABC):
    """Interface for face liveness / anti-spoofing."""

    @abstractmethod
    def check(self, aligned_face: np.ndarray) -> Tuple[bool, float]:
        """Check if face is live (not a spoof).

        Args:
            aligned_face: 112x112x3 BGR uint8 image.

        Returns:
            (is_live, confidence_score) where score in [0, 1].
        """
        ...
