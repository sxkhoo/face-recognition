"""Hailo-8 face embedder wrapper (AdaFace / ArcFace).

Requires hailo_platform and a compiled .hef model.
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from hailo_utils import activate_network_group, import_hailo, resolve_vstream_name
from pipeline.interfaces import FaceEmbedder

logger = logging.getLogger(__name__)


class HailoFaceEmbedder(FaceEmbedder):
    """AdaFace / ArcFace embedder running on Hailo-8.

    Expects a .hef compiled from AdaFace-IR101, ArcFace-R100, or
    similar 112x112 input recognition model.
    """

    def __init__(
        self,
        hef_path: str,
        embedding_dim: int = 512,
        vdevice=None,
    ) -> None:
        h = import_hailo()

        self._embedding_dim = embedding_dim

        logger.info("Loading embedder HEF: %s", hef_path)
        self._hef = h["HEF"](hef_path)

        self._owns_device = vdevice is None
        self._device = vdevice if vdevice is not None else h["VDevice"]()
        configure_params = h["ConfigureParams"].create_from_hef(
            hef=self._hef, interface=h["HailoStreamInterface"].PCIe
        )
        self._network_group = self._device.configure(self._hef, configure_params)[0]

        self._input_params = h["InputVStreamParams"].make(
            self._network_group,
            format_type=h["FormatType"].FLOAT32,
        )
        self._output_params = h["OutputVStreamParams"].make(
            self._network_group,
            format_type=h["FormatType"].FLOAT32,
        )
        self._input_name = resolve_vstream_name(self._input_params)

        logger.info("Hailo face embedder initialized (dim=%d).", embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        h = import_hailo()

        input_tensor = self._preprocess(aligned_face)

        input_dict = {
            self._input_name: input_tensor[np.newaxis, ...]
        }
        with activate_network_group(self._network_group):
            with h["InferVStreams"](
                self._network_group,
                self._input_params,
                self._output_params,
            ) as pipeline:
                output_dict = pipeline.infer(input_dict)

        # Get first (and typically only) output tensor
        raw_embedding = np.asarray(next(iter(output_dict.values())))
        if raw_embedding.ndim > 1 and raw_embedding.shape[0] == 1:
            raw_embedding = raw_embedding[0]
        raw_embedding = raw_embedding.flatten()

        # L2 normalize
        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            raw_embedding = raw_embedding / norm

        return raw_embedding.astype(np.float32)

    def embed_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """Batch embedding — runs sequentially through Hailo for now.

        For better throughput, implement async pipeline with double-buffering.
        """
        return np.stack([self.embed(f) for f in faces])

    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Convert aligned 112x112 face to float32 for model input.

        The arcface HEF has normalization baked in (UINT8 input layer),
        so we only need to cast to float32 — the Hailo SDK handles
        quantization internally via the VStream format conversion.
        """
        return aligned_face.astype(np.float32)

    def warmup(self) -> None:
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        self.embed(dummy)

    def release(self) -> None:
        if getattr(self, "_owns_device", False) and hasattr(self, "_device"):
            del self._device
