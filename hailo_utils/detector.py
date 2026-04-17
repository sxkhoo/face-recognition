"""Hailo-8 face detector wrapper (SCRFD / RetinaFace).

This module contains the real Hailo inference logic. It requires:
  - hailo_platform (HailoRT Python bindings)
  - A compiled .hef model file

When HailoRT is not installed, importing this module will raise ImportError
at class instantiation, not at module import — so the rest of the codebase
stays usable on a laptop.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from hailo_utils import activate_network_group, import_hailo, resolve_vstream_name
from pipeline.interfaces import FaceDetector
from pipeline.types import Detection

logger = logging.getLogger(__name__)


class HailoFaceDetector(FaceDetector):
    """SCRFD / RetinaFace face detector running on Hailo-8.

    Expects a .hef compiled from SCRFD-10G or RetinaFace-MobileNet
    from the Hailo Model Zoo.

    The postprocessing (anchor decoding, NMS) depends on the exact model.
    Below is a scaffold that shows the HailoRT inference flow; you will
    need to adapt _postprocess() to your specific model's output tensors.
    """

    def __init__(
        self,
        hef_path: str,
        score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        vdevice=None,
    ) -> None:
        h = import_hailo()

        self._strides = (8, 16, 32)
        self._num_anchors = 2
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_size = input_size  # (width, height)
        self._anchor_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._empty_frames = 0

        # Load HEF and configure device
        logger.info("Loading HEF: %s", hef_path)
        self._hef = h["HEF"](hef_path)

        self._owns_device = vdevice is None
        self._device = vdevice if vdevice is not None else h["VDevice"]()
        configure_params = h["ConfigureParams"].create_from_hef(
            hef=self._hef, interface=h["HailoStreamInterface"].PCIe
        )
        self._network_group = self._device.configure(self._hef, configure_params)[0]

        # Create virtual stream params
        self._input_params = h["InputVStreamParams"].make(
            self._network_group,
            format_type=h["FormatType"].FLOAT32,
        )
        self._output_params = h["OutputVStreamParams"].make(
            self._network_group,
            format_type=h["FormatType"].FLOAT32,
        )
        self._input_name = resolve_vstream_name(self._input_params)
        self._first_frame_logged = False

        logger.info("Hailo face detector initialized.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h = import_hailo()

        orig_h, orig_w = frame.shape[:2]

        # Preprocess: aspect-preserving resize + pad + normalize.
        input_tensor, det_scale = self._preprocess(frame)

        # Run inference
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

        # --- First-frame diagnostics: dump all output tensor stats ---
        if not self._first_frame_logged:
            self._first_frame_logged = True
            logger.info("=== FIRST-FRAME DIAGNOSTICS ===")
            logger.info("Input frame: %s, det_scale=%.4f", frame.shape, det_scale)
            for name, tensor in output_dict.items():
                arr = np.asarray(tensor)
                logger.info(
                    "OUTPUT %-50s shape=%-20s dtype=%-10s min=%+.4f max=%+.4f mean=%+.4f",
                    name, str(arr.shape), str(arr.dtype),
                    float(arr.min()), float(arr.max()), float(arr.mean()),
                )

        # Postprocess: decode boxes, landmarks, apply NMS
        detections = self._postprocess(output_dict, orig_w, orig_h, det_scale)
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize with top-left padding to preserve aspect ratio."""
        input_w, input_h = self.input_size
        image_h, image_w = frame.shape[:2]
        if image_h <= 0 or image_w <= 0:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        image_ratio = float(image_h) / float(image_w)
        model_ratio = float(input_h) / float(input_w)
        if image_ratio > model_ratio:
            new_h = input_h
            new_w = max(1, int(new_h / image_ratio))
        else:
            new_w = input_w
            new_h = max(1, int(new_w * image_ratio))

        det_scale = new_h / float(image_h)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        # HEF has normalization baked in — send raw [0, 255] range.
        blob = padded.astype(np.float32)
        return blob, det_scale

    def _postprocess(
        self,
        output_dict: dict,
        orig_w: int,
        orig_h: int,
        det_scale: float,
    ) -> List[Detection]:
        """Decode raw SCRFD outputs into face detections."""
        stride_outputs = self._group_outputs_by_stride(output_dict)

        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        landmarks_list: list[np.ndarray] = []
        raw_score_stats: dict[int, float] = {}
        input_h = self.input_size[1]
        input_w = self.input_size[0]

        for stride in self._strides:
            branch = stride_outputs[stride]
            scores_map = branch["score"]
            bbox_map = branch["bbox"]
            kps_map = branch["kps"]

            height, width, _ = scores_map.shape
            anchor_centers = self._get_anchor_centers(height, width, stride)

            scores = scores_map.reshape(-1, self._num_anchors).reshape(-1)
            # HEF already outputs post-sigmoid probabilities in [0, 1].
            raw_score_stats[stride] = float(np.max(scores)) if scores.size else float("-inf")
            bbox_preds = bbox_map.reshape(-1, self._num_anchors, 4).reshape(-1, 4)
            bbox_preds = bbox_preds * stride
            kps_preds = kps_map.reshape(-1, self._num_anchors, 10).reshape(-1, 10)
            kps_preds = kps_preds * stride

            pos_inds = np.where(scores >= self.score_threshold)[0]
            if pos_inds.size == 0:
                continue

            bboxes = self._distance2bbox(anchor_centers, bbox_preds, (input_h, input_w))
            kpss = self._distance2kps(anchor_centers, kps_preds, (input_h, input_w))
            kpss = kpss.reshape(kpss.shape[0], 5, 2)

            scores_list.append(scores[pos_inds].astype(np.float32, copy=False))
            bboxes_list.append(bboxes[pos_inds].astype(np.float32, copy=False))
            landmarks_list.append(kpss[pos_inds].astype(np.float32, copy=False))

        if not scores_list:
            self._log_empty_detection_stats(raw_score_stats)
            return []

        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0) / det_scale
        landmarks = np.concatenate(landmarks_list, axis=0) / det_scale

        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, orig_w - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, orig_h - 1)
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, orig_w - 1)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, orig_h - 1)

        valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        if not np.any(valid):
            self._log_empty_detection_stats(raw_score_stats)
            return []

        scores = scores[valid]
        bboxes = bboxes[valid]
        landmarks = landmarks[valid]

        order = np.argsort(scores)[::-1]
        bboxes = bboxes[order]
        scores = scores[order]
        landmarks = landmarks[order]

        keep = self._nms(bboxes, scores, self.nms_iou_threshold)
        if not keep:
            self._log_empty_detection_stats(raw_score_stats)
            return []

        self._empty_frames = 0
        detections = [
            Detection(
                bbox=bboxes[idx].astype(np.float32, copy=False),
                landmarks_5=landmarks[idx].astype(np.float32, copy=False),
                score=float(scores[idx]),
            )
            for idx in keep
        ]
        return detections

    def _group_outputs_by_stride(self, output_dict: dict) -> Dict[int, Dict[str, np.ndarray]]:
        grouped = {stride: {} for stride in self._strides}
        unknown_outputs: list[str] = []

        for name, tensor in output_dict.items():
            matched = self._match_output_tensor(np.asarray(tensor))
            if matched is None:
                unknown_outputs.append(f"{name}:{np.asarray(tensor).shape}")
                continue

            stride, kind, tensor_hwc = matched
            existing = grouped[stride].get(kind)
            if existing is not None:
                logger.warning(
                    "Replacing duplicate SCRFD branch tensor for stride %d kind %s: old=%s new=%s",
                    stride,
                    kind,
                    existing.shape,
                    tensor_hwc.shape,
                )
            grouped[stride][kind] = tensor_hwc

        missing = [
            f"stride {stride}: missing {sorted({'score', 'bbox', 'kps'} - set(grouped[stride].keys()))}"
            for stride in self._strides
            if set(grouped[stride].keys()) != {"score", "bbox", "kps"}
        ]
        if missing:
            raise NotImplementedError(
                "Could not map HEF outputs to SCRFD branches. "
                f"Missing groups: {missing}. "
                f"Unknown outputs: {unknown_outputs or 'none'}"
            )

        logger.debug(
            "SCRFD output mapping: %s",
            {
                stride: {kind: grouped[stride][kind].shape for kind in grouped[stride]}
                for stride in self._strides
            },
        )
        return grouped

    def _match_output_tensor(self, tensor: np.ndarray) -> tuple[int, str, np.ndarray] | None:
        array = self._squeeze_batch_dim(tensor)
        if array.ndim != 3:
            return None

        for stride in self._strides:
            expected_h = self.input_size[1] // stride
            expected_w = self.input_size[0] // stride

            if array.shape[:2] == (expected_h, expected_w):
                tensor_hwc = array
                channels = tensor_hwc.shape[2]
            elif array.shape[1:3] == (expected_h, expected_w):
                tensor_hwc = np.transpose(array, (1, 2, 0))
                channels = tensor_hwc.shape[2]
            else:
                continue

            if channels == self._num_anchors:
                return stride, "score", tensor_hwc.astype(np.float32, copy=False)
            if channels == self._num_anchors * 4:
                return stride, "bbox", tensor_hwc.astype(np.float32, copy=False)
            if channels == self._num_anchors * 10:
                return stride, "kps", tensor_hwc.astype(np.float32, copy=False)

        return None

    @staticmethod
    def _squeeze_batch_dim(tensor: np.ndarray) -> np.ndarray:
        array = tensor
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        return array

    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached

        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = ((centers + 0.5) * stride).reshape(-1, 2)
        centers = np.repeat(centers, self._num_anchors, axis=0)
        self._anchor_cache[key] = centers
        return centers

    @staticmethod
    def _distance2bbox(
        points: np.ndarray,
        distance: np.ndarray,
        max_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

        if max_shape is not None:
            max_h, max_w = max_shape
            x1 = np.clip(x1, 0, max_w)
            y1 = np.clip(y1, 0, max_h)
            x2 = np.clip(x2, 0, max_w)
            y2 = np.clip(y2, 0, max_h)

        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(
        points: np.ndarray,
        distance: np.ndarray,
        max_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        preds = np.empty_like(distance, dtype=np.float32)
        preds[:, 0::2] = points[:, 0:1] + distance[:, 0::2]
        preds[:, 1::2] = points[:, 1:2] + distance[:, 1::2]

        if max_shape is not None:
            max_h, max_w = max_shape
            preds[:, 0::2] = np.clip(preds[:, 0::2], 0, max_w)
            preds[:, 1::2] = np.clip(preds[:, 1::2], 0, max_h)

        return preds

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if boxes.size == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(scores)[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0.0, inter / union, 0.0)
            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    def _log_empty_detection_stats(self, raw_score_stats: dict[int, float]) -> None:
        self._empty_frames += 1
        # Log every 5 frames during debug (change back to 30 once detection works)
        if self._empty_frames % 5 != 0:
            return

        stats = " ".join(
            f"s{stride}={raw_score_stats.get(stride, float('nan')):.4f}"
            for stride in self._strides
        )
        logger.info(
            "No detections for %d frames. score_threshold=%.3f post_sigmoid_max_scores=%s",
            self._empty_frames,
            self.score_threshold,
            stats,
        )

    def warmup(self) -> None:
        dummy = np.zeros(
            (self.input_size[1], self.input_size[0], 3), dtype=np.uint8
        )
        self.detect(dummy)

    def release(self) -> None:
        if getattr(self, "_owns_device", False) and hasattr(self, "_device"):
            del self._device
