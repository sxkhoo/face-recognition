#!/usr/bin/env python3
"""Live camera face recognition pipeline.

Usage:
    python main.py                    # defaults from config.yaml
    python main.py --config my.yaml   # custom config
    python main.py --source video.mp4 # override camera source
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

from pipeline.alignment import align_face
from pipeline.alert import AlertManager
from pipeline.cpu_fallbacks import DummyEmbedder, OpenCVCascadeDetector
from pipeline.interfaces import FaceDetector, FaceEmbedder, LivenessChecker
from pipeline.liveness import StubLivenessChecker
from pipeline.matching import WatchlistManager
from pipeline.quality import assess_quality
from pipeline.types import Alert, Detection, PipelineResult

logger = logging.getLogger("face_recognition")


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def maybe_create_hailo_device(cfg: Dict[str, Any]):
    needs_hailo = (
        cfg["detection"]["backend"] == "hailo"
        or cfg["embedding"]["backend"] == "hailo"
    )
    if not needs_hailo:
        return None

    from hailo_utils import create_shared_vdevice

    return create_shared_vdevice()


def build_detector(cfg: Dict[str, Any], hailo_device=None) -> FaceDetector:
    backend = cfg["detection"]["backend"]
    if backend == "hailo":
        from hailo_utils.detector import HailoFaceDetector
        return HailoFaceDetector(
            hef_path=cfg["detection"]["hef_path"],
            score_threshold=cfg["detection"]["score_threshold"],
            nms_iou_threshold=cfg["detection"]["nms_iou_threshold"],
            vdevice=hailo_device,
        )
    else:
        logger.info("Using OpenCV cascade detector (CPU fallback).")
        return OpenCVCascadeDetector()


def build_embedder(cfg: Dict[str, Any], hailo_device=None) -> FaceEmbedder:
    backend = cfg["embedding"]["backend"]
    if backend == "hailo":
        from hailo_utils.embedder import HailoFaceEmbedder
        return HailoFaceEmbedder(
            hef_path=cfg["embedding"]["hef_path"],
            embedding_dim=cfg["embedding"]["embedding_dim"],
            vdevice=hailo_device,
        )
    else:
        logger.info("Using dummy embedder (CPU fallback).")
        return DummyEmbedder(embedding_dim=cfg["embedding"]["embedding_dim"])


def build_liveness(cfg: Dict[str, Any]) -> LivenessChecker:
    # Swap in real implementation later
    return StubLivenessChecker()


def process_frame(
    frame: np.ndarray,
    frame_idx: int,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    liveness: LivenessChecker,
    watchlist: WatchlistManager,
    alert_mgr: AlertManager,
    cfg: Dict[str, Any],
) -> List[PipelineResult]:
    """Run full pipeline on one frame. Returns results per detected face."""

    results: List[PipelineResult] = []
    detections = detector.detect(frame)

    # Cap faces per frame
    max_faces = cfg["detection"].get("max_faces", 10)
    detections = detections[:max_faces]

    for det in detections:
        pr = PipelineResult(detection=det)

        # 1. Align
        aligned = align_face(frame, det.landmarks_5)

        # 2. Liveness (if enabled)
        if cfg["liveness"]["enabled"]:
            is_live, liveness_score = liveness.check(aligned)
            pr.liveness_passed = is_live
            if not is_live:
                results.append(pr)
                continue
        else:
            pr.liveness_passed = True

        # 3. Quality check
        qr = assess_quality(
            aligned,
            det,
            blur_threshold=cfg["quality"]["blur_threshold"],
            min_detection_score=cfg["quality"]["min_detection_score"],
            min_face_size=cfg["quality"]["min_face_size"],
        )
        pr.quality = qr
        if not qr.passed:
            results.append(pr)
            continue

        # 4. Embed
        embedding = embedder.embed(aligned)
        pr.embedding = embedding

        # 5. Match
        match = watchlist.match(embedding)
        pr.match = match

        # 6. Alert
        alert = alert_mgr.try_alert(match, det.bbox, frame_idx)
        pr.alert = alert

        results.append(pr)

    return results


def draw_results(frame: np.ndarray, results: List[PipelineResult]) -> np.ndarray:
    """Draw bounding boxes and labels on frame for display."""
    vis = frame.copy()
    for pr in results:
        det = pr.detection
        x1, y1, x2, y2 = det.bbox.astype(int)

        # Color: green=matched, yellow=detected but no match, red=failed quality
        if pr.match and pr.match.matched:
            color = (0, 255, 0)
            label = f"{pr.match.name} ({pr.match.similarity:.2f})"
        elif pr.quality and not pr.quality.passed:
            color = (0, 165, 255)  # Orange
            label = f"low quality: {pr.quality.reason}"
        else:
            color = (255, 255, 0)  # Cyan
            label = f"face ({det.score:.2f})"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw landmarks
        for lx, ly in det.landmarks_5.astype(int):
            cv2.circle(vis, (lx, ly), 2, (0, 255, 0), -1)

    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--source", default=None, help="Override camera source")
    parser.add_argument("--display", action="store_true", help="Show live window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=infinite)")
    parser.add_argument("--save-first-frame", default=None, help="Save first frame to this path for debugging")
    args = parser.parse_args()

    # Config
    cfg = load_config(args.config)
    log_level = cfg.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Build components
    hailo_device = maybe_create_hailo_device(cfg)
    detector = build_detector(cfg, hailo_device)
    embedder = build_embedder(cfg, hailo_device)
    liveness = build_liveness(cfg)

    watchlist = WatchlistManager(
        embedding_dim=cfg["embedding"]["embedding_dim"],
        threshold=cfg["matching"]["threshold"],
    )
    watchlist.load(
        cfg["matching"]["watchlist_embeddings"],
        cfg["matching"]["watchlist_identities"],
    )

    alert_mgr = AlertManager(
        cooldown_seconds=cfg["alert"]["cooldown_seconds"],
        camera_id=cfg["alert"]["camera_id"],
    )

    # Video source
    source = args.source or cfg["camera"]["source"]
    # Try int conversion for device index
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    logger.info("Opening video source: %s", source)
    # Try V4L2 backend first (GStreamer often fails with raw device indices on NXP).
    # If source is a GStreamer pipeline string, fall back to default backend.
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.warning("V4L2 backend failed for device %d, trying default backend.", source)
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Failed to open video source: %s", source)
        sys.exit(1)

    # Apply camera resolution from config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps"])
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Camera resolution: requested %dx%d@%d, actual %dx%d@%.1f",
        cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"],
        actual_w, actual_h, actual_fps,
    )

    logger.info("Pipeline running. Press 'q' to quit.")
    frame_idx = 0
    fps_counter = 0
    fps_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            # First-frame diagnostics
            if frame_idx == 0:
                logger.info(
                    "First frame: shape=%s dtype=%s min=%d max=%d",
                    frame.shape, frame.dtype, int(frame.min()), int(frame.max()),
                )
                save_path = args.save_first_frame or "/tmp/debug_frame_0.jpg"
                cv2.imwrite(save_path, frame)
                logger.info("Saved first frame to %s", save_path)

            results = process_frame(
                frame, frame_idx, detector, embedder, liveness,
                watchlist, alert_mgr, cfg,
            )

            # FPS tracking
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 2.0:
                fps = fps_counter / elapsed
                logger.info("FPS: %.1f | faces: %d | frame: %d", fps, len(results), frame_idx)
                fps_counter = 0
                fps_time = time.time()

            # Display
            if args.display:
                vis = draw_results(frame, results)
                if elapsed > 0:
                    display_fps = fps_counter / elapsed
                    cv2.putText(vis, f"FPS: {display_fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Face Recognition", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        detector.release()
        embedder.release()
        if hailo_device is not None:
            del hailo_device
        logger.info("Pipeline stopped. Processed %d frames, %d alerts.",
                     frame_idx, len(alert_mgr.alert_log))


if __name__ == "__main__":
    main()
