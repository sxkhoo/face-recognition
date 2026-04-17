#!/usr/bin/env python3
"""Enroll a person into the watchlist from one or more face images.

Usage:
    python scripts/enroll_face.py --name "John Doe" --id john_01 --images img1.jpg img2.jpg
    python scripts/enroll_face.py --name "Jane" --id jane_01 --images photos/jane/ --config config.yaml
    python scripts/enroll_face.py --name "Jane" --id jane_01 --images photos/jane/ --enroll-detector cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.alignment import align_face
from pipeline.cpu_fallbacks import DummyEmbedder, OpenCVCascadeDetector
from pipeline.interfaces import FaceDetector, FaceEmbedder
from pipeline.matching import WatchlistManager
from pipeline.quality import assess_quality
from pipeline.types import Detection

logger = logging.getLogger("enroll")


def collect_image_paths(inputs: List[str]) -> List[Path]:
    """Expand directories and collect image paths."""
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths.extend(sorted(p.glob(ext)))
        elif p.is_file():
            paths.append(p)
        else:
            logger.warning("Skipping: %s (not found)", inp)
    return paths


def detect_with_rotations(
    detector: FaceDetector,
    frame: np.ndarray,
) -> tuple[np.ndarray, Detection] | tuple[None, None]:
    """Try face detection on common right-angle rotations.

    Many phone images are stored with orientation metadata that OpenCV ignores.
    Retrying a few rotations makes enrollment much more robust.
    """
    rotation_candidates = [
        ("0", frame),
        ("90_cw", cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)),
        ("180", cv2.rotate(frame, cv2.ROTATE_180)),
        ("90_ccw", cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_frame = None
    best_detection = None
    best_rotation = None

    for rotation_name, rotated in rotation_candidates:
        detections = detector.detect(rotated)
        if not detections:
            continue

        det = max(detections, key=lambda d: d.score)
        if best_detection is None or det.score > best_detection.score:
            best_frame = rotated
            best_detection = det
            best_rotation = rotation_name

    if best_detection is None:
        return None, None

    if best_rotation != "0":
        logger.info("Detected face after %s rotation.", best_rotation)

    return best_frame, best_detection


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a face into the watchlist")
    parser.add_argument("--name", required=True, help="Person name")
    parser.add_argument("--id", default=None, help="Identity ID (auto-generated if omitted)")
    parser.add_argument("--images", nargs="+", required=True, help="Image files or directories")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--enroll-detector",
        choices=("config", "cpu", "hailo"),
        default="config",
        help="Override detector backend for enrollment only",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    identity_id = args.id or f"{args.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    detector_backend = (
        cfg["detection"]["backend"]
        if args.enroll_detector == "config"
        else args.enroll_detector
    )
    if detector_backend != cfg["detection"]["backend"]:
        logger.info(
            "Enrollment detector override: using %s instead of config backend %s",
            detector_backend,
            cfg["detection"]["backend"],
        )

    hailo_device = None
    if detector_backend == "hailo" or cfg["embedding"]["backend"] == "hailo":
        from hailo_utils import create_shared_vdevice

        hailo_device = create_shared_vdevice()

    # Build components (use config backend selection)
    if detector_backend == "hailo":
        from hailo_utils.detector import HailoFaceDetector
        detector: FaceDetector = HailoFaceDetector(
            hef_path=cfg["detection"]["hef_path"],
            score_threshold=cfg["detection"]["score_threshold"],
            nms_iou_threshold=cfg["detection"]["nms_iou_threshold"],
            vdevice=hailo_device,
        )
    else:
        detector = OpenCVCascadeDetector()

    if cfg["embedding"]["backend"] == "hailo":
        from hailo_utils.embedder import HailoFaceEmbedder
        embedder: FaceEmbedder = HailoFaceEmbedder(
            hef_path=cfg["embedding"]["hef_path"],
            vdevice=hailo_device,
        )
    else:
        embedder = DummyEmbedder(embedding_dim=cfg["embedding"]["embedding_dim"])

    watchlist = WatchlistManager(
        embedding_dim=cfg["embedding"]["embedding_dim"],
        threshold=cfg["matching"]["threshold"],
    )
    watchlist.load(
        cfg["matching"]["watchlist_embeddings"],
        cfg["matching"]["watchlist_identities"],
    )

    # Process images
    image_paths = collect_image_paths(args.images)
    if not image_paths:
        logger.error("No images found.")
        sys.exit(1)

    logger.info("Processing %d images for '%s' (id=%s)", len(image_paths), args.name, identity_id)
    collected_embeddings: List[np.ndarray] = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Cannot read: %s", img_path)
            continue

        det_frame, det = detect_with_rotations(detector, frame)
        if det is None or det_frame is None:
            logger.warning("No face detected in %s", img_path)
            continue

        aligned = align_face(det_frame, det.landmarks_5)

        qr = assess_quality(
            aligned, det,
            blur_threshold=cfg["quality"]["blur_threshold"],
            min_detection_score=cfg["quality"]["min_detection_score"],
            min_face_size=cfg["quality"]["min_face_size"],
        )
        if not qr.passed:
            logger.warning("Quality too low for %s: %s", img_path, qr.reason)
            continue

        embedding = embedder.embed(aligned)
        collected_embeddings.append(embedding)
        logger.info("Accepted %s", img_path)

    enrolled_count = len(collected_embeddings)
    if enrolled_count > 0:
        watchlist.enroll_batch(
            identity_id, args.name, np.stack(collected_embeddings),
        )

    if enrolled_count > 0:
        # Ensure watchlist directory exists
        Path(cfg["matching"]["watchlist_embeddings"]).parent.mkdir(parents=True, exist_ok=True)
        watchlist.save(
            cfg["matching"]["watchlist_embeddings"],
            cfg["matching"]["watchlist_identities"],
        )
        logger.info("Enrolled %d embeddings for '%s'. Watchlist size: %d",
                     enrolled_count, args.name, watchlist.size)
    else:
        logger.error("No valid faces enrolled.")
        sys.exit(1)

    detector.release()
    embedder.release()
    if hailo_device is not None:
        del hailo_device


if __name__ == "__main__":
    main()
