#!/usr/bin/env python3
"""Evaluate matching thresholds on a labeled dataset.

Expected dataset layout:
    dataset/
        person_A/
            img1.jpg
            img2.jpg
        person_B/
            img1.jpg

Each subdirectory = one identity. The script computes all pairwise
distances and reports EER, TAR@FAR, and optimal threshold.

Usage:
    python scripts/evaluate.py --dataset eval_faces/ --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import build_detector, build_embedder, maybe_create_hailo_device
from pipeline.alignment import align_face

logger = logging.getLogger("evaluate")


def compute_embeddings(
    dataset_path: Path,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, List[np.ndarray]], int]:
    """Compute embeddings for all faces in a labeled directory."""

    hailo_device = maybe_create_hailo_device(cfg)
    detector = build_detector(cfg, hailo_device)
    embedder = build_embedder(cfg, hailo_device)

    identity_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    skipped = 0

    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue
        identity = person_dir.name

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue

            detections = detector.detect(frame)
            if not detections:
                skipped += 1
                continue

            aligned = align_face(frame, detections[0].landmarks_5)
            emb = embedder.embed(aligned)
            identity_embeddings[identity].append(emb)

    detector.release()
    embedder.release()
    if hailo_device is not None:
        del hailo_device

    return dict(identity_embeddings), skipped


def compute_pairs(
    identity_embeddings: Dict[str, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build genuine/impostor distance arrays."""

    genuine_dists = []
    impostor_dists = []

    identities = list(identity_embeddings.keys())

    # Genuine pairs: same identity
    for ident in identities:
        embs = identity_embeddings[ident]
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                dist = 1.0 - float(np.dot(embs[i], embs[j]))
                genuine_dists.append(dist)

    # Impostor pairs: different identity (all cross-pairs, capped at 50k)
    max_impostor = 50_000
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            embs_a = identity_embeddings[identities[i]]
            embs_b = identity_embeddings[identities[j]]
            for ea in embs_a:
                for eb in embs_b:
                    impostor_dists.append(1.0 - float(np.dot(ea, eb)))
                    if len(impostor_dists) >= max_impostor:
                        break
                if len(impostor_dists) >= max_impostor:
                    break
            if len(impostor_dists) >= max_impostor:
                break
        if len(impostor_dists) >= max_impostor:
            break

    return np.array(genuine_dists), np.array(impostor_dists)


def find_threshold_at_far(
    genuine: np.ndarray, impostor: np.ndarray, target_far: float = 0.01,
) -> Tuple[float, float]:
    """Find threshold that achieves target FAR, return (threshold, TAR)."""
    thresholds = np.linspace(0, 2, 1000)
    for t in thresholds:
        far = np.mean(impostor < t)
        if far >= target_far:
            tar = np.mean(genuine < t)
            return float(t), float(tar)
    return float(thresholds[-1]), 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face recognition thresholds")
    parser.add_argument("--dataset", required=True, help="Labeled dataset directory")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_path = Path(args.dataset)
    logger.info("Computing embeddings from %s ...", dataset_path)
    identity_embeddings, skipped = compute_embeddings(dataset_path, cfg)

    n_ids = len(identity_embeddings)
    n_embs = sum(len(v) for v in identity_embeddings.values())
    logger.info("Identities: %d, embeddings: %d, skipped: %d", n_ids, n_embs, skipped)

    if n_ids < 2:
        logger.error("Need >= 2 identities for evaluation.")
        sys.exit(1)

    genuine, impostor = compute_pairs(identity_embeddings)
    logger.info("Genuine pairs: %d, impostor pairs: %d", len(genuine), len(impostor))

    # Report
    for target_far in [0.1, 0.01, 0.001]:
        thresh, tar = find_threshold_at_far(genuine, impostor, target_far)
        logger.info("FAR=%.3f → threshold=%.4f, TAR=%.4f", target_far, thresh, tar)

    # EER approximation
    thresholds = np.linspace(0, 2, 2000)
    best_eer = 1.0
    best_t = 0.0
    for t in thresholds:
        far = float(np.mean(impostor < t))
        frr = float(np.mean(genuine >= t))
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer = eer
            best_t = t

    far_at_eer = float(np.mean(impostor < best_t))
    logger.info("Approximate EER: %.4f at threshold %.4f", far_at_eer, best_t)


if __name__ == "__main__":
    main()
