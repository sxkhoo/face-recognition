#!/usr/bin/env python3
"""Run the pipeline on a folder of test images (no camera needed).

Usage:
    python scripts/demo_offline.py --input test_images/ --config config.yaml
    python scripts/demo_offline.py --input test_images/ --display
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (
    build_detector,
    build_embedder,
    build_liveness,
    draw_results,
    maybe_create_hailo_device,
    process_frame,
)
from pipeline.alert import AlertManager
from pipeline.matching import WatchlistManager

logger = logging.getLogger("demo_offline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline face recognition demo")
    parser.add_argument("--input", required=True, help="Directory of test images")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--display", action="store_true", help="Show results in window")
    parser.add_argument("--output-dir", default=None, help="Save annotated images here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

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

    # Collect images
    input_dir = Path(args.input)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in extensions
    )

    if not image_paths:
        logger.error("No images found in %s", input_dir)
        sys.exit(1)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %d images...", len(image_paths))

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Cannot read: %s", img_path)
            continue

        results = process_frame(
            frame, idx, detector, embedder, liveness, watchlist, alert_mgr, cfg,
        )

        n_faces = len(results)
        n_alerts = sum(1 for r in results if r.alert is not None)
        logger.info("[%d/%d] %s — %d faces, %d alerts",
                     idx + 1, len(image_paths), img_path.name, n_faces, n_alerts)

        if args.display or args.output_dir:
            vis = draw_results(frame, results)
            if args.display:
                cv2.imshow("Demo", vis)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
            if args.output_dir:
                cv2.imwrite(str(out_dir / img_path.name), vis)

    if args.display:
        cv2.destroyAllWindows()
    detector.release()
    embedder.release()
    if hailo_device is not None:
        del hailo_device

    logger.info("Done. Total alerts: %d", len(alert_mgr.alert_log))


if __name__ == "__main__":
    main()
