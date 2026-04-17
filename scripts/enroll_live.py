#!/usr/bin/env python3
"""Enroll a person into the watchlist using the board's live camera.

Captures faces at multiple distances for robust matching.
Press SPACE to capture a frame, 'q' to finish and enroll.

Usage:
    python scripts/enroll_live.py --name "Sean" --id sean
    python scripts/enroll_live.py --name "Sean" --id sean --num-captures 10
    python scripts/enroll_live.py --name "Sean" --id sean --headless
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.alignment import align_face
from pipeline.interfaces import FaceDetector, FaceEmbedder
from pipeline.matching import WatchlistManager
from pipeline.quality import laplacian_blur_score

logger = logging.getLogger("enroll_live")


def augment_aligned_face(
    aligned: np.ndarray,
    embedder: FaceEmbedder,
    save_dir: str | None = None,
    capture_idx: int = 0,
) -> list[np.ndarray]:
    """Generate augmented embeddings from a single aligned 112x112 face.

    Simulates what the face looks like at various distances by
    downscaling then upscaling, plus minor blur/brightness shifts.

    Returns list of augmented embeddings (does NOT include the original).
    """
    h, w = aligned.shape[:2]  # 112x112
    augmented_embeddings: list[np.ndarray] = []
    aug_idx = 0

    # 1. Downscale+upscale to simulate distance
    #    56px = medium distance, 40px = far
    #    (28px removed — too degraded, causes cross-identity confusion)
    for sim_size in [56, 40]:
        small = cv2.resize(aligned, (sim_size, sim_size), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        emb = embedder.embed(restored)
        augmented_embeddings.append(emb)
        aug_idx += 1
        if save_dir:
            path = Path(save_dir) / f"capture_{capture_idx:02d}_aug_dist{sim_size}.jpg"
            cv2.imwrite(str(path), restored)

    # 2. Small rotations (simulates head tilt / misaligned landmarks)
    center = (w // 2, h // 2)
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(aligned, M, (w, h), borderValue=(0, 0, 0))
        emb = embedder.embed(rotated)
        augmented_embeddings.append(emb)
        aug_idx += 1
        if save_dir:
            path = Path(save_dir) / f"capture_{capture_idx:02d}_aug_rot{angle}.jpg"
            cv2.imwrite(str(path), rotated)

    # 4. Slight Gaussian blur (simulates motion blur / focus issues)
    for ksize in [3, 5]:
        blurred = cv2.GaussianBlur(aligned, (ksize, ksize), 0)
        emb = embedder.embed(blurred)
        augmented_embeddings.append(emb)
        aug_idx += 1

    # 5. Brightness shifts (simulates lighting variation)
    for delta in [-30, 30]:
        shifted = np.clip(aligned.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        emb = embedder.embed(shifted)
        augmented_embeddings.append(emb)
        aug_idx += 1

    return augmented_embeddings


def _distance_guidance(captured: int, total: int) -> str:
    """Return distance instruction based on capture progress."""
    progress = captured / total
    if progress < 0.33:
        return "Stand CLOSE to camera"
    elif progress < 0.66:
        return "Step back to MEDIUM distance"
    else:
        return "Step back to FAR distance"


def capture_faces(
    detector: FaceDetector,
    embedder: FaceEmbedder,
    cfg: dict,
    num_captures: int = 10,
    headless: bool = False,
    save_dir: str | None = None,
) -> list[np.ndarray]:
    """Capture face embeddings from the live camera with display.

    Shows a live preview window with:
    - Face bounding box (green=good quality, red=bad)
    - Distance guidance (close/medium/far)
    - Capture count and flash on capture
    - Auto-captures every ~1.5s when face quality is good

    Press 'q' to finish early.

    Args:
        detector: Face detector instance.
        embedder: Face embedder instance.
        cfg: Pipeline config dict.
        num_captures: Target number of captures.
        headless: If True, no GUI window — auto-capture with console output.
        save_dir: If set, save aligned face crops here for review.

    Returns:
        List of L2-normalized embedding vectors.
    """
    import time

    source = cfg["camera"]["source"]
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps"])

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Warm up camera
    for _ in range(10):
        cap.read()

    embeddings: list[np.ndarray] = []
    num_real_captures = 0
    min_quality_score = cfg["quality"]["min_detection_score"]
    min_face_size = cfg["quality"]["min_face_size"]
    blur_threshold = cfg["quality"]["blur_threshold"]

    flash_until = 0.0  # timestamp until which to show green flash

    if headless:
        print(f"Headless mode: auto-capturing {num_captures} faces.")
        print("Move closer/further between captures for variety.")
        time.sleep(2)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera read failed.")
                break

            now = time.time()
            dets = detector.detect(frame)

            # Pick largest face
            best_det = None
            if dets:
                best_det = max(dets, key=lambda d: d.width * d.height)

            # Compute quality info
            face_size = 0
            blur = 0.0
            quality_ok = False
            aligned = None

            if best_det:
                face_size = min(best_det.width, best_det.height)
                aligned = align_face(frame, best_det.landmarks_5)
                blur = laplacian_blur_score(aligned)
                quality_ok = (best_det.score >= min_quality_score
                              and face_size >= min_face_size
                              and blur >= blur_threshold)

            # Manual capture via SPACE — show preview for confirmation
            captured_this_frame = False

            if getattr(capture_faces, '_manual_trigger', False):
                capture_faces._manual_trigger = False
                if best_det and aligned is not None:
                    if not quality_ok:
                        print(f"  Skipped: low quality (size={face_size:.0f}px blur={blur:.1f})")
                    else:
                        # Show aligned face preview for review
                        preview = cv2.resize(aligned, (336, 336), interpolation=cv2.INTER_LINEAR)
                        info_text = f"blur={blur:.1f}  size={face_size:.0f}px"
                        cv2.putText(preview, info_text, (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        cv2.putText(preview, "SPACE=accept  R=reject", (10, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.imshow("Preview", preview)

                        # Wait for accept/reject
                        while True:
                            rkey = cv2.waitKey(0) & 0xFF
                            if rkey == ord(" "):
                                # Accept
                                num_real_captures += 1
                                emb = embedder.embed(aligned)
                                embeddings.append(emb)

                                aug_embs = augment_aligned_face(
                                    aligned, embedder,
                                    save_dir=save_dir,
                                    capture_idx=num_real_captures,
                                )
                                embeddings.extend(aug_embs)

                                flash_until = time.time() + 0.3
                                captured_this_frame = True

                                if save_dir:
                                    path = Path(save_dir) / f"capture_{num_real_captures:02d}_original.jpg"
                                    cv2.imwrite(str(path), aligned)

                                print(f"  [{num_real_captures}/{num_captures}] "
                                      f"size={face_size:.0f}px blur={blur:.1f} "
                                      f"score={best_det.score:.2f} "
                                      f"(+{len(aug_embs)} augmented, {len(embeddings)} total)")
                                break
                            elif rkey == ord("r"):
                                print("  Rejected, try again.")
                                break
                            elif rkey == ord("q"):
                                cv2.destroyWindow("Preview")
                                cap.release()
                                cv2.destroyAllWindows()
                                return embeddings

                        cv2.destroyWindow("Preview")

                        if num_real_captures >= num_captures:
                            print("Done capturing.")
                            break

            # Display
            if not headless:
                vis = frame.copy()

                # Green flash border on capture
                if now < flash_until:
                    cv2.rectangle(vis, (0, 0),
                                  (vis.shape[1] - 1, vis.shape[0] - 1),
                                  (0, 255, 0), 8)

                # Distance guidance banner
                guidance = _distance_guidance(num_real_captures, num_captures)
                bar_color = (80, 50, 20)
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 70), bar_color, -1)
                cv2.putText(vis, guidance, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(vis,
                            f"Captures: {num_real_captures}/{num_captures} ({len(embeddings)} total w/ aug)  |  SPACE=capture  q=done",
                            (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Face bounding box + info
                if best_det:
                    x1, y1, x2, y2 = best_det.bbox.astype(int)
                    color = (0, 255, 0) if quality_ok else (0, 0, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                    status = "READY" if quality_ok else "low quality"
                    label = f"{face_size:.0f}px | blur={blur:.0f} | {status}"
                    cv2.putText(vis, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Draw landmarks
                    for lx, ly in best_det.landmarks_5.astype(int):
                        cv2.circle(vis, (lx, ly), 2, (0, 255, 0), -1)

                    # Show "CAPTURED!" text briefly
                    if captured_this_frame:
                        cx = (x1 + x2) // 2 - 60
                        cy = (y1 + y2) // 2
                        cv2.putText(vis, "CAPTURED!", (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "No face detected", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Progress bar
                bar_y = vis.shape[0] - 20
                bar_w = vis.shape[1] - 20
                progress = num_real_captures / num_captures
                cv2.rectangle(vis, (10, bar_y), (10 + bar_w, bar_y + 15),
                              (100, 100, 100), -1)
                cv2.rectangle(vis, (10, bar_y),
                              (10 + int(bar_w * progress), bar_y + 15),
                              (0, 255, 0), -1)

                cv2.imshow("Enrollment", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    capture_faces._manual_trigger = True

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll face from live camera")
    parser.add_argument("--name", required=True, help="Person name")
    parser.add_argument("--id", default=None, help="Identity ID (auto-generated if omitted)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-captures", type=int, default=10,
                        help="Number of face captures to collect")
    parser.add_argument("--headless", action="store_true",
                        help="No GUI — auto-capture with console prompts")
    parser.add_argument("--save-crops", default=None,
                        help="Directory to save aligned face crops for review")
    parser.add_argument("--replace", action="store_true",
                        help="Remove existing entries for this identity before enrolling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    identity_id = args.id or f"{args.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"

    # Build components
    hailo_device = None
    if cfg["detection"]["backend"] == "hailo" or cfg["embedding"]["backend"] == "hailo":
        from hailo_utils import create_shared_vdevice
        hailo_device = create_shared_vdevice()

    if cfg["detection"]["backend"] == "hailo":
        from hailo_utils.detector import HailoFaceDetector
        detector: FaceDetector = HailoFaceDetector(
            hef_path=cfg["detection"]["hef_path"],
            score_threshold=cfg["detection"]["score_threshold"],
            nms_iou_threshold=cfg["detection"]["nms_iou_threshold"],
            vdevice=hailo_device,
        )
    else:
        from pipeline.cpu_fallbacks import OpenCVCascadeDetector
        detector = OpenCVCascadeDetector()

    if cfg["embedding"]["backend"] == "hailo":
        from hailo_utils.embedder import HailoFaceEmbedder
        embedder: FaceEmbedder = HailoFaceEmbedder(
            hef_path=cfg["embedding"]["hef_path"],
            vdevice=hailo_device,
        )
    else:
        from pipeline.cpu_fallbacks import DummyEmbedder
        embedder = DummyEmbedder(embedding_dim=cfg["embedding"]["embedding_dim"])

    # Load existing watchlist
    watchlist = WatchlistManager(
        embedding_dim=cfg["embedding"]["embedding_dim"],
        threshold=cfg["matching"]["threshold"],
    )
    watchlist.load(
        cfg["matching"]["watchlist_embeddings"],
        cfg["matching"]["watchlist_identities"],
    )

    # Remove existing entries if --replace
    if args.replace:
        removed = watchlist.remove(identity_id)
        if removed:
            logger.info("Replaced: removed %d existing entries for '%s' (id=%s)",
                        removed, args.name, identity_id)

    # Capture
    embeddings = capture_faces(
        detector=detector,
        embedder=embedder,
        cfg=cfg,
        num_captures=args.num_captures,
        headless=args.headless,
        save_dir=args.save_crops,
    )

    if not embeddings:
        logger.error("No faces captured.")
        sys.exit(1)

    # Enroll
    watchlist.enroll_batch(identity_id, args.name, np.stack(embeddings))
    Path(cfg["matching"]["watchlist_embeddings"]).parent.mkdir(parents=True, exist_ok=True)
    watchlist.save(
        cfg["matching"]["watchlist_embeddings"],
        cfg["matching"]["watchlist_identities"],
    )
    logger.info("Enrolled %d embeddings for '%s' (id=%s). Watchlist size: %d",
                len(embeddings), args.name, identity_id, watchlist.size)

    detector.release()
    embedder.release()
    if hailo_device is not None:
        del hailo_device


if __name__ == "__main__":
    main()
