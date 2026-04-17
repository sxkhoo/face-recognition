# Face Recognition Pipeline

Real-time face recognition system running on an NXP board with Hailo-8 AI accelerator. Detects faces via SCRFD, generates embeddings via ArcFace, and matches against an enrolled watchlist using FAISS.

## Hardware Requirements

- NXP i.MX board (or similar ARM SBC) with Hailo-8 module
- USB camera (tested with V4L2-compatible cameras)
- HDMI display (optional, for live preview)

## Software Requirements

- Python 3.10+
- OpenCV (with V4L2 support)
- NumPy
- PyYAML
- FAISS (`faiss-cpu`)
- HailoRT Python bindings (`hailo_platform`)

## Project Structure

```
face_recognition/
├── main.py                     # Live pipeline entry point
├── config.yaml                 # All configuration and thresholds
├── hailo_utils/
│   ├── __init__.py             # Hailo SDK lazy imports and shared device
│   ├── detector.py             # SCRFD face detector on Hailo-8
│   └── embedder.py             # ArcFace face embedder on Hailo-8
├── pipeline/
│   ├── interfaces.py           # Abstract base classes (FaceDetector, FaceEmbedder, LivenessChecker)
│   ├── types.py                # Data classes (Detection, MatchResult, Alert, etc.)
│   ├── alignment.py            # 5-point landmark similarity transform (112x112 output)
│   ├── quality.py              # Blur, face size, and detection score quality gates
│   ├── matching.py             # Watchlist manager with FAISS index
│   ├── alert.py                # Alert manager with per-identity cooldown
│   ├── liveness.py             # Liveness checker (stub, not yet implemented)
│   └── cpu_fallbacks.py        # CPU-only detector/embedder for laptop testing
├── scripts/
│   ├── enroll_live.py          # Enroll faces from live camera with preview + augmentation
│   ├── enroll_face.py          # Enroll faces from image files
│   ├── demo_offline.py         # Run pipeline on a folder of images
│   └── evaluate.py             # Evaluate matching thresholds on labeled dataset
├── models/                     # Hailo HEF model files (not committed, see below)
│   ├── scrfd_10g.hef
│   └── arcface_r50.hef
└── watchlist/                  # Enrolled identity data (not committed)
    ├── embeddings.npy
    └── identities.json
```

## Setup

### 1. Place model files

Download or compile the following HEF files and place them in `models/`:

- `scrfd_10g.hef` — SCRFD-10G face detector
- `arcface_r50.hef` — ArcFace ResNet-50 face embedder

These are Hailo-compiled models and are not included in the repository due to size.

### 2. Configure

Edit `config.yaml` to match your setup:

```yaml
camera:
  source: 2          # V4L2 device index (check with: ls /dev/video*)
  width: 1280
  height: 720
  fps: 30
```

### 3. Enroll faces

Enroll people into the watchlist using the live camera:

```bash
# Enroll a new person (15 captures recommended)
python3 scripts/enroll_live.py --name "Alice" --id alice --num-captures 15

# Re-enroll (replaces existing data for this person)
python3 scripts/enroll_live.py --name "Alice" --id alice --num-captures 15 --replace

# Save aligned face crops for review
python3 scripts/enroll_live.py --name "Alice" --id alice --num-captures 15 --save-crops /tmp/alice_crops
```

During enrollment:
- **SPACE** captures a frame and shows a preview
- **SPACE** again to accept, **R** to reject
- **q** to finish early
- Capture at **close**, **medium**, and **far** distances
- Vary head angles: frontal, left, right, up, down

Each capture automatically generates augmented embeddings (downscaled, rotated, blurred, brightness-shifted) to improve robustness.

Alternatively, enroll from existing image files:

```bash
python3 scripts/enroll_face.py --name "Alice" --id alice --images photos/alice/
```

### 4. Run live recognition

```bash
# With display window
python3 main.py --config config.yaml --display

# Without display (logs only)
python3 main.py --config config.yaml

# Debug mode: set logging.level to "DEBUG" in config.yaml
```

Display overlay:
- **Green box**: recognized person (name and similarity score shown)
- **Orange box**: detected face but quality too low
- **Cyan box**: detected face but no watchlist match
- **FPS counter**: top-left corner

Press **q** to quit.

### 5. Offline demo

Run the pipeline on a folder of images:

```bash
python3 scripts/demo_offline.py --input test_images/ --display
python3 scripts/demo_offline.py --input test_images/ --output-dir results/
```

## Configuration Reference

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `camera.source` | | `2` | V4L2 device index or GStreamer pipeline string |
| `camera.width` / `height` | | `1280` / `720` | Requested camera resolution |
| `detection.score_threshold` | | `0.4` | Minimum detector confidence to keep a face |
| `detection.max_faces` | | `10` | Max faces processed per frame |
| `quality.blur_threshold` | | `30.0` | Laplacian variance minimum (higher = sharper required) |
| `quality.min_detection_score` | | `0.6` | Quality gate on detector confidence |
| `quality.min_face_size` | | `56` | Minimum face bbox side in pixels |
| `matching.threshold` | | `0.4` | Cosine distance threshold (lower = stricter matching) |
| `alert.cooldown_seconds` | | `30.0` | Seconds before same person triggers another alert |

## CPU-Only Testing

The pipeline can run on a laptop without Hailo hardware. Set backends to `"cpu"` in `config.yaml`:

```yaml
detection:
  backend: "cpu"
embedding:
  backend: "cpu"
```

This uses an OpenCV Haar cascade detector and a deterministic dummy embedder. Not accurate — only for verifying the pipeline runs end-to-end.

## Notes

- The Hailo-8 device can only be used by one process at a time. Kill any existing pipeline before starting a new one.
- HEF models have input normalization baked in. The code sends raw `[0, 255]` pixel values as float32 — do not add manual normalization.
- Watchlist data (`watchlist/`) contains biometric embeddings and is excluded from version control.
