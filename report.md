# Face Recognition System — Project Report

## Overview

This project is a real-time face recognition system built for an NXP board with a Hailo-8 AI chip. It watches a camera feed, detects faces, and checks if they match anyone in a "watchlist" — a database of known people. When a match is found, the system identifies who the person is and can trigger alerts.

## How It Works

The system processes each video frame through a five-step pipeline:

### Step 1: Face Detection

The camera captures a 1280x720 video frame. A model called SCRFD scans the image and finds faces. For each face, it outputs:
- A bounding box (rectangle around the face)
- Five landmark points (two eyes, nose tip, two mouth corners)
- A confidence score (how sure it is that this is a real face)

### Step 2: Face Alignment

The five landmark points are used to "straighten" the face. Using a similarity transform, the face is warped and cropped into a standard 112x112 pixel image — always upright, centred, and consistently sized. This makes comparison fair regardless of where in the frame the face appeared.

### Step 3: Quality Check

Before spending time on recognition, the system checks if the face image is good enough:
- **Sharpness**: Is the image too blurry? (measured using Laplacian variance)
- **Size**: Was the face too small in the original frame?
- **Confidence**: Was the detector confident this is actually a face?

If any check fails, the face is skipped for that frame.

### Step 4: Embedding

The aligned face is fed into an ArcFace neural network, which converts the face image into a list of 512 numbers — called an "embedding" or "face fingerprint." This fingerprint captures the unique characteristics of a person's face. Two photos of the same person produce similar fingerprints; two different people produce very different ones.

### Step 5: Matching

The fingerprint is compared against all enrolled fingerprints in the watchlist using cosine similarity. If the closest match is within the threshold (distance < 0.4), the person is identified. Otherwise, they appear as an unknown face.

## Performance

| Metric | Value |
|--------|-------|
| Frame rate | ~5.6-5.9 FPS (two faces in frame) |
| Camera resolution | 1280x720 @ 30fps |
| Detection input | 640x640 (resized internally) |
| Embedding input | 112x112 aligned crop |
| Embedding dimension | 512 (float32) |
| Matching speed | Sub-millisecond (FAISS inner product search) |

Both the detector and embedder run on the Hailo-8 accelerator, not the CPU. The main bottleneck is the sequential Hailo inference calls per face.

## Enrolment

People are enrolled into the watchlist by capturing face photos from the live camera. The recommended approach is:
- 15 captures: 5 angles (front, left, right, up, down) at 3 distances (close, medium, far)
- Each capture is automatically augmented with downscaled, rotated, blurred, and brightness-shifted versions
- This produces ~165 embeddings per person, covering a wide range of conditions

Each capture is augmented with the following transformations to improve robustness:

| Augmentation | Variants | Purpose |
|---|---|---|
| Downscale + upscale (56px, 40px) | 2 | Simulates medium and far distance |
| Rotation (-10°, -5°, +5°, +10°) | 4 | Simulates head tilt and landmark misalignment |
| Gaussian blur (kernel 3, 5) | 2 | Simulates motion blur and focus issues |
| Brightness shift (-30, +30) | 2 | Simulates lighting changes |

This gives 10 augmented embeddings per capture, plus the original — so 15 captures produce 165 total embeddings per person.

Note: a 28px downscale augmentation was originally included to simulate very far distances, but was removed because it degraded faces so heavily that different people became indistinguishable — causing cross-identity false matches.

Enrolling from the same camera that does the recognition is important. Using phone photos for enrollment and a different camera for recognition creates a "domain gap" where the face looks different enough to fail matching.

## Key Issues Discovered During Development

### Double Normalization Bug

During initial testing, the system identified every person as "Sean" regardless of who was in front of the camera. The root cause was a preprocessing bug in the face embedder.

The ArcFace model file (HEF) had input normalization already built into it — it expects raw pixel values (0 to 255) and handles the scaling internally. However, the code was also applying normalization manually, converting pixels from [0, 255] to [-1, 1] before sending them to the model. This double normalization crushed the pixel values into a tiny range, washing out the differences between faces.

The result: every face produced nearly identical embeddings. Sean had the most photos in the watchlist, so everyone matched to Sean.

The fix was a single line — removing the manual normalization. After the fix, the cosine distance between different people jumped from ~0.04 (nearly identical) to ~0.91 (clearly different).

### Domain Gap: Phone Photos vs Board Camera

After fixing the normalization bug, the system was enrolled using phone selfies. It worked up close but failed at medium distances. Investigation revealed that even when quality checks passed, the match distances were too high.

The cause was a "domain gap" — the phone camera and the board's camera produce visually different images of the same face. Different lenses, different sensors, different lighting behaviour. The embeddings from phone photos lived in a slightly different region of the 512-dimensional space than embeddings from the board camera.

The solution was to re-enrol everyone using the board's own camera at various distances (close, medium, far). This ensured the enrolled embeddings matched the live camera's characteristics. After re-enrollment, recognition worked reliably at all distances where the face was large enough to detect.

### Quality Gate Tuning

The blur quality gate was originally set to a Laplacian variance threshold of 80. During testing, this rejected around 40% of close-up face captures. The cause was autofocus instability — when a person stands close to the camera, the camera's autofocus sometimes struggles, producing frames with varying sharpness.

Lowering the threshold from 80 to 30 allowed more frames through while still rejecting genuinely unusable images (motion blur, completely out of focus). The result was more consistent recognition at close range without noticeable false matches.

## Known Limitations

### 1. Distance sensitivity

At far distances, the face occupies fewer pixels. When a tiny face is stretched to 112x112 for recognition, important details are lost. The system works most reliably within 0.5-2 metres. Beyond that, recognition accuracy drops.

Augmentation (simulating distance by downscaling) helps but cannot fully replace real captures at the actual distance.

### 2. Facial expressions (smiling problem)

When a person smiles broadly, their face changes shape — the mouth widens, cheeks push up, eyes narrow. This shifts the embedding away from their neutral-expression enrolment photos. The system may briefly fail to recognise them or show them as "unknown."

**Possible solutions:**

- **Enrol with expressions**: Capture photos while smiling, neutral, and with other common expressions. This is the simplest and most effective approach — the watchlist then contains embeddings for multiple expressions.

- **Expression-invariant models**: Some newer face recognition models (e.g., AdaFace, ElasticFace) are specifically trained to be more robust to expression changes. Switching to one of these models could reduce the problem at the model level.

- **Temporal smoothing**: Instead of deciding identity frame-by-frame, the system could maintain a short history (e.g., last 5-10 frames) and use majority voting. A single frame where smiling causes a mismatch would be overridden by the surrounding frames where the person was recognised. This would prevent the "flicker" effect where someone briefly shows as unknown.

- **Multi-template averaging**: When matching, instead of comparing against each enrolled embedding independently, the system could compare against an averaged embedding per identity. This smooths out the effect of any single unusual enrolment image.

- **Increase matching threshold**: A looser threshold (e.g., 0.5 instead of 0.4) accepts more variation, but risks matching the wrong person. This is a trade-off that must be tuned per deployment.

### 3. Occlusion (hands, hair, masks)

When part of the face is covered (touching hair, hand on face, wearing a mask), the landmark detection becomes less accurate and the visible face area changes. This shifts the embedding. Similar to the expression issue, enrolling with common occlusions helps.

### 4. Lighting changes

Moving between bright and shadowed areas changes the face appearance. The brightness augmentation during enrolment helps to some extent, but extreme lighting changes (backlit, very dark) can still cause mismatches.

### 5. Single-process Hailo constraint

The Hailo-8 device can only be used by one process at a time. If a previous pipeline is still running, a new one will fail with a "no free devices" error. The previous process must be killed first.

### 6. No liveness detection

The system currently has no anti-spoofing. It will recognise a person from a printed photo or phone screen held up to the camera. The liveness checker interface exists in the code but is currently a stub that always returns "live."

## Architecture Decisions

- **Hailo-8 for inference**: Both detection and embedding run on the dedicated AI chip, freeing the CPU for frame capture and post-processing.
- **FAISS for matching**: Enables fast nearest-neighbour search even with hundreds of enrolled embeddings. Falls back to NumPy if FAISS is not installed.
- **Pluggable backends**: The detector, embedder, and liveness checker are behind abstract interfaces. CPU fallbacks exist for laptop testing without hardware.
- **Augmented enrollment**: Each captured face generates multiple augmented variants automatically, reducing the number of manual captures needed for robust recognition.

## Future Improvements

- **Temporal smoothing / tracking**: Smooth identity labels across frames to eliminate flicker. Could use a simple face tracker (e.g., IoU-based or centroid-based) to maintain identity continuity.
- **Liveness detection**: Implement the liveness checker to reject spoofing attempts.
- **Expression-robust model**: Evaluate AdaFace or ElasticFace as a replacement for ArcFace-R50.
- **Dynamic enrolment**: Allow on-the-fly enrolment from the live display (e.g., click on an unknown face to enrol it).
- **Formal evaluation with metrics**: Run structured testing using the `evaluate.py` script on a labelled dataset. Measure metrics such as mean Average Precision (mAP), True Accept Rate at various False Accept Rates (TAR@FAR), Equal Error Rate (EER), and find the optimal matching threshold per deployment. This would replace the current trial-and-error threshold tuning with data-driven decisions.
- **Resolution vs FPS trade-off**: Test lower camera resolutions (e.g., 640x480) to improve frame rate at the cost of reduced detection range. Profile to find the optimal balance for the deployment environment.
- **Multi-camera support**: Extend the pipeline to handle multiple camera streams.
- **Alert integration**: Connect the alert system to external services (email, Slack, webhook).
