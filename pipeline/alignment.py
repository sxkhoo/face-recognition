"""Face alignment via 5-point landmark similarity transform to 112x112."""

from __future__ import annotations

import cv2
import numpy as np

# Standard ArcFace 5-point reference template for 112x112 output.
# Points: left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner.
ARCFACE_REF_5PT = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

OUTPUT_SIZE = (112, 112)


def estimate_similarity_transform(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """Estimate 2x3 similarity transform matrix from src to dst points.

    Uses Umeyama's method (rotation + uniform scale + translation).

    Args:
        src: (N, 2) source landmark points.
        dst: (N, 2) destination reference points.

    Returns:
        (2, 3) affine matrix.
    """
    num = src.shape[0]
    dim = 2

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    src_var = np.sum(src_demean**2) / num

    H = (dst_demean.T @ src_demean) / num  # (2, 2)

    U, S, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.ones(dim)
    if np.linalg.det(H) < 0:
        d[dim - 1] = -1

    T_rot = U @ np.diag(d) @ Vt

    scale = np.sum(S * d) / src_var

    t = dst_mean - scale * (T_rot @ src_mean)

    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = scale * T_rot
    M[:2, 2] = t

    return M


def align_face(
    frame: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    reference: np.ndarray = ARCFACE_REF_5PT,
) -> np.ndarray:
    """Align and crop a face using 5-point landmarks.

    Args:
        frame: Full BGR image (HxWx3).
        landmarks_5: (5, 2) detected landmark coordinates.
        output_size: (width, height) of output crop.
        reference: (5, 2) reference landmark template.

    Returns:
        Aligned face crop, output_size BGR uint8.
    """
    M = estimate_similarity_transform(
        landmarks_5.astype(np.float64),
        reference.astype(np.float64),
    )
    aligned = cv2.warpAffine(
        frame, M, output_size, borderValue=(0, 0, 0)
    )
    return aligned
