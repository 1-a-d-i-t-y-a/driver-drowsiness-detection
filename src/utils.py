"""
utils.py
--------
Utility functions for computing facial geometry metrics
used in drowsiness and yawn detection.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance as dist

# MediaPipe 468 FaceMesh indices for eyes and mouth
LEFT_EYE_IDX: list[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX: list[int] = [33, 160, 158, 133, 153, 144]
MOUTH_IDX_8: list[int] = [61, 291, 0, 17, 269, 405, 321, 375]


def eye_aspect_ratio(eye_pts: list[tuple[float, float]] | np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 2D coordinates.

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_pts) != 6:
        raise ValueError(f"EAR requires exactly 6 landmark points, got {len(eye_pts)}")
    p1, p2, p3, p4, p5, p6 = [np.array(p, dtype=np.float64) for p in eye_pts]
    a = float(dist.euclidean(p2, p6))
    b = float(dist.euclidean(p3, p5))
    c = float(dist.euclidean(p1, p4))
    if c == 0.0:
        return 0.0
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(mouth_pts: list[tuple[float, float]] | np.ndarray) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) from 8 2D coordinates.

    Formula:
        MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (3 * ||p1 - p5||)
    """
    if len(mouth_pts) != 8:
        raise ValueError(f"MAR requires exactly 8 landmark points, got {len(mouth_pts)}")
    pts = [np.array(p, dtype=np.float64) for p in mouth_pts]
    a = float(dist.euclidean(pts[1], pts[7]))
    b = float(dist.euclidean(pts[2], pts[6]))
    c = float(dist.euclidean(pts[3], pts[5]))
    d = float(dist.euclidean(pts[0], pts[4]))
    if d == 0.0:
        return 0.0
    return (a + b + c) / (3.0 * d)


def get_landmark_coords(
    face_landmarks, indices: list[int], frame_shape: tuple[int, ...]
) -> list[tuple[float, float]]:
    """Convert normalized facial landmarks into absolute frame pixel coordinates."""
    h, w = frame_shape[:2]
    return [
        (float(face_landmarks.landmark[i].x * w), float(face_landmarks.landmark[i].y * h))
        for i in indices
    ]


def perclos(closed_frames: int, total_frames: int) -> float:
    """
    Compute Percentage of Eye Closure (PERCLOS) over a window.

    Formula:
        PERCLOS = closed_frames / total_frames
    """
    if total_frames <= 0:
        return 0.0
    return float(closed_frames) / float(total_frames)
