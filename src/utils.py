#Author : Aditya
"""
utils.py
--------
Utility functions for computing facial geometry metrics
used in drowsiness and yawn detection.
"""

import numpy as np
from scipy.spatial import distance as dist

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
MOUTH_IDX_8   = [61, 291, 0, 17, 269, 405, 321, 375]


def eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in eye_pts]
    A = dist.euclidean(p2, p6)
    B = dist.euclidean(p3, p5)
    C = dist.euclidean(p1, p4)
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth_pts):
    pts = [np.array(p) for p in mouth_pts]
    A = dist.euclidean(pts[1], pts[7])
    B = dist.euclidean(pts[2], pts[6])
    C = dist.euclidean(pts[3], pts[5])
    D = dist.euclidean(pts[0], pts[4])
    return (A + B + C) / (3.0 * D)


def get_landmark_coords(face_landmarks, indices, frame_shape):
    h, w = frame_shape[:2]
    return [
        (face_landmarks.landmark[i].x * w,
         face_landmarks.landmark[i].y * h)
        for i in indices
    ]


def perclos(closed_frames, total_frames):
    if total_frames == 0:
        return 0.0
    return closed_frames / total_frames