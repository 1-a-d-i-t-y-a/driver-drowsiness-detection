from __future__ import annotations

import numpy as np
import pytest

from src.utils import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    perclos,
)


class TestFacialGeometryMetrics:
    def test_ear_open_eye(self):
        # Open eye geometry (p1=(0,0), p4=(10,0), p2=(3,3), p6=(3,-3), p3=(7,3), p5=(7,-3))
        # ||p2-p6|| = 6, ||p3-p5|| = 6, ||p1-p4|| = 10
        # EAR = (6 + 6) / (2 * 10) = 12 / 20 = 0.60
        open_eye = [
            (0.0, 0.0),    # p1 (corner)
            (3.0, 3.0),    # p2 (top-left)
            (7.0, 3.0),    # p3 (top-right)
            (10.0, 0.0),   # p4 (corner)
            (7.0, -3.0),   # p5 (bottom-right)
            (3.0, -3.0),   # p6 (bottom-left)
        ]
        ear = eye_aspect_ratio(open_eye)
        assert pytest.approx(ear, 0.01) == 0.60

    def test_ear_closed_eye(self):
        # Closed eye geometry (vertical distance drops to 0.5)
        # ||p2-p6|| = 1, ||p3-p5|| = 1, ||p1-p4|| = 10 -> EAR = (1 + 1)/20 = 0.10
        closed_eye = [
            (0.0, 0.0),
            (3.0, 0.5),
            (7.0, 0.5),
            (10.0, 0.0),
            (7.0, -0.5),
            (3.0, -0.5),
        ]
        ear = eye_aspect_ratio(closed_eye)
        assert pytest.approx(ear, 0.01) == 0.10
        assert ear < 0.22  # Below EAR_THRESHOLD

    def test_ear_division_by_zero_guard(self):
        # All points coincident
        zero_eye = [(0.0, 0.0)] * 6
        ear = eye_aspect_ratio(zero_eye)
        assert ear == 0.0

    def test_mar_neutral_mouth(self):
        # Horizontal width = 10, vertical heights = 3
        # MAR = (3 + 3 + 3) / (3 * 10) = 9 / 30 = 0.30
        neutral_mouth = [
            (0.0, 0.0),     # p1 (left corner)
            (2.5, 1.5),     # p2
            (5.0, 1.5),     # p3
            (7.5, 1.5),     # p4
            (10.0, 0.0),    # p5 (right corner)
            (7.5, -1.5),    # p6
            (5.0, -1.5),    # p7
            (2.5, -1.5),    # p8
        ]
        mar = mouth_aspect_ratio(neutral_mouth)
        assert pytest.approx(mar, 0.01) == 0.30
        assert mar < 0.65  # Below MAR_THRESHOLD

    def test_mar_yawn_mouth(self):
        # Vertical opening increases to 8
        # MAR = (8 + 8 + 8) / (3 * 10) = 24 / 30 = 0.80
        yawn_mouth = [
            (0.0, 0.0),
            (2.5, 4.0),
            (5.0, 4.0),
            (7.5, 4.0),
            (10.0, 0.0),
            (7.5, -4.0),
            (5.0, -4.0),
            (2.5, -4.0),
        ]
        mar = mouth_aspect_ratio(yawn_mouth)
        assert pytest.approx(mar, 0.01) == 0.80
        assert mar > 0.65  # Triggers YAWN threshold

    def test_mar_division_by_zero_guard(self):
        zero_mouth = [(0.0, 0.0)] * 8
        mar = mouth_aspect_ratio(zero_mouth)
        assert mar == 0.0

    def test_perclos_calculation(self):
        assert perclos(0, 75) == 0.0
        assert perclos(15, 75) == 0.20
        assert perclos(75, 75) == 1.0
        assert perclos(10, 0) == 0.0  # Zero frames guard


class TestDetectorInitialization:
    def test_headless_instantiation(self):
        from src.detector import DrowsinessDetector
        detector = DrowsinessDetector(enable_audio=False)
        assert detector.status == "ALERT"
        assert detector.total_alerts == 0
        assert detector.total_yawns == 0

        # Feed a blank frame
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed = detector.process_frame(blank_frame)
        assert processed.shape == (480, 640, 3)
        assert detector.status == "NO FACE"
