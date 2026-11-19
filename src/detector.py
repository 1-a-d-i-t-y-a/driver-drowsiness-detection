"""
detector.py
-----------
DrowsinessDetector class: detects drowsiness via EAR,
yawning via MAR, and tracks PERCLOS metric.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Handle direct and package imports
try:
    from utils import (  # type: ignore
        LEFT_EYE_IDX,
        MOUTH_IDX_8,
        RIGHT_EYE_IDX,
        eye_aspect_ratio,
        get_landmark_coords,
        mouth_aspect_ratio,
        perclos,
    )
except ImportError:
    from src.utils import (
        LEFT_EYE_IDX,
        MOUTH_IDX_8,
        RIGHT_EYE_IDX,
        eye_aspect_ratio,
        get_landmark_coords,
        mouth_aspect_ratio,
        perclos,
    )

EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65
EAR_CONSEC_FRAMES = 20
YAWN_CONSEC_FRAMES = 15
PERCLOS_WINDOW = 75
PERCLOS_THRESHOLD = 0.15


def _generate_beep(frequency: int = 880, duration_ms: int = 800) -> Any:
    try:
        import pygame
        sample_rate = 44100
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples, False)
        wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        stereo = np.column_stack([wave, wave])
        return pygame.sndarray.make_sound(stereo)
    except Exception as exc:
        logger.debug("Sound synthesis unavailable: %s", exc)
        return None


class DrowsinessDetector:
    def __init__(self, enable_audio: bool = True, face_mesh_instance: Any = None):
        if face_mesh_instance is not None:
            self.face_mesh = face_mesh_instance
        else:
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                    self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
                else:
                    logger.info("MediaPipe 1.x detected (legacy mp.solutions unavailable).")
                    self.face_mesh = None
            except Exception as exc:
                logger.warning("Could not initialize MediaPipe FaceMesh: %s", exc)
                self.face_mesh = None

        self.audio_enabled = False
        self.alarm_sound = None
        self.yawn_sound = None

        if enable_audio:
            try:
                import pygame
                if os.environ.get("SDL_AUDIODRIVER") is None and sys.platform.startswith("linux"):
                    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                self.alarm_sound = _generate_beep(880, 800)
                self.yawn_sound = _generate_beep(440, 600)
                self.audio_enabled = True
            except Exception as exc:
                logger.warning("Audio playback disabled (headless or unsupported environment): %s", exc)
                self.audio_enabled = False

        self.alarm_playing = False
        self.ear_frame_count = 0
        self.mar_frame_count = 0
        self.perclos_window: list[int] = []
        self.total_alerts = 0
        self.total_yawns = 0
        self.session_start = time.time()
        self.current_ear = 0.0
        self.current_mar = 0.0
        self.status = "ALERT"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        face_detected = False
        left_pts, right_pts = [], []

        if self.face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results and getattr(results, "multi_face_landmarks", None):
                face_detected = True
                lm = results.multi_face_landmarks[0]

                left_pts = get_landmark_coords(lm, LEFT_EYE_IDX, frame.shape)
                right_pts = get_landmark_coords(lm, RIGHT_EYE_IDX, frame.shape)
                self.current_ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0

                mouth_pts = get_landmark_coords(lm, MOUTH_IDX_8, frame.shape)
                self.current_mar = mouth_aspect_ratio(mouth_pts)

                eye_closed = self.current_ear < EAR_THRESHOLD
                self.perclos_window.append(1 if eye_closed else 0)
                if len(self.perclos_window) > PERCLOS_WINDOW:
                    self.perclos_window.pop(0)

                # EAR Drowsiness logic
                if eye_closed:
                    self.ear_frame_count += 1
                    if self.ear_frame_count >= EAR_CONSEC_FRAMES:
                        self.status = "DROWSY"
                        if not self.alarm_playing:
                            self.total_alerts += 1
                            if self.audio_enabled and self.alarm_sound:
                                self.alarm_sound.play(-1)
                            self.alarm_playing = True
                else:
                    self.ear_frame_count = 0
                    if self.alarm_playing:
                        if self.audio_enabled and self.alarm_sound:
                            self.alarm_sound.stop()
                        self.alarm_playing = False
                    self.status = "ALERT"

                # MAR Yawn logic
                if self.current_mar > MAR_THRESHOLD:
                    self.mar_frame_count += 1
                    if self.mar_frame_count >= YAWN_CONSEC_FRAMES:
                        if self.status != "DROWSY":
                            self.status = "YAWNING"
                        if self.mar_frame_count == YAWN_CONSEC_FRAMES:
                            self.total_yawns += 1
                            if self.audio_enabled and self.yawn_sound:
                                self.yawn_sound.play()
                else:
                    self.mar_frame_count = 0

                # Draw visual landmarks
                for pts in (left_pts, right_pts):
                    pts_int = np.array(pts, dtype=np.int32)
                    cv2.polylines(frame, [pts_int], True, (0, 255, 200), 1)

        if not face_detected:
            self.status = "NO FACE"
            self.current_ear = 0.0
            self.current_mar = 0.0

        # Draw Heads-Up Display (HUD)
        color = (0, 255, 0)
        if self.status == "DROWSY":
            color = (0, 0, 255)
        elif self.status == "YAWNING":
            color = (0, 165, 255)
        elif self.status == "NO FACE":
            color = (128, 128, 128)

        if self.status == "DROWSY" and int(time.time() * 2) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"STATUS: {self.status}",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            2,
            cv2.LINE_AA,
        )

        if face_detected:
            metrics = [
                (f"EAR : {self.current_ear:.3f}", self.current_ear < EAR_THRESHOLD),
                (f"MAR : {self.current_mar:.3f}", self.current_mar > MAR_THRESHOLD),
            ]
            for i, (txt, warn) in enumerate(metrics):
                mc = (0, 100, 255) if warn else (200, 200, 200)
                cv2.putText(
                    frame, txt, (10, 75 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mc, 2, cv2.LINE_AA
                )

        elapsed = int(time.time() - self.session_start)
        mins, secs = divmod(elapsed, 60)
        for i, txt in enumerate(
            [
                f"Session  : {mins:02d}:{secs:02d}",
                f"Alerts   : {self.total_alerts}",
                f"Yawns    : {self.total_yawns}",
            ]
        ):
            cv2.putText(
                frame,
                txt,
                (10, h - 80 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (w - 190, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        return frame

    def run(self, source: int | str = 0) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[INFO] Drowsiness detector started. Press 'Q' to quit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                output = self.process_frame(frame)
                cv2.imshow("Driver Drowsiness Detection  |  Q to exit", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.audio_enabled:
                try:
                    import pygame
                    pygame.mixer.quit()
                except Exception:
                    pass

            elapsed = int(time.time() - self.session_start)
            mins, secs = divmod(elapsed, 60)
            print(f"\n{'='*45}\n  SESSION SUMMARY\n{'='*45}")
            print(f"  Duration       : {mins:02d}:{secs:02d}")
            print(f"  Drowsy alerts  : {self.total_alerts}")
            print(f"  Yawns detected : {self.total_yawns}")
            print("=" * 45)
