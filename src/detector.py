"""
detector.py
-----------
DrowsinessDetector class: detects drowsiness via EAR,
yawning via MAR, and tracks PERCLOS metric.
"""

import cv2
import numpy as np
import mediapipe as mp
import pygame
import time

from utils import (
    eye_aspect_ratio, mouth_aspect_ratio,
    get_landmark_coords, perclos,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX_8,
)

EAR_THRESHOLD      = 0.25
MAR_THRESHOLD      = 0.65
EAR_CONSEC_FRAMES  = 20
YAWN_CONSEC_FRAMES = 15
PERCLOS_WINDOW     = 90
PERCLOS_THRESHOLD  = 0.15


def _generate_beep(frequency=880, duration_ms=800):
    sample_rate = 44100
    n_samples   = int(sample_rate * duration_ms / 1000)
    t           = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave        = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    stereo      = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)


class DrowsinessDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.alarm_sound    = _generate_beep(880, 800)
        self.yawn_sound     = _generate_beep(440, 600)
        self.alarm_playing  = False
        self.ear_frame_count   = 0
        self.mar_frame_count   = 0
        self.perclos_window    = []
        self.total_alerts      = 0
        self.total_yawns       = 0
        self.session_start     = time.time()
        self.current_ear       = 0.0
        self.current_mar       = 0.0
        self.status            = "ALERT"

    def process_frame(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0]

            left_pts  = get_landmark_coords(lm, LEFT_EYE_IDX,  frame.shape)
            right_pts = get_landmark_coords(lm, RIGHT_EYE_IDX, frame.shape)
            self.current_ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0

            mouth_pts        = get_landmark_coords(lm, MOUTH_IDX_8, frame.shape)
            self.current_mar = mouth_aspect_ratio(mouth_pts)

            eye_closed = self.current_ear < EAR_THRESHOLD
            self.perclos_window.append(int(eye_closed))
            if len(self.perclos_window) > PERCLOS_WINDOW:
                self.perclos_window.pop(0)
            current_perclos = perclos(sum(self.perclos_window), len(self.perclos_window))

            if eye_closed:
                self.ear_frame_count += 1
            else:
                if self.ear_frame_count >= EAR_CONSEC_FRAMES:
                    self.total_alerts += 1
                self.ear_frame_count = 0

            drowsy = (self.ear_frame_count >= EAR_CONSEC_FRAMES or
                      current_perclos >= PERCLOS_THRESHOLD)

            if self.current_mar > MAR_THRESHOLD:
                self.mar_frame_count += 1
            else:
                if self.mar_frame_count >= YAWN_CONSEC_FRAMES:
                    self.total_yawns += 1
                self.mar_frame_count = 0

            yawning = self.mar_frame_count >= YAWN_CONSEC_FRAMES

            if drowsy:
                self.status = "DROWSY"
                if not self.alarm_playing:
                    self.alarm_sound.play(-1)
                    self.alarm_playing = True
            elif yawning:
                self.status = "YAWNING"
                if not pygame.mixer.get_busy():
                    self.yawn_sound.play()
            else:
                self.status = "ALERT"
                if self.alarm_playing:
                    pygame.mixer.stop()
                    self.alarm_playing = False

            self._draw_eye_contour(frame, left_pts)
            self._draw_eye_contour(frame, right_pts)
        else:
            self.status = "NO FACE"
            if self.alarm_playing:
                pygame.mixer.stop()
                self.alarm_playing = False

        return self._draw_hud(frame, face_detected)

    def _draw_eye_contour(self, frame, pts):
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.drawContours(frame, [hull], -1, (0, 255, 180), 1)

    def _draw_hud(self, frame, face_detected):
        h, w = frame.shape[:2]
        status_colors = {
            "ALERT":   (0, 200, 0),
            "DROWSY":  (0, 0, 255),
            "YAWNING": (0, 165, 255),
            "NO FACE": (120, 120, 120),
        }
        color = status_colors.get(self.status, (255, 255, 255))

        if self.status == "DROWSY" and int(time.time() * 2) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(frame, f"STATUS: {self.status}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

        if face_detected:
            metrics = [
                (f"EAR : {self.current_ear:.3f}", self.current_ear < EAR_THRESHOLD),
                (f"MAR : {self.current_mar:.3f}", self.current_mar > MAR_THRESHOLD),
            ]
            for i, (txt, warn) in enumerate(metrics):
                mc = (0, 100, 255) if warn else (200, 200, 200)
                cv2.putText(frame, txt, (10, 75 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mc, 2, cv2.LINE_AA)

        elapsed = int(time.time() - self.session_start)
        mins, secs = divmod(elapsed, 60)
        for i, txt in enumerate([f"Session  : {mins:02d}:{secs:02d}",
                                   f"Alerts   : {self.total_alerts}",
                                   f"Yawns    : {self.total_yawns}"]):
            cv2.putText(frame, txt, (10, h - 80 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(frame, "Press Q to quit", (w - 190, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[INFO] Drowsiness detector started. Press 'Q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame  = cv2.flip(frame, 1)
            output = self.process_frame(frame)
            cv2.imshow("Driver Drowsiness Detection  |  Q to exit", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

        elapsed = int(time.time() - self.session_start)
        mins, secs = divmod(elapsed, 60)
        print(f"\n{'='*45}\n  SESSION SUMMARY\n{'='*45}")
        print(f"  Duration       : {mins:02d}:{secs:02d}")
        print(f"  Drowsy alerts  : {self.total_alerts}")
        print(f"  Yawns detected : {self.total_yawns}")
        print('='*45)