# Project Report: Real-Time Driver Drowsiness Detection System

**Course:** Computer Vision
**Project Type:** BYOP (Build Your Own Project)
**Student Name:** Aditya
**GitHub Repository:** https://github.com/1-a-d-i-t-y-a/driver-drowsiness-detection

---

## 1. Introduction

Driver drowsiness is one of the leading causes of road accidents worldwide. According to the National Highway Traffic Safety Administration (NHTSA), fatigue-related crashes account for thousands of deaths and injuries every year. Traditional methods of detecting fatigue — such as steering behavior monitoring or lane deviation detection — require additional hardware and complex infrastructure.

This project proposes a software-only, real-time drowsiness detection system that uses a standard webcam and computer vision techniques to monitor a driver's facial features and raise an alert when signs of drowsiness or fatigue are detected. The system requires no specialized hardware and can be deployed on any laptop or embedded device with a camera.

---

## 2. Problem Statement

Design and implement a real-time system that:
- Continuously monitors a driver's face using a webcam
- Detects signs of drowsiness through eye closure analysis
- Detects yawning as an early indicator of fatigue
- Tracks long-term fatigue using the PERCLOS metric
- Triggers an immediate audio-visual alert to wake the driver

---

## 3. Objectives

- Implement Eye Aspect Ratio (EAR) to detect prolonged eye closure
- Implement Mouth Aspect Ratio (MAR) to detect yawning
- Track PERCLOS (Percentage of Eye Closure) over a rolling time window
- Provide real-time audio alarms using programmatically generated beeps
- Display a live HUD (Heads-Up Display) with all metrics overlaid on the video feed
- Log session statistics including total alerts and yawns

---

## 4. Literature Review

### 4.1 Eye Aspect Ratio (EAR)
Soukupová and Čech (2016) introduced the Eye Aspect Ratio as a simple and effective metric for blink and eye-closure detection. EAR is computed using six facial landmark points around the eye and drops significantly when the eye closes. This makes it ideal for detecting sustained eye closure associated with drowsiness.

### 4.2 PERCLOS
The PERCLOS (PERcentage of eye CLOSure) metric was defined by Wierwille et al. (1994) and is considered one of the most reliable indicators of driver fatigue. PERCLOS measures the proportion of time during which the eyes are more than 80% closed over a given time window. A PERCLOS value above 15% is widely accepted as an indicator of fatigue.

### 4.3 MediaPipe FaceMesh
Google's MediaPipe FaceMesh provides a lightweight, real-time solution for detecting 468 three-dimensional facial landmarks. It runs efficiently on consumer hardware without requiring a GPU, making it suitable for in-vehicle deployment.

---

## 5. System Architecture

The system is organized into three modules:

```
main.py
  └── DrowsinessDetector (detector.py)
        └── Utility Functions (utils.py)
```

### 5.1 utils.py — Geometric Metric Functions
Contains pure functions for computing EAR, MAR, PERCLOS, and extracting landmark coordinates from MediaPipe output.

### 5.2 detector.py — Core Detection Engine
Contains the `DrowsinessDetector` class which:
- Initializes MediaPipe FaceMesh and Pygame audio mixer
- Processes each video frame
- Computes EAR, MAR, and PERCLOS
- Manages alert state and audio playback
- Renders the HUD overlay on the frame

### 5.3 main.py — Entry Point
Parses command-line arguments (camera index or video file path), initializes the detector, and launches the main loop. Prints a session summary on exit.

---

## 6. Methodology

### 6.1 Facial Landmark Detection
MediaPipe FaceMesh processes each BGR video frame (converted to RGB) and returns 468 normalized landmark coordinates. Specific landmark indices are selected for the left eye, right eye, and mouth.

| Region | Landmark Indices Used |
|---|---|
| Left Eye | 362, 385, 387, 263, 373, 380 |
| Right Eye | 33, 160, 158, 133, 153, 144 |
| Mouth (8 points) | 61, 291, 0, 17, 269, 405, 321, 375 |

### 6.2 Eye Aspect Ratio (EAR)

EAR is computed for each eye using the formula:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

Where p1–p6 are the six eye landmark points. The average EAR of both eyes is used as the final value.

- Normal open eye: EAR ≈ 0.30 – 0.40
- Closed eye: EAR drops below 0.22 (threshold)
- Alert fires if EAR remains below threshold for 20+ consecutive frames (~0.67 seconds at 30 fps)

### 6.3 Mouth Aspect Ratio (MAR)

MAR measures the vertical openness of the mouth relative to its horizontal width:

```
MAR = (||p1-p7|| + ||p2-p6|| + ||p3-p5||) / (3 × ||p0-p4||)
```

- MAR > 0.65 sustained for 15+ consecutive frames → yawn detected

### 6.4 PERCLOS

A rolling window of 75 frames tracks whether the eye is classified as closed (EAR < threshold) at each frame:

```
PERCLOS = closed_frames / total_frames_in_window
```

- PERCLOS ≥ 0.15 (15%) triggers a fatigue-level drowsiness alert, even if no single continuous closure triggered the EAR counter

### 6.5 Audio Alarm
Two distinct tones are generated programmatically using NumPy sine waves and played via Pygame:
- **Drowsiness alarm:** 880 Hz continuous loop (high-pitched, urgent)
- **Yawn alert:** 440 Hz single play (lower, informational)

No external audio files are required.

---

## 7. Detection Thresholds

| Parameter | Value | Rationale |
|---|---|---|
| EAR Threshold | 0.22 | Below this value, eye is classified as closed |
| EAR Consecutive Frames | 20 | ~0.67 sec at 30 fps — avoids blink false positives |
| MAR Threshold | 0.65 | Mouth must be distinctly open to register as yawn |
| MAR Consecutive Frames | 15 | ~0.5 sec sustained opening required |
| PERCLOS Window | 75 frames | ~2.5 sec rolling window |
| PERCLOS Threshold | 0.15 | 15% closure rate — standard fatigue indicator |

---

## 8. HUD Interface

The system overlays a real-time Heads-Up Display on the video feed:

| Element | Description |
|---|---|
| Top status bar | Displays current state: ALERT / DROWSY / YAWNING / NO FACE |
| EAR value | Live eye aspect ratio, turns orange when below threshold |
| MAR value | Live mouth aspect ratio, turns orange when above threshold |
| Eye contours | Teal convex hull drawn over detected eye regions |
| Red pulsing overlay | Full-frame red flash when DROWSY state is active |
| Session timer | MM:SS elapsed since start |
| Alert count | Total drowsiness alerts in the session |
| Yawn count | Total yawns detected in the session |

---

## 9. Implementation Details

### Language and Libraries

| Library | Version | Purpose |
|---|---|---|
| Python | 3.11.x | Core language |
| OpenCV | 4.9.0.80 | Video capture, frame rendering |
| MediaPipe | 0.10.14 | Facial landmark detection |
| SciPy | 1.13.0 | Euclidean distance computation |
| NumPy | 2.0.0 | Array operations, audio synthesis |
| Pygame | 2.5.2 | Audio playback |
| imutils | 0.5.4 | Video stream utilities |

### Hardware Used
- **Laptop:** HP Victus
- **Processor:** Intel Core i7
- **GPU:** NVIDIA RTX 3050
- **OS:** Windows 11
- **Camera:** Built-in 720p webcam

---

## 10. Results

The system was tested under the following conditions:

| Test Scenario | Expected Behaviour | Result |
|---|---|---|
| Eyes closed for 2–3 seconds | Red flash + continuous alarm | Passed |
| Normal blinking | No false alert | Passed |
| Wide mouth open (yawn) | Short beep alert | Passed |
| Face out of frame | Status changes to NO FACE | Passed |
| Gradual fatigue (eyes half-open) | PERCLOS triggers alert | Passed |
| Press Q to quit | Session summary printed | Passed |

The system achieved reliable detection under good lighting conditions with minimal false positives during normal blinking. Detection latency was under 100ms on the test hardware.

---

## 11. Limitations

- Performance degrades in low-light or backlit environments
- Occlusions (glasses, face mask) may affect landmark accuracy
- Currently supports single-face detection only
- No data logging or cloud integration for fleet monitoring
- Head pose variations (looking sideways) can affect EAR accuracy

---

## 12. Future Enhancements

- Add head pose estimation to detect microsleep with head drop
- Integrate IR camera support for night-time driving
- Implement data logging to CSV for post-trip analysis
- Add a mobile companion app for remote monitoring
- Deploy on Raspberry Pi for low-cost embedded vehicle integration
- Train a custom CNN model for higher accuracy in adverse lighting

---

## 13. Conclusion

This project successfully demonstrates a real-time, software-only driver drowsiness detection system using computer vision. By combining EAR-based eye closure detection, MAR-based yawn detection, and PERCLOS-based long-term fatigue tracking, the system provides a multi-layered approach to drowsiness detection. The system is lightweight, requires no special hardware, and can be deployed immediately on any laptop with a webcam.

The project highlights the practical applicability of computer vision in road safety, and lays a strong foundation for more sophisticated driver monitoring systems in the future.

---

## 14. References

1. Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.
2. Wierwille, W.W., et al. (1994). *Research on Vehicle-Based Driver Status/Performance Monitoring.* NHTSA Technical Report.
3. Google MediaPipe Team. *MediaPipe FaceMesh.* https://mediapipe.dev
4. OpenCV Documentation. https://docs.opencv.org
5. Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.

---

*Report prepared for Computer Vision Course — BYOP Submission*
*Author: Aditya | GitHub: https://github.com/1-a-d-i-t-y-a/driver-drowsiness-detection*
