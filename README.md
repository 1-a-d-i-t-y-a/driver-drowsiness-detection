# Driver Drowsiness Detection System

A real-time driver drowsiness and fatigue monitoring system using Computer Vision and facial geometry analysis. Developed for the Computer Vision Course — BYOP (Build Your Own Project).

**Author:** Aditya ([@1-a-d-i-t-y-a](https://github.com/1-a-d-i-t-y-a))  
**Academic Context:** VIT — Computer Vision Course Project  
**Repository:** [https://github.com/1-a-d-i-t-y-a/driver-drowsiness-detection](https://github.com/1-a-d-i-t-y-a/driver-drowsiness-detection)  

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-00C4B4)](https://developers.google.com/mediapipe)
[![Tests](https://img.shields.io/badge/Tests-8%20Passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Important Safety Disclaimer:** This software is an academic engineering prototype and computer vision demonstration. It is not medically certified and must not replace alert, responsible driving practices.

---

## Overview

This system monitors a driver's face through a webcam to detect signs of drowsiness or fatigue in real time. It utilizes facial geometry metrics — Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and PERCLOS (Percentage of Eye Closure) — to trigger instant audio-visual alerts before hazardous micro-sleep episodes occur.

---

## Features

* **Real-time eye monitoring** using Eye Aspect Ratio (EAR)
* **Yawn detection** using Mouth Aspect Ratio (MAR)
* **PERCLOS metric tracking** over a rolling 75-frame observation window
* **Multi-tone audio alarms** with distinct frequencies for drowsiness (880 Hz) vs. yawning (440 Hz)
* **Visual Heads-Up Display (HUD)** with live EAR/MAR telemetry, session stopwatch, alert count, and yawn count
* **Dynamic red flash overlay** during active drowsiness events
* **Comprehensive terminal session summary** on application exit
* **Headless testing support** with zero-division geometric safeguards

---

## Tech Stack

| Library | Version Target | Purpose |
| :--- | :--- | :--- |
| **OpenCV** | `opencv-python>=4.9.0` | Webcam frame capture, overlay rendering, HUD graphics |
| **MediaPipe** | `mediapipe>=0.10.9` | 468-point 3D facial landmark mesh detection |
| **SciPy** | `scipy>=1.12.0` | Euclidean distance calculations for EAR and MAR |
| **NumPy** | `numpy>=1.26.0` | Array mathematics and audio waveform synthesis |
| **Pygame** | `pygame>=2.5.2` | Real-time audio alert playback with headless fallback |
| **imutils** | `imutils>=0.5.4` | Video processing convenience functions |

---

## How It Works

### 1. Facial Landmark Detection
MediaPipe FaceMesh extracts 468 normalized facial landmarks per frame:
* **Left Eye Indices:** `[362, 385, 387, 263, 373, 380]`
* **Right Eye Indices:** `[33, 160, 158, 133, 153, 144]`
* **Mouth Indices:** `[61, 291, 0, 17, 269, 405, 321, 375]`

### 2. Eye Aspect Ratio (EAR)
EAR measures the degree of eyelid opening using 6 key coordinates per eye:

$$\text{EAR} = \frac{\lVert p_2 - p_6 \rVert + \lVert p_3 - p_5 \rVert}{2 \cdot \lVert p_1 - p_4 \rVert}$$

* Normal open eye: $\text{EAR} \approx 0.30 - 0.40$
* Closed eye threshold: $\text{EAR} < 0.22$
* Trigger condition: Sustained below threshold for $\ge 20$ consecutive frames ($\approx 0.7$ seconds at 30 FPS).

### 3. Mouth Aspect Ratio (MAR)
MAR detects yawning by calculating vertical lip separation against horizontal mouth width:

$$\text{MAR} = \frac{\lVert p_2 - p_8 \rVert + \lVert p_3 - p_7 \rVert + \lVert p_4 - p_6 \rVert}{3 \cdot \lVert p_1 - p_5 \rVert}$$

* Neutral mouth: $\text{MAR} \approx 0.25 - 0.35$
* Yawn threshold: $\text{MAR} > 0.65$ sustained for $\ge 15$ consecutive frames.

### 4. PERCLOS (Percentage of Eye Closure)
Calculates proportion of closed frames over a rolling 75-frame window:

$$\text{PERCLOS} = \frac{\text{closed\_frames}}{\text{total\_frames}}$$

* Fatigue warning threshold: $\text{PERCLOS} \ge 0.15$ (15% closure over observation window).

---

## Detection Thresholds

| Parameter | Value | Operational Description |
| :--- | :---: | :--- |
| **EAR Threshold** | `0.22` | Eyes are considered closed below this value |
| **EAR Consecutive Frames** | `20` | Minimum frames before drowsiness alarm triggers |
| **MAR Threshold** | `0.65` | Mouth is considered open (yawning) above this value |
| **MAR Consecutive Frames** | `15` | Minimum frames before yawn alert triggers |
| **PERCLOS Window** | `75 frames` | Rolling observation window for fatigue tracking |
| **PERCLOS Threshold** | `0.15 (15%)` | Prolonged fatigue warning trigger level |

---

## Heads-Up Display (HUD) Interface

| Element | Location | Description |
| :--- | :--- | :--- |
| **STATUS Banner** | Top bar | Real-time state indicator (`ALERT`, `DROWSY`, `YAWNING`, `NO FACE`) |
| **EAR Telemetry** | Left side | Current Eye Aspect Ratio value (highlights orange when low) |
| **MAR Telemetry** | Left side | Current Mouth Aspect Ratio value (highlights orange when high) |
| **Session Timer** | Bottom left | Elapsed monitoring time formatted as `MM:SS` |
| **Alert Counter** | Bottom left | Total cumulative drowsiness alerts triggered during the session |
| **Yawn Counter** | Bottom left | Total cumulative yawn events detected during the session |
| **Eye Contours** | On face | Real-time teal polygon outlines rendered over ocular landmarks |
| **Red Flash Overlay** | Full frame | Pulsing semi-transparent red alert when `DROWSY` state is active |

---

## Installation & Setup

### Prerequisites
* Python 3.11 or 3.12
* Webcam or video capture device

### Setup Commands
```bash
git clone https://github.com/1-a-d-i-t-y-a/driver-drowsiness-detection.git
cd driver-drowsiness-detection

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

**Start with default webcam (device 0):**
```bash
python src/main.py
```

**Run with secondary external camera:**
```bash
python src/main.py --source 1
```

**Run on a recorded test video file:**
```bash
python src/main.py --source path/to/video.mp4
```

*Press `Q` at any time to exit.* A formatted session summary prints directly to your terminal.

---

## Automated Verification & Tests

Run the test suite with pytest:
```bash
pytest tests/ -v
```

8 automated unit tests verify:
* EAR calculations on open vs. closed eye geometry.
* MAR calculations on neutral vs. yawn mouth geometry.
* Division-by-zero protection when landmarks are obscured.
* PERCLOS rolling window mathematics and bounds.
* Headless detector instantiation and frame processing.

---

## Tested Hardware & System Environment

* **Hardware:** HP Victus Laptop (Intel Core i7, NVIDIA RTX 3050, 16GB RAM)
* **Operating System:** Windows 11 64-bit
* **Camera:** Integrated 720p HD webcam
* **Python Runtime:** 3.11.x / 3.12.x

---

## Project Structure

```text
driver-drowsiness-detection/
├── report/
│   └── project_report.md  # Comprehensive academic project report
├── src/
│   ├── detector.py        # Core detection loop and HUD rendering
│   ├── main.py            # Command-line entrypoint and argument parser
│   └── utils.py           # Geometric calculation functions (EAR, MAR, PERCLOS)
├── tests/
│   └── test_metrics.py    # 8 automated unit test cases
├── requirements.txt       # Python dependencies
├── .gitignore             # Version control exclusion rules
└── README.md              # Standardized project documentation
```

---

## License

This project was developed for academic and educational purposes as part of the Computer Vision curriculum at VIT. See the [Project Report](report/project_report.md) for full project details.
