
# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using Computer Vision and facial landmark analysis. Built as part of the Computer Vision course — BYOP (Build Your Own Project).

---

## Overview

This system monitors a driver's face through a webcam and detects signs of drowsiness or fatigue in real time. It uses facial geometry metrics — Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and PERCLOS — to trigger audio-visual alerts before a dangerous situation occurs.

---

## Features

- **Real-time eye monitoring** using Eye Aspect Ratio (EAR)
- **Yawn detection** using Mouth Aspect Ratio (MAR)
- **PERCLOS metric** — tracks percentage of eye closure over a rolling 75-frame window
- **Audio alarm** — distinct beeps for drowsiness vs yawning
- **Visual HUD** — live EAR/MAR values, session timer, alert count, yawn count
- **Red flash overlay** when drowsiness is detected
- **Session summary** printed in terminal on exit

---

## Tech Stack

| Library | Purpose |
|---|---|
| OpenCV | Webcam capture, frame rendering, HUD drawing |
| MediaPipe | 468-point facial landmark detection |
| SciPy | Euclidean distance for EAR/MAR calculation |
| NumPy | Array math and audio wave generation |
| Pygame | Real-time audio alarm playback |
| imutils | Video stream utility helpers |

---

## Project Structure

```
driver-drowsiness-detection/
├── src/
│   ├── main.py          # Entry point — parses args, launches detector
│   ├── detector.py      # DrowsinessDetector class — core logic + HUD
│   └── utils.py         # EAR, MAR, PERCLOS helper functions
├── report/
│   └── project_report.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How It Works

### 1. Facial Landmark Detection
MediaPipe FaceMesh detects 468 facial landmarks per frame. Specific landmark indices are selected for the left eye, right eye, and mouth.

### 2. Eye Aspect Ratio (EAR)
EAR measures the openness of each eye using 6 key points:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

- Normal open eye → EAR ≈ 0.30+
- Closed eye → EAR drops below **0.22** (threshold)
- If EAR stays below threshold for **20+ consecutive frames** → drowsiness alert

### 3. Mouth Aspect Ratio (MAR)
MAR measures mouth openness using 8 points around the lips:

```
MAR = (A + B + C) / (3 × D)
```

- If MAR exceeds **0.65** for **15+ consecutive frames** → yawn detected

### 4. PERCLOS
Percentage of Eye Closure over a rolling 75-frame window:

```
PERCLOS = closed_frames / total_frames
```

- PERCLOS ≥ 0.15 (15%) → fatigue warning triggered

---

## Installation

### Prerequisites
- Python 3.11.x
- Webcam

### Steps

**1. Clone the repository:**
```bash
git clone https://github.com/1-a-d-i-t-y-a/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

**2. Create and activate virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

**Default webcam (index 0):**
```bash
python src/main.py
```

**Second camera:**
```bash
python src/main.py --source 1
```

**Video file:**
```bash
python src/main.py --source path/to/video.mp4
```

**Press `Q` to quit.** A session summary will print in the terminal.

---

## Detection Thresholds

| Parameter | Value | Description |
|---|---|---|
| EAR Threshold | 0.22 | Eye considered closed below this |
| EAR Consecutive Frames | 20 | Frames before drowsiness alert fires |
| MAR Threshold | 0.65 | Mouth considered open (yawn) above this |
| MAR Consecutive Frames | 15 | Frames before yawn alert fires |
| PERCLOS Window | 75 frames | Rolling window for fatigue metric |
| PERCLOS Threshold | 0.15 (15%) | Fatigue alert trigger level |

---

## HUD Interface

| Element | Location | Description |
|---|---|---|
| STATUS label | Top bar | ALERT / DROWSY / YAWNING / NO FACE |
| EAR value | Left side | Live eye aspect ratio (turns orange when low) |
| MAR value | Left side | Live mouth aspect ratio (turns orange when high) |
| Session timer | Bottom left | MM:SS elapsed since start |
| Alert count | Bottom left | Total drowsiness alerts this session |
| Yawn count | Bottom left | Total yawns detected this session |
| Eye contours | On face | Teal outline drawn over detected eyes |
| Red flash | Full frame | Pulsing overlay when DROWSY state is active |

---

## Requirements

```
opencv-python==4.9.0.80
mediapipe==0.10.14
scipy==1.13.0
numpy==2.0.0
pygame==2.5.2
imutils==0.5.4
```

---

## System Info

- **Tested on:** HP Victus, Intel Core i7, NVIDIA RTX 3050, Windows 11
- **Python version:** 3.11.x
- **Camera:** Built-in 720p webcam

---

## Author

**Aditya**
VIT — Computer Vision Course | BYOP Project
GitHub: [@1-a-d-i-t-y-a](https://github.com/1-a-d-i-t-y-a)

---

## License

This project is for academic purposes only.
