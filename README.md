# Real-Time Driver Drowsiness and Yawn Detection

A computer vision application that detects driver fatigue and yawning in real time using webcam video and facial landmark geometry.

---

## Technical Overview

The system processes video frames to compute facial geometric indicators:
* **Eye Aspect Ratio (EAR):** Tracks eye opening and flags prolonged eye closure.
* **Mouth Aspect Ratio (MAR):** Measures vertical mouth opening to detect yawning.
* **PERCLOS (Percentage of Eye Closure):** Measures the proportion of closed-eye frames across a rolling 75-frame observation window.

> **Operational Disclaimer:** This application is an educational engineering project and computer vision demonstration. It is not certified for medical or safety-critical driver assistance.

---

## Detection Logic

1. **Facial Landmark Mesh:** MediaPipe FaceMesh tracks facial coordinates across consecutive frames.
2. **EAR Calculation:** Euclidean distances between vertical eyelid landmarks are compared against the horizontal eye width. Sustained low values indicate closure.
3. **MAR Calculation:** Vertical lip landmarks are compared against mouth width. Sustained high values indicate yawning.
4. **Heads-Up Display (HUD):** An OpenCV overlay renders current status, live metrics, elapsed session time, and alert counts.

---

## Getting Started

### Requirements
* Python 3.11 or 3.12
* Standard webcam or video input

### Installation
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

### Running the Detector
* Default camera:
  ```bash
  python src/main.py
  ```
* Specific camera index:
  ```bash
  python src/main.py --source 1
  ```
* Video file playback:
  ```bash
  python src/main.py --source sample.mp4
  ```

Press `q` to terminate the session and view the summary in your terminal.

---

## Automated Verification

Execute the test suite with pytest:
```bash
pytest tests/ -v
```

Unit tests validate:
* EAR calculations on open, closed, and edge-case geometries.
* MAR calculations on neutral and open mouth coordinates.
* Division-by-zero resilience when points are obscured or identical.
* Rolling window PERCLOS arithmetic.
* Headless execution without audio hardware.

---

## Repository Structure

```text
driver-drowsiness-detection/
├── src/
│   ├── detector.py      # Core detection loop and HUD rendering
│   ├── main.py          # Command-line entrypoint
│   └── utils.py         # Geometric calculation utilities
├── tests/
│   └── test_metrics.py  # Unit test suite
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## License

This project is available under the MIT License.
