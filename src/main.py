"""
main.py
-------
Entry point for the Driver Drowsiness Detection System.

Usage:
    python src/main.py               # Default webcam
    python src/main.py --source 1    # Second camera
    python src/main.py --source video.mp4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detector import DrowsinessDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Driver Drowsiness Detection")
    parser.add_argument("--source", default="0",
                        help="Camera index (0,1,...) or video file path")
    return parser.parse_args()


def main():
    args   = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    print("="*55)
    print("   Driver Drowsiness Detection System")
    print("   Computer Vision Course — BYOP")
    print("="*55)
    print("   Version: 1.0  |  Author: Aditya")

    detector = DrowsinessDetector()
    try:
        detector.run(source=source)
    except IOError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")


if __name__ == "__main__":
    main()