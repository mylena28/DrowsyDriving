# DrowsyDriving — Real-Time Driver Safety Monitor

A lightweight, edge-deployable system that detects dangerous driving behaviors in real time.
It runs on a **Raspberry Pi 5** (CSI camera) or any desktop machine (USB webcam) using
YOLOv8-Pose and YOLOv8 object detection—fully via **ONNX**, with no PyTorch at runtime.

---

## Table of Contents

- [Overview](#overview)
- [Detected Behaviors](#detected-behaviors)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Docker (recommended)](#docker-recommended)
  - [Bare-metal on Raspberry Pi 5](#bare-metal-on-raspberry-pi-5)
  - [Desktop / development](#desktop--development)
- [Configuration](#configuration)
- [Deploying to a Raspberry Pi](#deploying-to-a-raspberry-pi)
- [Project Structure](#project-structure)

---

## Overview

DrowsyDriving processes a live camera feed frame-by-frame and flags behaviors associated
with distracted or fatigued driving. The pipeline is intentionally lean so it sustains
~30 FPS on a Raspberry Pi 5 without a GPU.

Key design decisions:

| Decision | Reason |
|---|---|
| ONNX-only runtime | Removes the ~2 GB PyTorch/CUDA stack from the deployment image |
| System-package NumPy & OpenCV | Avoids pip conflicts with `python3-picamera2` on Debian/RPi OS |
| Multi-stage Docker build | `.pt → .onnx` export happens in the builder; the runtime image stays lean |
| Picamera2 (not V4L2) | OpenCV cannot read CSI cameras on Pi 5 directly; Picamera2 is required |

---

## Detected Behaviors

| Behavior | Signal used | Trigger condition |
|---|---|---|
| **Head turn** | Nose + eye keypoints (YOLOv8-Pose) | Yaw asymmetry exceeds threshold for ≥ 5 consecutive frames |
| **Eye rubbing** | Wrist keypoints + eye positions | Hand near an eye with small oscillatory movement for ≥ 5 frames |
| **Phone in hand** | YOLOv8 `cell phone` class + geometry | Phone center is near both face and wrist, normalized by inter-eye distance |
| **On call / phone use** | Phone state classifier | Phone held to face (`EM LIGACAO`) or in hand (`USANDO CELULAR`) |
| **Hands busy** | Object proximity + wrist keypoints | Any object of interest (phone, bottle, cup) close to face and hand |

---

## Architecture

```
main.py
├── detectors/
│   ├── yolo_onnx.py        # Thin ONNX inference wrapper (replaces ultralytics at runtime)
│   ├── pose_detector.py    # YOLOv8n-pose → nose, eyes, wrist keypoints
│   ├── object_detector.py  # YOLOv8n → phone / bottle / cup detection + geometric filter
│   └── face_detector.py    # Auxiliary face detection
├── behaviors/
│   ├── head_turn.py        # Yaw-based head-turn classifier
│   ├── eye_rub.py          # Proximity + oscillation eye-rub classifier
│   └── phone_usage.py      # Phone-state machine (none / in hand / on call)
└── utils/
    ├── display.py          # Frame annotation (desktop mode only)
    ├── stabilizer.py       # Temporal state smoother (sliding-window vote)
    ├── risk.py             # Composite risk score
    ├── classifier.py       # State label from score
    ├── geometry.py         # Euclidean distance helper
    └── constants.py        # Shared thresholds
```

**Models** (not committed — export or download separately):

| File | Source |
|---|---|
| `yolov8n.onnx` | Exported from `yolov8n.pt` via `export_models.py` |
| `yolov8n-pose.onnx` | Exported from `yolov8n-pose.pt` via `export_models.py` |

---

## Requirements

### Raspberry Pi 5

- Raspberry Pi OS Bookworm (64-bit)
- CSI camera (e.g. Camera Module 3)
- Docker Engine ≥ 24 **or** Python 3.11 + `python3-picamera2` from the RPi repo
- `libcamera-apps` (`rpicam-hello` must be present for auto-detection)

### Desktop / CI

- Linux or macOS
- Docker Engine ≥ 24 **or** Python 3.10+ with `opencv-python` and `onnxruntime`
- USB webcam

---

## Quick Start

### Docker (recommended)

**1. Export the ONNX models** (requires the `.pt` weight files in the project root):

```bash
# The builder stage in the Dockerfile runs this automatically.
# To run it manually outside Docker:
pip install ultralytics onnx onnxslim onnxruntime
python3 export_models.py
```

**2. Build and run:**

```bash
make build   # docker compose build
make up      # docker compose up (attach mode)
# or
make up-d    # detached
make logs    # follow logs
```

**3. Stop:**

```bash
make down
```

Other useful targets:

```
make shell        # interactive bash inside the container
make clean        # remove __pycache__ and .pyc files
make clean-docker # docker system + volume prune
```

---

### Bare-metal on Raspberry Pi 5

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy \
                 libcamera-ipa rpicam-apps-lite

# 2. Install Python runtime dependency
pip3 install --break-system-packages onnxruntime

# 3. Place the exported ONNX models in the project root, then:
python3 main.py
```

> **Camera check:** before running, verify your camera is visible:
> ```bash
> libcamera-hello --list-cameras
> libcamera-hello -t 5   # 5-second preview
> ```

---

### Desktop / development

```bash
# Install dependencies
pip install onnxruntime opencv-python numpy

# Run (opens a display window if $DISPLAY is set)
python3 main.py
```

Press `Q` to quit the display window, or `Ctrl+C` to stop the terminal loop.

---

## Configuration

All tunable constants live in `config.py` and at the top of each behavior class:

| Parameter | Default | Description |
|---|---|---|
| `DIST_THRESHOLD` | `1.5` | Object-to-face normalized distance threshold |
| `FRAMES_THRESHOLD` | `5` | Consecutive frames required to trigger an alert |
| `MOUTH_THRESHOLD` | `0.35` | Mouth-openness ratio threshold |
| `HeadTurnDetector.yaw_threshold` | `0.4` | Yaw asymmetry to classify a head turn |
| `EyeRubDetector.proximity_factor` | `0.8` | Hand-eye distance factor (× inter-eye distance) |
| Camera resolution | `640×480` | Set in `initialize_picamera2()` / `initialize_opencv_fallback()` |
| Target FPS | `30` | Frame-time cap in the main loop (`0.033 s`) |

---

## Deploying to a Raspberry Pi

Use the included rsync helper to push the project over SSH:

```bash
# Edit the variables at the top of the script first:
#   PI_USER, PI_IP, PI_PATH
./sync_to_pi.sh
```

The script excludes `__pycache__`, `.pyc`, `.git`, `venv`, and `*.log`.

---

## Project Structure

```
DrowsyDriving/
├── main.py                 # Entry point + camera initialization
├── config.py               # Global thresholds
├── export_models.py        # .pt → .onnx export (requires ultralytics)
├── requirements.txt        # Pip deps for non-Docker installs
├── Dockerfile              # Multi-stage build (builder + runtime)
├── docker-compose.yml      # Service definition with /dev passthrough
├── Makefile                # Common docker compose shortcuts
├── sync_to_pi.sh           # rsync helper for Pi deployment
├── detectors/              # YOLO inference wrappers
├── behaviors/              # Per-behavior state machines
├── utils/                  # Shared utilities
└── logs/                   # Mounted log volume (Docker)
```
