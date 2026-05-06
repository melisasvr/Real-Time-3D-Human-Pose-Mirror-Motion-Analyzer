# Real-Time 3D Human Pose Mirror & Motion Analyzer

<div align="center">

![FORM Banner](https://img.shields.io/badge/FORM-3D%20Pose%20Mirror-00d4ff?style=for-the-badge&labelColor=04080c)

[![License: MIT](https://img.shields.io/badge/License-MIT-00d4ff.svg?style=flat-square&labelColor=04080c)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20(Pyodide)-39ff8a?style=flat-square&logo=python&logoColor=white&labelColor=04080c)](https://pyodide.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%200.5-ff6b35?style=flat-square&labelColor=04080c)](https://mediapipe.dev)
[![Three.js](https://img.shields.io/badge/Three.js-r128-00d4ff?style=flat-square&labelColor=04080c)](https://threejs.org)
[![NumPy](https://img.shields.io/badge/NumPy-SciPy-39ff8a?style=flat-square&logo=numpy&logoColor=white&labelColor=04080c)](https://numpy.org)

**A fully browser-based real-time 3D motion capture and posture analysis tool.**  
No backend. No installation. Just Python running in your browser via WebAssembly.

</div>

---

## Overview

Real Time 3D captures your body movements through a webcam, estimates 33 full-body 3D keypoints using MediaPipe Pose, and renders a live-mirrored 3D skeleton in Three.js, all while a Python engine (running in-browser via Pyodide) computes joint angles, applies signal filtering, analyses your posture, and tracks motion dynamics in real time.

### Key Highlights
- **Zero-install Python in the browser** Pyodide runs CPython 3.11 as WebAssembly, executing `pose_engine.py` with full NumPy and SciPy support directly in the browser tab
- **No server, no cloud** your webcam data never leaves your machine
- **SciPy Butterworth filtering** per-landmark 2nd-order low-pass filter maintains state across frames for stable, jitter-free 3D rendering
- **True 3D joint angles** computed from world-space 3D vectors using NumPy dot products and arccos, not 2D projections
- **~13–16 FPS** on a typical laptop with Python processing latency of ~20ms per frame

---

## Screenshots

| Posture Feedback | Arm Detection | Skeleton View |
|:---:|:---:|:---:|
| Score 100 · Great Form | Score 91 · Shoulder Tilt | Score 78 · Needs Work |

---

## Features

### Live Pose Detection (MediaPipe)
- Detects **33 full-body landmarks** in real time via MediaPipe BlazePose
- Runs entirely on the GPU/CPU in the browser using WebAssembly
- 2D overlay drawn on the camera feed with gradient-coloured skeleton connections

### In-Browser Python Engine (`pose_engine.py`)
- Loaded and executed by **Pyodide** no Python installation required
- **`ButterworthSmoother`** 2nd-order SciPy `sosfilt` per landmark per axis, with persistent filter state for zero-transient smoothing
- **`MotionTracker`** sliding-window velocity and acceleration computed with NumPy norms
- **`compute_angles()`** 12 joint angles (elbow, shoulder, knee, hip, spine, ankle, neck) using true 3D vector geometry
- **`analyse_posture()`** rule-based geometry checks with severity grading and scored feedback
- **`SessionRecorder`** in-memory frame recorder with timestamped pose + angle data for replay

### 3D Avatar (Three.js)
- Real-time 3D skeleton rendered with cylinder bone meshes and sphere joints
- Orbit (drag), zoom (scroll), and pan (right-drag) controls
- Bone colour transitions cyan → orange based on live motion intensity
- Hip-anchored coordinate system keeps the skeleton centred regardless of position

### Posture Analysis
| Check | Method |
|---|---|
| Shoulder drop | Y-axis ratio normalised by shoulder width |
| Spine lean | Shoulder–hip midpoint vector angle |
| Forward head posture | Nose Z-offset relative to shoulder midpoint |
| Hip tilt | Y-axis ratio normalised by hip width |
| Arm asymmetry | Elbow angle delta between left and right |

### Joint Angles Panel
12 joints tracked with ideal range indicators:

| Joint | Ideal Range |
|---|---|
| L / R Elbow | 65° – 180° |
| L / R Shoulder | 10° – 95° |
| L / R Knee | 155° – 180° |
| L / R Hip | 155° – 180° |
| Spine | 140° – 180° |
| L / R Ankle | 55° – 125° |
| Neck / Head | 50° – 90° |

### Motion Replay
- Record any motion sequence in-session
- Replay frame-by-frame at 30 FPS on the 3D skeleton
- Frame count, duration, and average FPS displayed

---

## Project Structure

```
FORM/
├── index.html          # Frontend — MediaPipe, Three.js, Pyodide bootstrap, UI
├── pose_engine.py      # Python engine — loaded in-browser by Pyodide
├── README.md           # This file
└── LICENSE             # MIT License
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                        Browser Tab                       │
│                                                          │
│  ┌──────────────┐    raw       ┌─────────────────────┐  │
│  │  MediaPipe   │  landmarks   │  Pyodide (WASM)     │  │
│  │  BlazePose   │ ──────────▶  │  pose_engine.py     │  │
│  │  (WASM/GPU)  │              │  ├─ ButterworthSmth  │  │
│  └──────┬───────┘              │  ├─ compute_angles   │  │
│         │ 2D overlay           │  ├─ analyse_posture  │  │
│         ▼                      │  └─ MotionTracker    │  │
│  ┌──────────────┐  smoothed    └──────────┬──────────┘  │
│  │  <canvas>    │  landmarks              │ JSON         │
│  │  2D overlay  │  ◀───────────────────── │              │
│  └──────────────┘                         ▼              │
│                              ┌─────────────────────┐     │
│                              │  Three.js r128       │     │
│                              │  3D Skeleton Render  │     │
│                              │  Orbit Controls      │     │
│                              └─────────────────────┘     │
└─────────────────────────────────────────────────────────┘
          ▲
     Webcam feed (stays local — never transmitted)
```

---

## Getting Started

### Prerequisites

- A modern browser (Chrome 90+, Firefox 88+, Edge 90+)
- A webcam
- Python 3 (only for running a local file server)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/form-pose-mirror.git
cd form-pose-mirror
```

**2. Start a local HTTP server**

- The browser must load `pose_engine.py` via HTTP (not `file://`). Use any of these:

```bash
# Python (recommended)
python -m http.server 8080

# Node.js
npx serve .

# VS Code
# Install the "Live Server" extension and click "Go Live"
```

**3. Open in browser**
```
http://localhost:8080
```

### First Load
- On first load, Pyodide downloads the Python runtime + NumPy + SciPy (~30 MB). This takes **15–30 seconds** depending on your connection. After that, everything is cached by the browser.

A loading screen shows progress:
```
Pyodide: loading runtime… → installing packages… → loading engine… ✓ ready
MediaPipe: loading… ✓ ready
```

### Usage

1. Wait for both **PYTHON ✓** and **POSE ✓** chips in the header to turn green
2. Click **"Enable Camera & Start"**
3. Grant camera permission
4. **Step back** so your upper body (head to hips) is fully visible the 3D skeleton requires hip landmarks to anchor itself
5. Use the controls at the bottom of the 3D view:

| Button | Action |
|---|---|
| `SKELETON` | Toggle bone meshes |
| `JOINTS` | Toggle joint spheres |
| `GRID` | Toggle floor grid |
| `SMOOTH` | Toggle Butterworth filter (resets state) |
| `● REC` | Start / stop motion recording |
| `▶ REPLAY` | Replay recorded motion |
| `RESET` | Reset 3D camera to default view |

---

## Browser Compatibility

| Browser | Status |
|---|---|
| Chrome 90+ | ✅ Fully supported |
| Edge 90+ | ✅ Fully supported |
| Firefox 88+ | ✅ Supported |
| Safari 15+ | ⚠️ May require camera permission reset |
| Mobile | ⚠️ Limited — Pyodide is memory-intensive |

---

## Performance
- Typical performance on a mid-range laptop (Intel Core i5, integrated GPU):

| Metric | Value |
|---|---|
| FPS | 13–16 FPS |
| Python call latency | 18–26 ms/frame |
| Pyodide boot time (cached) | ~2s |
| Pyodide boot time (first load) | ~15–30s |
| Memory usage | ~400 MB |

---

## How It Works

### Coordinate System
- MediaPipe outputs two sets of landmarks per frame:

- **Image landmarks** (`poseLandmarks`) normalised `x/y` in `[0,1]` image space, used for 2D overlay
- **World landmarks** (`poseWorldLandmarks`) metric 3D coordinates centred at the hip midpoint, used for Python angle math and 3D rendering

The 3D skeleton anchors itself at the hip midpoint each frame, so the avatar stays centred regardless of where you stand.

### Butterworth Filter

```python
# 2nd-order low-pass at 7 Hz, sample rate 30 Hz
sos = butter(2, 7.0 / 15.0, btype="low", output="sos")
```

- Filter state (`zi`) is maintained across frames using `sosfilt_zi`, which initialises the state to avoid the step-response transient that would otherwise cause a visible jump at startup. This gives smooth, stable 3D motion without introducing lag.

### Angle Calculation
- All joint angles use true 3D vector geometry on Butterworth-smoothed world coordinates:

```python
def angle_3d(A, B, C):
    BA = A - B
    BC = C - B
    cos_t = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
```

---

## Contributing
- Contributions are welcome and appreciated! Here's how to get involved:

### How to Contribute
1. **Fork** the repository
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** — keep `pose_engine.py` and `index.html` as the two-file structure
4. **Test** by running `python -m http.server 8080` and verifying in Chrome
5. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add rep counter for squat detection"
   ```
6. **Push** and open a **Pull Request** against `main`

### Ideas for Contributions
- **Exercise rep counters** detect squat, push-up, or curl repetitions from angle thresholds
- **Pose presets** compare against reference poses (yoga, powerlifting, etc.)
- **Export** download session data as CSV or JSON
- **Audio cues** speak posture corrections via Web Speech API
- **Mobile optimisation** lighter MediaPipe model for phones
- **Historical tracking** chart posture score over time using localStorage
- **Additional joints** wrist rotation, finger tracking
- **WebGL shader effects** per-joint colour coding by angle deviation

### Reporting Issues
- Please open a GitHub Issue with:
- Browser and OS
- Console error messages (F12 → Console)
- Steps to reproduce

### Code Style
- `pose_engine.py` follows PEP 8, type-hints all functions, docstrings all classes
- `index.html` keep JS compact; document any non-obvious Three.js or Pyodide calls with inline comments

---

## License

```
MIT License

Copyright (c) 2026 Real-Time-3D Human Pose Mirror Motion Analyzer Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgements

- [MediaPipe](https://mediapipe.dev) Google's real-time ML framework for BlazePose landmark detection
- [Pyodide](https://pyodide.org) CPython compiled to WebAssembly, enabling Python in the browser
- [Three.js](https://threejs.org) 3D rendering library for the skeleton viewer
- [NumPy](https://numpy.org) Array math for angle computation and motion tracking
- [SciPy](https://scipy.org) Signal processing for the Butterworth low-pass smoother

---

<div align="center">

Made with Python, WebAssembly, and a webcam.

**[Report a Bug](https://github.com/your-username/form-pose-mirror/issues)** · **[Request a Feature](https://github.com/your-username/form-pose-mirror/issues)** · **[Open a PR](https://github.com/your-username/form-pose-mirror/pulls)**

</div>
