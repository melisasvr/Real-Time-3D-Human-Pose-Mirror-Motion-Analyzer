# pose_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# FORM — In-Browser Python Pose Engine
# Loaded by Pyodide and executed entirely in the browser via WebAssembly.
#
# Exposed to JavaScript via pyodide.globals:
#   process_frame(lm_json)   → JSON string with all computed data
#   reset_smoother()         → resets Butterworth filter state
#   get_replay_frame(idx)    → JSON of a recorded frame
#   record_control(cmd)      → "start" | "stop" | "clear"
#   get_session_info()       → JSON with recording metadata
# ─────────────────────────────────────────────────────────────────────────────

import json
import math
import time
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

# ── Landmark index map ────────────────────────────────────────────────────────
LM = dict(
    NOSE=0,
    LEFT_EYE_INNER=1, LEFT_EYE=2, LEFT_EYE_OUTER=3,
    RIGHT_EYE_INNER=4, RIGHT_EYE=5, RIGHT_EYE_OUTER=6,
    LEFT_EAR=7, RIGHT_EAR=8,
    LEFT_MOUTH=9, RIGHT_MOUTH=10,
    LEFT_SHOULDER=11, RIGHT_SHOULDER=12,
    LEFT_ELBOW=13, RIGHT_ELBOW=14,
    LEFT_WRIST=15, RIGHT_WRIST=16,
    LEFT_PINKY=17, RIGHT_PINKY=18,
    LEFT_INDEX=19, RIGHT_INDEX=20,
    LEFT_THUMB=21, RIGHT_THUMB=22,
    LEFT_HIP=23, RIGHT_HIP=24,
    LEFT_KNEE=25, RIGHT_KNEE=26,
    LEFT_ANKLE=27, RIGHT_ANKLE=28,
    LEFT_HEEL=29, RIGHT_HEEL=30,
    LEFT_FOOT=31, RIGHT_FOOT=32,
)

# ── Angle definitions (vertex B, rays B→A and B→C) ───────────────────────────
ANGLE_DEFS = [
    {"name": "L_ELBOW",    "a": 11, "b": 13, "c": 15, "ideal": (65, 180), "label": "Left Elbow"},
    {"name": "R_ELBOW",    "a": 12, "b": 14, "c": 16, "ideal": (65, 180), "label": "Right Elbow"},
    {"name": "L_SHOULDER", "a": 13, "b": 11, "c": 23, "ideal": (10, 95),  "label": "Left Shoulder"},
    {"name": "R_SHOULDER", "a": 14, "b": 12, "c": 24, "ideal": (10, 95),  "label": "Right Shoulder"},
    {"name": "L_KNEE",     "a": 23, "b": 25, "c": 27, "ideal": (155, 180),"label": "Left Knee"},
    {"name": "R_KNEE",     "a": 24, "b": 26, "c": 28, "ideal": (155, 180),"label": "Right Knee"},
    {"name": "L_HIP",      "a": 11, "b": 23, "c": 25, "ideal": (155, 180),"label": "Left Hip"},
    {"name": "R_HIP",      "a": 12, "b": 24, "c": 26, "ideal": (155, 180),"label": "Right Hip"},
    {"name": "SPINE",      "a": 24, "b": 12, "c":  0, "ideal": (140, 180),"label": "Spine"},
    {"name": "L_ANKLE",    "a": 25, "b": 27, "c": 29, "ideal": (55, 125), "label": "Left Ankle"},
    {"name": "R_ANKLE",    "a": 26, "b": 28, "c": 30, "ideal": (55, 125), "label": "Right Ankle"},
    {"name": "NECK",       "a":  0, "b": 11, "c": 12, "ideal": (50, 90),  "label": "Neck/Head"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def vec3(lm: dict) -> np.ndarray:
    return np.array([lm["x"], lm["y"], lm.get("z", 0.0)], dtype=np.float64)


def angle_3d(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """Angle at vertex B (degrees) using true 3-D vectors."""
    BA = A - B
    BC = C - B
    n_ba = np.linalg.norm(BA)
    n_bc = np.linalg.norm(BC)
    if n_ba < 1e-9 or n_bc < 1e-9:
        return 0.0
    cos_t = np.dot(BA, BC) / (n_ba * n_bc)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def midpt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Butterworth low-pass smoother
# ─────────────────────────────────────────────────────────────────────────────

class ButterworthSmoother:
    """
    2nd-order zero-phase Butterworth low-pass filter per landmark/coordinate.
    Maintains SOS filter state between frames for causal, stable filtering.
    """

    def __init__(self, n: int = 33, cutoff_hz: float = 7.0, sample_hz: float = 30.0):
        nyq = sample_hz / 2.0
        self.sos = butter(2, cutoff_hz / nyq, btype="low", output="sos")
        self.n   = n
        self._zi: np.ndarray | None = None  # shape: (n, 3, n_sec, 2)

    def _init(self, frame: np.ndarray) -> np.ndarray:
        zi0 = sosfilt_zi(self.sos)           # (n_sec, 2)
        zi  = np.zeros((self.n, 3, *zi0.shape))
        for i in range(self.n):
            for c in range(3):
                zi[i, c] = zi0 * frame[i, c]
        return zi

    def update(self, landmarks: list[dict]) -> np.ndarray:
        """Return smoothed (n, 3) array."""
        frame = np.array([[lm["x"], lm["y"], lm.get("z", 0.0)] for lm in landmarks],
                         dtype=np.float64)
        if self._zi is None:
            self._zi = self._init(frame)

        out = np.empty_like(frame)
        for i in range(self.n):
            for c in range(3):
                y, self._zi[i, c] = sosfilt(self.sos, frame[i:i+1, c], zi=self._zi[i, c])
                out[i, c] = y[0]
        return out

    def reset(self):
        self._zi = None


# ─────────────────────────────────────────────────────────────────────────────
# Motion tracker
# ─────────────────────────────────────────────────────────────────────────────

class MotionTracker:
    def __init__(self, window: int = 8):
        self._hist: deque[np.ndarray] = deque(maxlen=window)

    def update(self, frame: np.ndarray) -> dict:
        self._hist.append(frame.copy())
        if len(self._hist) < 2:
            return {"velocity": 0.0, "acceleration": 0.0, "active": []}
        vel = np.linalg.norm(self._hist[-1] - self._hist[-2], axis=1)   # (33,)
        avg = float(np.mean(vel))
        acc = 0.0
        if len(self._hist) >= 3:
            vp = np.linalg.norm(self._hist[-2] - self._hist[-3], axis=1)
            acc = float(np.mean(np.abs(vel - vp)))
        thr    = max(avg * 2.0, 0.001)
        active = [int(i) for i, v in enumerate(vel) if v > thr]
        return {"velocity": round(avg * 1000, 2), "acceleration": round(acc * 1000, 2), "active": active}


# ─────────────────────────────────────────────────────────────────────────────
# Session recorder
# ─────────────────────────────────────────────────────────────────────────────

class SessionRecorder:
    def __init__(self):
        self.frames: list[dict] = []
        self.recording = False
        self._t0 = 0.0

    def start(self):
        self.frames   = []
        self.recording = True
        self._t0      = time.time()

    def stop(self) -> dict:
        self.recording = False
        dur = time.time() - self._t0
        return {
            "frame_count": len(self.frames),
            "duration_s":  round(dur, 2),
            "fps_avg":     round(len(self.frames) / max(dur, 1e-3), 1),
        }

    def push(self, data: dict):
        if self.recording:
            self.frames.append({"t": round(time.time() - self._t0, 3), **data})

    def get_frame(self, idx: int) -> dict | None:
        if 0 <= idx < len(self.frames):
            return self.frames[idx]
        return None

    def clear(self):
        self.frames    = []
        self.recording = False


# ─────────────────────────────────────────────────────────────────────────────
# Angle computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_angles(landmarks: list[dict], smoothed: np.ndarray) -> list[dict]:
    results = []
    for d in ANGLE_DEFS:
        ai, bi, ci = d["a"], d["b"], d["c"]
        if max(ai, bi, ci) >= len(landmarks):
            results.append({"name": d["name"], "label": d["label"], "angle": None, "in_ideal": None, "ideal": list(d["ideal"])})
            continue
        vis = min(
            landmarks[ai].get("visibility", 0),
            landmarks[bi].get("visibility", 0),
            landmarks[ci].get("visibility", 0),
        )
        if vis < 0.3:
            results.append({"name": d["name"], "label": d["label"], "angle": None, "in_ideal": None, "ideal": list(d["ideal"])})
            continue
        ang = round(angle_3d(smoothed[ai], smoothed[bi], smoothed[ci]), 1)
        results.append({
            "name":     d["name"],
            "label":    d["label"],
            "angle":    ang,
            "in_ideal": bool(d["ideal"][0] <= ang <= d["ideal"][1]),
            "ideal":    list(d["ideal"]),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Posture analysis (rule-based, NumPy geometry)
# ─────────────────────────────────────────────────────────────────────────────

def _shoulder_drop(landmarks: list[dict]) -> tuple[float, str]:
    ls = vec3(landmarks[LM["LEFT_SHOULDER"]])
    rs = vec3(landmarks[LM["RIGHT_SHOULDER"]])
    w  = float(np.linalg.norm(ls - rs)) + 1e-9
    dy = float(ls[1] - rs[1])
    return abs(dy / w), ("left" if dy > 0 else "right")


def _spine_lean(landmarks: list[dict]) -> tuple[float, str]:
    ls = vec3(landmarks[LM["LEFT_SHOULDER"]])
    rs = vec3(landmarks[LM["RIGHT_SHOULDER"]])
    lh = vec3(landmarks[LM["LEFT_HIP"]])
    rh = vec3(landmarks[LM["RIGHT_HIP"]])
    spine = midpt(ls, rs) - midpt(lh, rh)
    ang   = float(np.degrees(np.arctan2(spine[0], -spine[1])))
    return abs(ang), ("right" if ang > 0 else "left")


def _forward_head(landmarks: list[dict]) -> float:
    nose = vec3(landmarks[LM["NOSE"]])
    ls   = vec3(landmarks[LM["LEFT_SHOULDER"]])
    rs   = vec3(landmarks[LM["RIGHT_SHOULDER"]])
    mid  = midpt(ls, rs)
    w    = float(np.linalg.norm(ls - rs)) + 1e-9
    return float((nose[2] - mid[2]) / w)   # positive = head forward


def _hip_tilt(landmarks: list[dict]) -> tuple[float, str]:
    lh = vec3(landmarks[LM["LEFT_HIP"]])
    rh = vec3(landmarks[LM["RIGHT_HIP"]])
    w  = float(np.linalg.norm(lh - rh)) + 1e-9
    dy = float(lh[1] - rh[1])
    return abs(dy / w), ("left" if dy > 0 else "right")


def analyse_posture(landmarks: list[dict], angles: list[dict], smoothed: np.ndarray) -> tuple[list[dict], int]:
    feedback: list[dict] = []
    score = 100
    vis = lambda i: landmarks[i].get("visibility", 0) if i < len(landmarks) else 0

    # Shoulder drop
    if vis(11) > 0.5 and vis(12) > 0.5:
        ratio, side = _shoulder_drop(landmarks)
        if ratio > 0.09:
            feedback.append({"type": "bad",  "title": "SHOULDER DROP",
                             "body": f"{side.title()} shoulder significantly lower. Level your shoulders."})
            score -= 22
        elif ratio > 0.04:
            feedback.append({"type": "warn", "title": "SHOULDER TILT",
                             "body": f"Slight {side}-side shoulder drop ({ratio*100:.0f}%). Try to even out."})
            score -= 9
        else:
            feedback.append({"type": "good", "title": "SHOULDERS",
                             "body": "Shoulder alignment is level ✓"})

    # Spine lean
    if all(vis(i) > 0.5 for i in [11, 12, 23, 24]):
        lean, direction = _spine_lean(landmarks)
        if lean > 14:
            feedback.append({"type": "bad",  "title": "SPINE LEAN",
                             "body": f"Leaning {direction} by {lean:.0f}°. Stand vertically."})
            score -= 20
        elif lean > 7:
            feedback.append({"type": "warn", "title": "SLIGHT LEAN",
                             "body": f"Mild {direction} lean ({lean:.0f}°). Shift your centre."})
            score -= 8
        else:
            feedback.append({"type": "good", "title": "SPINE",
                             "body": "Spine is nicely vertical ✓"})

    # Forward head
    if vis(0) > 0.5 and vis(11) > 0.5 and vis(12) > 0.5:
        fhr = _forward_head(landmarks)
        if fhr > 0.28:
            feedback.append({"type": "bad",  "title": "FORWARD HEAD",
                             "body": "Head jutting forward. Pull your chin back and lift your chest."})
            score -= 16
        elif fhr > 0.14:
            feedback.append({"type": "warn", "title": "HEAD POSITION",
                             "body": "Head slightly forward. Tuck your chin gently."})
            score -= 7

    # Hip tilt
    if vis(23) > 0.5 and vis(24) > 0.5:
        ratio, side = _hip_tilt(landmarks)
        if ratio > 0.06:
            feedback.append({"type": "warn", "title": "HIP TILT",
                             "body": f"{side.title()} hip raised. Distribute weight evenly."})
            score -= 7

    # Arm asymmetry
    le = next((a for a in angles if a["name"] == "L_ELBOW" and a["angle"]), None)
    re = next((a for a in angles if a["name"] == "R_ELBOW" and a["angle"]), None)
    if le and re:
        diff = abs(le["angle"] - re["angle"])
        if diff > 40:
            feedback.append({"type": "warn", "title": "ARM ASYMMETRY",
                             "body": f"Elbows differ by {diff:.0f}°. Check both arm positions."})
            score -= 6

    # Overall good
    if not feedback:
        feedback.append({"type": "good", "title": "STAND BY",
                         "body": "Step into frame for posture analysis."})

    return feedback, max(0, score)


# ─────────────────────────────────────────────────────────────────────────────
# Engine singleton (accessed from JavaScript)
# ─────────────────────────────────────────────────────────────────────────────

_smoother  = ButterworthSmoother(n=33, cutoff_hz=7.0, sample_hz=30.0)
_motion    = MotionTracker(window=8)
_recorder  = SessionRecorder()
_frame_idx = 0


def process_frame(lm_json: str) -> str:
    """
    Called from JS every MediaPipe frame.
    lm_json: JSON array of 33 landmark objects {x,y,z,visibility}
    Returns: JSON string with smoothed landmarks, angles, motion, feedback, score.
    """
    global _frame_idx

    landmarks: list[dict] = json.loads(lm_json)
    if len(landmarks) < 33:
        return json.dumps({"error": "Need 33 landmarks"})

    # Butterworth smooth
    smoothed = _smoother.update(landmarks)   # ndarray (33,3)

    # Build smoothed list for JS (3D render)
    smoothed_lm = [
        {"x": float(smoothed[i, 0]),
         "y": float(smoothed[i, 1]),
         "z": float(smoothed[i, 2]),
         "visibility": landmarks[i].get("visibility", 1.0)}
        for i in range(33)
    ]

    # Joint angles (NumPy geometry)
    angles = compute_angles(landmarks, smoothed)

    # Motion dynamics
    motion = _motion.update(smoothed)

    # Posture analysis
    feedback, score = analyse_posture(landmarks, angles, smoothed)

    # Record frame
    _recorder.push({
        "smoothed": smoothed_lm,
        "angles":   angles,
        "score":    score,
    })

    _frame_idx += 1

    return json.dumps({
        "frame":    _frame_idx,
        "smoothed": smoothed_lm,
        "angles":   angles,
        "motion":   motion,
        "feedback": feedback,
        "score":    score,
    })


def reset_smoother() -> str:
    _smoother.reset()
    return "ok"


def record_control(cmd: str) -> str:
    """cmd: 'start' | 'stop' | 'clear'"""
    if cmd == "start":
        _recorder.start()
        return json.dumps({"recording": True})
    elif cmd == "stop":
        info = _recorder.stop()
        return json.dumps({"recording": False, **info})
    elif cmd == "clear":
        _recorder.clear()
        return json.dumps({"recording": False, "frame_count": 0})
    return json.dumps({"error": "unknown command"})


def get_replay_frame(idx: int) -> str:
    frame = _recorder.get_frame(int(idx))
    if frame is None:
        return json.dumps({"end": True})
    return json.dumps({"idx": int(idx), **frame})


def get_session_info() -> str:
    return json.dumps({
        "recording":   _recorder.recording,
        "frame_count": len(_recorder.frames),
    })