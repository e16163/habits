"""
Quell — collect.py  (fixed)
Collect training data from your webcam.

Fixes vs original:
  • Tracks your FACE too — fingertip-to-head distances are now recorded,
    so the model can learn "hand near hair" not just "hand shape".
  • Each run gets a unique session ID, enabling honest session-aware
    train/test splits in train.py.
  • If quell_model.pkl exists, shows the CURRENT model's live prediction
    so you can see where it goes wrong while you record corrections.
  • Saves to landmarks.csv (appends if file already exists).

Controls:
  T — hold to record TOUCHING hair
  N — hold to record NOT touching
  C — clear all data collected THIS session (won't affect saved CSV)
  S — save and quit
  Q — quit without saving
"""

import cv2
import numpy as np
import csv
import os
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import time
from datetime import datetime
from features import build_feature_vector, feature_names, N_FEATURES, HAND_POINTS, HEAD_ANCHORS

# ── Download models if missing ────────────────────────────────────────────────

def download_if_missing(path, url, label):
    if not os.path.exists(path):
        print(f"Downloading {label} (~10–30 MB)…")
        urllib.request.urlretrieve(url, path)
        print("  Done.")

download_if_missing(
    "hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task",
    "hand landmarker"
)
download_if_missing(
    "face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task",
    "face landmarker"
)

# ── MediaPipe detectors ───────────────────────────────────────────────────────

hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# ── Load existing model for live preview (optional) ───────────────────────────

live_model = None
if os.path.exists("quell_model.pkl"):
    try:
        with open("quell_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        live_model = bundle["pipeline"] if isinstance(bundle, dict) else bundle
        print("Loaded quell_model.pkl — live predictions enabled.")
    except Exception as e:
        print(f"Could not load model: {e}")

# ── Webcam ────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)

# ── Session ID ────────────────────────────────────────────────────────────────
# Every run of collect.py is a new session.
# train.py uses this to split by session, not by frame.

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Session ID: {SESSION_ID}")

# ── Drawing helpers ───────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

COLOR_OK    = (140, 210, 90)
COLOR_WARN  = (60, 100, 255)
COLOR_WHITE = (240, 240, 240)
COLOR_DARK  = (30, 30, 30)
COLOR_FACE  = (200, 160, 80)

# ── State ─────────────────────────────────────────────────────────────────────

rows          = []       # collected this session
touch_count   = 0
no_touch_count= 0
SAVE_PATH     = "landmarks.csv"

print("Controls:  T = touching hair  |  N = not touching  |  C = clear session  |  S = save  |  Q = quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    key = cv2.waitKey(1) & 0xFF

    recording = False
    label     = None

    if key == ord('t') or key == ord('T'):
        recording, label = True, 1
    elif key == ord('n') or key == ord('N'):
        recording, label = True, 0
    elif key == ord('c') or key == ord('C'):
        rows, touch_count, no_touch_count = [], 0, 0
        print("Session data cleared.")
        continue
    elif key == ord('s') or key == ord('S'):
        break
    elif key == ord('q') or key == ord('Q'):
        rows = []
        break

    # ── Detect face ───────────────────────────────────────────────────────────

    face_lms = None
    if face_result.face_landmarks:
        face_lms = face_result.face_landmarks[0]

        # Draw all head anchors
        for anc_idx in HEAD_ANCHORS:
            lm = face_lms[anc_idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 6, COLOR_FACE, -1)
            cv2.circle(frame, (cx, cy), 6, COLOR_WHITE, 1)

    # ── Detect hands ─────────────────────────────────────────────────────────

    hand_detected = bool(hand_result.hand_landmarks)
    live_pred     = None
    live_conf     = None

    for landmarks in hand_result.hand_landmarks:

        # Draw skeleton
        for a, b in HAND_CONNECTIONS:
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, COLOR_OK, -1)

        # Draw lines from all hand points to all head anchors (shows what model sees)
        if face_lms:
            for hp_idx in HAND_POINTS:
                hx, hy = int(landmarks[hp_idx].x * w), int(landmarks[hp_idx].y * h)
                for anc_idx in HEAD_ANCHORS:
                    ax, ay = int(face_lms[anc_idx].x * w), int(face_lms[anc_idx].y * h)
                    cv2.line(frame, (hx, hy), (ax, ay), (80, 80, 160), 1)

        # Build feature vector
        feat = build_feature_vector(landmarks, face_lms)

        # Live model prediction
        if live_model is not None:
            arr       = feat.reshape(1, -1)
            lp        = live_model.predict(arr)[0]
            lc        = live_model.predict_proba(arr)[0][int(lp)]
            if lp == 1:                        # touching wins if any hand says so
                live_pred = 1
                live_conf = max(live_conf or 0, lc)
            elif live_pred is None:
                live_pred = 0
                live_conf = lc

        # Record if key held
        if recording:
            rows.append(feat.tolist() + [SESSION_ID, label])
            if label == 1:
                touch_count   += 1
            else:
                no_touch_count+= 1

    # ── UI ────────────────────────────────────────────────────────────────────

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if recording and hand_detected:
        rec_color = COLOR_WARN if label == 1 else COLOR_OK
        rec_text  = "● REC: touching hair" if label == 1 else "● REC: not touching"
        cv2.putText(frame, rec_text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)
    elif recording:
        cv2.putText(frame, "no hand detected", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)
    else:
        cv2.putText(frame, "T = touching  |  N = not touching  |  C = clear  |  S = save",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_WHITE, 1)

    # Live model prediction panel (right side, only if model loaded)
    if live_model is not None and live_pred is not None:
        pred_color = COLOR_WARN if live_pred == 1 else COLOR_OK
        pred_text  = "model: TOUCHING" if live_pred == 1 else "model: clear"
        conf_pct   = f"{int(live_conf * 100)}%"

        # Background pill
        panel_x = w - 200
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel_x - 8, 60), (w, 110), COLOR_DARK, -1)
        cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, pred_text,  (panel_x, 82),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, pred_color, 2)
        cv2.putText(frame, f"conf {conf_pct}", (panel_x, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

        # Confidence bar
        bar_w = int((w - panel_x) * live_conf)
        cv2.rectangle(frame, (panel_x - 8, 108), (w, 112), (60, 60, 60), -1)
        cv2.rectangle(frame, (panel_x - 8, 108), (panel_x - 8 + bar_w, 112), pred_color, -1)

    elif live_model is None:
        cv2.putText(frame, "no model loaded", (w - 190, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # Face status indicator
    face_status_color = COLOR_FACE if face_lms else (80, 80, 80)
    face_status_text  = "face ✓" if face_lms else "face ✗"
    cv2.putText(frame, face_status_text, (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_status_color, 1)

    # Bottom bar — sample counts
    bottom = frame.copy()
    cv2.rectangle(bottom, (0, h - 36), (w, h), COLOR_DARK, -1)
    cv2.addWeighted(bottom, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame,
                f"touching: {touch_count}   not touching: {no_touch_count}   "
                f"session: {SESSION_ID}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)

    cv2.imshow("Quell — Data Collection", frame)

cap.release()
cv2.destroyAllWindows()

# ── Save ──────────────────────────────────────────────────────────────────────

if rows:
    header   = feature_names() + ["session", "label"]
    file_exists = os.path.exists(SAVE_PATH) and os.path.getsize(SAVE_PATH) > 0

    with open(SAVE_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)

    print(f"\nAppended {len(rows)} samples to {SAVE_PATH}")
    print(f"  Touching:     {touch_count}")
    print(f"  Not touching: {no_touch_count}")
    print(f"  Session ID:   {SESSION_ID}")
    print(f"\nRun train.py to retrain.")
else:
    print("No data saved.")
