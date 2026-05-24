"""
Quell — webcam.py
"""

import cv2, pickle, numpy as np, time, math, subprocess, os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import urllib.request
from features import build_feature_vector, HAND_POINTS, HEAD_ANCHORS

# ── Download models ───────────────────────────────────────────────────────────

def grab(path, url, label):
    if not os.path.exists(path):
        print(f"Downloading {label}…")
        urllib.request.urlretrieve(url, path)

grab("hand_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
     "hand landmarker")
grab("face_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
     "face landmarker")

# ── Model ─────────────────────────────────────────────────────────────────────

with open("quell_model.pkl", "rb") as f:
    b = pickle.load(f)
model = b["pipeline"] if isinstance(b, dict) else b

# ── MediaPipe ─────────────────────────────────────────────────────────────────

hand_det = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2, min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))

face_det = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
        num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5))

CAMERA_INDEX = 2
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.5)

# ── Palette  (BGR) ────────────────────────────────────────────────────────────

MINT_BGR  = (140, 220, 80)
RED_BGR   = (60,  60,  240)
MINT_RGB  = (80,  220, 140)
RED_RGB   = (240, 60,  60)
PANEL     = (20,  18,  16)
WHITE_RGB = (240, 240, 248)
DIM_RGB   = (90,  90,  105)
SUBDIM    = (55,  55,  68)

HAND_SKEL = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

# ── Fonts ─────────────────────────────────────────────────────────────────────

def load_font(size, bold=False):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size, index=(1 if bold and path.endswith(".ttc") else 0))
        except: pass
    return ImageFont.load_default()

F_APP   = load_font(12, bold=True)
F_STAT  = load_font(14, bold=True)
F_BIG   = load_font(44, bold=True)
F_MED   = load_font(13)
F_SM    = load_font(10)

# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_text(frame, text, xy, font, color, anchor="lt"):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(xy, text, font=font, fill=color, anchor=anchor)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def tw(text, font):
    b = font.getbbox(text); return b[2]-b[0]

def draw_bar(frame, x1, y1, x2, y2, fill_ratio, color_bgr, bg=(38,36,34)):
    cv2.rectangle(frame, (x1,y1), (x2,y2), bg, -1)
    filled = int((x2-x1) * max(0, min(1, fill_ratio)))
    if filled: cv2.rectangle(frame, (x1,y1), (x1+filled,y2), color_bgr, -1)

def vignette(frame, color_bgr, strength):
    h, w = frame.shape[:2]
    brd  = int(min(h,w)*0.2)
    mask = np.zeros((h,w), dtype=np.float32)
    for i in range(brd):
        t = ((1-i/brd)**2) * strength
        mask[i,:] = mask[h-1-i,:] = np.maximum(mask[i,:], t)
        mask[:,i] = mask[:,w-1-i] = np.maximum(mask[:,i], t)
    tint = np.full_like(frame, color_bgr, dtype=np.float32)
    out  = frame.astype(np.float32)*(1-mask[:,:,None]) + tint*mask[:,:,None]
    return np.clip(out,0,255).astype(np.uint8)

# ── Notifications ─────────────────────────────────────────────────────────────

NOTIFY_COOLDOWN = 12
_last_notify = 0.0

def alert(count):
    global _last_notify
    if time.time() - _last_notify < NOTIFY_COOLDOWN: return
    _last_notify = time.time()

    # Close all tabs in Chrome and Safari
    subprocess.Popen(["osascript", "-e", """
        tell application "Google Chrome"
            if it is running then
                repeat with w in windows
                    repeat with t in tabs of w
                        close t
                    end repeat
                end repeat
            end if
        end tell
        tell application "Safari"
            if it is running then
                repeat with w in windows
                    close w
                end repeat
            end if
        end tell
    """], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Banner
    subprocess.Popen(["osascript", "-e",
        f'display notification "Touch #{count} — tabs closed" '
        f'with title "Quell"'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── State ─────────────────────────────────────────────────────────────────────

touch_count   = 0
is_touching   = False
pred_buf      = []
SMOOTH        = 3
session_start = time.time()

# ── Layout constants ──────────────────────────────────────────────────────────
# Everything in two bars — nothing floats over the video feed.

TOP_H  = 48   # top bar height
BOT_H  = 80   # bottom bar height

print("Running — Q to quit")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    frame   = cv2.flip(frame, 1)
    H, W    = frame.shape[:2]
    now     = time.time()
    elapsed = int(now - session_start)
    mm, ss  = divmod(elapsed, 60)

    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h_res   = hand_det.detect(mp_img)
    f_res   = face_det.detect(mp_img)

    face_lms   = f_res.face_landmarks[0] if f_res.face_landmarks else None
    prediction = 0
    confidence = 0.0

    # Skeleton — very subtle
    for lms in h_res.hand_landmarks:
        for a,b in HAND_SKEL:
            cv2.line(frame,
                     (int(lms[a].x*W), int(lms[a].y*H)),
                     (int(lms[b].x*W), int(lms[b].y*H)),
                     (50,50,60), 1)
        for lm in lms:
            cv2.circle(frame, (int(lm.x*W), int(lm.y*H)), 2, (75,75,90), -1)

        feat = build_feature_vector(lms, face_lms)
        pred = model.predict(feat.reshape(1,-1))[0]
        conf = model.predict_proba(feat.reshape(1,-1))[0][int(pred)]
        if pred == 1:   prediction = 1; confidence = max(confidence, conf)
        elif prediction == 0: confidence = max(confidence, conf)

    # Small face anchor dots — only if face detected, very minimal
    if face_lms:
        for idx in HEAD_ANCHORS:
            lm = face_lms[idx]
            cv2.circle(frame, (int(lm.x*W), int(lm.y*H)), 2, (80,130,180), -1)

    # Smoothing
    pred_buf.append(prediction)
    if len(pred_buf) > SMOOTH: pred_buf.pop(0)
    smoothed = 1 if sum(pred_buf) >= SMOOTH else 0

    if not h_res.hand_landmarks:
        is_touching = False; pred_buf.clear()
    elif smoothed == 1 and not is_touching:
        is_touching = True; touch_count += 1
        alert(touch_count)
    elif smoothed == 0:
        is_touching = False

    accent_bgr = RED_BGR  if smoothed else MINT_BGR
    accent_rgb = RED_RGB  if smoothed else MINT_RGB

    # Vignette when touching
    if smoothed:
        frame = vignette(frame, RED_BGR, 0.28 + 0.08*math.sin(now*7))

    # ── TOP BAR ───────────────────────────────────────────────────────────────

    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W, TOP_H), PANEL, -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, TOP_H), (W, TOP_H), (42,40,38), 1)

    # Left: dot + QUELL
    cv2.circle(frame, (18, TOP_H//2), 5, accent_bgr, -1)
    frame = pil_text(frame, "QUELL", (30, TOP_H//2-7), F_APP, WHITE_RGB)

    # Centre: status
    status = "TOUCHING HAIR" if smoothed else "HANDS CLEAR"
    sw = tw(status, F_STAT)
    cx = W//2 - sw//2
    # coloured underline behind text
    cv2.rectangle(frame, (cx-8, TOP_H-3), (cx+sw+8, TOP_H-1), accent_bgr, -1)
    frame = pil_text(frame, status, (cx, TOP_H//2-7), F_STAT, accent_rgb)

    # Right: face lock + timer
    face_label = "FACE ✓" if face_lms else "FACE ✗"
    face_col   = (120,200,160) if face_lms else (100,80,80)
    face_dot   = (0,200,130)   if face_lms else (90,60,60)
    timer_str  = f"{mm:02d}:{ss:02d}"
    t_x = W - 6 - tw(timer_str, F_SM)
    frame = pil_text(frame, timer_str,   (t_x,   TOP_H//2-5), F_SM, DIM_RGB)
    f_x   = t_x - 8 - tw(face_label, F_SM)
    cv2.circle(frame, (f_x-6, TOP_H//2), 3, face_dot, -1)
    frame = pil_text(frame, face_label,  (f_x,   TOP_H//2-5), F_SM, face_col)

    # ── BOTTOM BAR ────────────────────────────────────────────────────────────

    by = H - BOT_H
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, by), (W, H), PANEL, -1)
    cv2.addWeighted(ov2, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, by), (W, by), (42,40,38), 1)

    pad = 22

    # Touch count — left
    c_str = str(touch_count)
    frame = pil_text(frame, c_str,      (pad, by+8),  F_BIG, accent_rgb)
    cw    = tw(c_str, F_BIG)
    frame = pil_text(frame, "touches",  (pad, by+56), F_SM,  DIM_RGB)

    # Divider
    div_x = pad + cw + 20
    cv2.line(frame, (div_x, by+12), (div_x, H-12), (48,46,44), 1)

    # Confidence bar — middle
    bar_x1 = div_x + 16
    bar_x2 = W - pad - 80
    bar_y  = by + BOT_H//2
    draw_bar(frame, bar_x1, bar_y-3, bar_x2, bar_y+3, confidence, accent_bgr)
    frame = pil_text(frame, "confidence", (bar_x1, by+14), F_SM, DIM_RGB)
    frame = pil_text(frame, f"{int(confidence*100)}%",
                     (bar_x2+8, bar_y-6), F_SM, DIM_RGB)

    # Session time — right
    sess_str = f"{mm:02d}:{ss:02d}"
    frame = pil_text(frame, sess_str,   (W-pad-tw(sess_str,F_MED), by+14), F_MED, WHITE_RGB)
    frame = pil_text(frame, "session",  (W-pad-tw("session",F_SM), by+32), F_SM,  SUBDIM)

    cv2.imshow("Quell", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone — {touch_count} touch{'es' if touch_count!=1 else ''} in {mm:02d}:{ss:02d}")
