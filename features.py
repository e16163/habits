"""
Quell — features.py
Shared feature extraction. Import in collect.py, train.py, and webcam.py.

Feature vector layout  (136 numbers total):
  [0:63]    21 hand landmarks × (x,y,z), wrist-centred + scale-normalised
  [63:135]  72 depth-aware hand-to-head features
              6 hand points × 6 head anchors × 2 values:
                • 3D Euclidean distance
                • z-delta (hand_point.z − anchor.z)
                  negative = hand closer to camera = likely touching
  [135]     1 global depth cue: wrist.z − nose_tip.z
              negative = hand in front of face plane

Hand points (6):
  0  = wrist
  4  = thumb tip
  8  = index tip
  12 = middle tip
  16 = ring tip
  20 = pinky tip

Head anchors (6) — covers the full hair region:
  10  = forehead centre
  151 = top of skull
  234 = right temple
  454 = left temple
  127 = right side of head
  356 = left side of head

Nose tip anchor (depth reference):
  1   = nose tip  (used only for global wrist z offset)
"""

import numpy as np

HAND_POINTS  = [0, 4, 8, 12, 16, 20]
HEAD_ANCHORS = [10, 151, 234, 454, 127, 356]
NOSE_TIP     = 1

N_HAND_FEATS  = 63
N_DEPTH_FEATS = len(HAND_POINTS) * len(HEAD_ANCHORS) * 2   # 6×6×2 = 72
N_GLOBAL_FEATS= 1
N_FEATURES    = N_HAND_FEATS + N_DEPTH_FEATS + N_GLOBAL_FEATS  # 136


def normalise_hand(hand_landmarks_list) -> np.ndarray:
    """
    Returns (63,) wrist-centred, scale-normalised hand landmark array.
    """
    row = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_list])
    row = row - row[0]
    scale = np.linalg.norm(row[9])
    if scale > 1e-6:
        row = row / scale
    return row.flatten()


def depth_features(hand_landmarks_list, face_landmarks_list) -> np.ndarray:
    """
    Returns (72,) array — for each hand point × head anchor:
      [3D distance,  z-delta]
    """
    feats = []
    for hp_idx in HAND_POINTS:
        hp = hand_landmarks_list[hp_idx]
        for anc_idx in HEAD_ANCHORS:
            anc = face_landmarks_list[anc_idx]
            d3 = np.sqrt(
                (hp.x - anc.x) ** 2 +
                (hp.y - anc.y) ** 2 +
                (hp.z - anc.z) ** 2
            )
            dz = hp.z - anc.z
            feats.extend([d3, dz])
    return np.array(feats, dtype=np.float32)


def global_depth(hand_landmarks_list, face_landmarks_list) -> np.ndarray:
    """
    Returns (1,) — wrist.z minus nose_tip.z.
    Negative = hand is in front of the face plane = closer to camera.
    """
    wrist = hand_landmarks_list[0]
    nose  = face_landmarks_list[NOSE_TIP]
    return np.array([wrist.z - nose.z], dtype=np.float32)


def build_feature_vector(hand_landmarks_list, face_landmarks_list=None) -> np.ndarray:
    """
    Full 136-feature vector for one hand.
    If face_landmarks_list is None, depth features are zeroed out.
    """
    hand_feats = normalise_hand(hand_landmarks_list)

    if face_landmarks_list is not None:
        d_feats = depth_features(hand_landmarks_list, face_landmarks_list)
        g_feats = global_depth(hand_landmarks_list, face_landmarks_list)
    else:
        d_feats = np.zeros(N_DEPTH_FEATS,   dtype=np.float32)
        g_feats = np.zeros(N_GLOBAL_FEATS,  dtype=np.float32)

    return np.concatenate([hand_feats, d_feats, g_feats]).astype(np.float32)


def feature_names() -> list:
    names = []
    for i in range(21):
        names += [f"x{i}", f"y{i}", f"z{i}"]
    hp_labels  = {0:"wrist", 4:"thumb", 8:"index", 12:"mid", 16:"ring", 20:"pinky"}
    anc_labels = {10:"forehead", 151:"skull_top", 234:"r_temple",
                  454:"l_temple",  127:"r_side",    356:"l_side"}
    for hp in HAND_POINTS:
        for anc in HEAD_ANCHORS:
            names.append(f"{hp_labels[hp]}_{anc_labels[anc]}_dist3d")
            names.append(f"{hp_labels[hp]}_{anc_labels[anc]}_zdelta")
    names.append("wrist_nose_zdelta")
    return names
