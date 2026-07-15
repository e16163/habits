# Habits

Real-time hand-to-face contact and seated-posture classifiers using MediaPipe landmarks. Detects hair-touching and personalized good/bad posture from a webcam, with separate data-collection, training, and live-inference pipelines. 90% reduction of the hair-touching habit over 2 months.


## Overview

Habits implements an end-to-end ML workflow:
1. **Collection**: Real-time hand and face landmark detection, feature extraction, session-based labeling
2. **Training**: Multi-model evaluation (Random Forest, Gradient Boosting, Logistic Regression) with session-aware train/test splitting
3. **Inference**: Live prediction display during data collection with model confidence scoring

The hair-touching system extracts 136 hand-shape and depth-aware hand-to-head features. The optional posture layer extracts normalized 2D/3D upper-body landmarks plus interpretable alignment features. Session IDs enable honest evaluation by keeping temporally adjacent frames together.

## Posture Module

The posture module is personalized: you record examples of your own good and bad seated posture, then train a separate `posture_model.pkl`. It includes:

- Normalized image and MediaPipe world landmarks for the head, shoulders, elbows, and hips
- Engineered shoulder, hip, torso, neck, centering, and depth features
- Pose-visibility gating so partially hidden bodies are not recorded or scored
- Session-held-out model evaluation when multiple recording sessions exist
- Exponential probability smoothing, separate warning/recovery thresholds, and dwell times
- A conservative geometry preview before a personal model has been trained
- Up to five-fold stratified group validation that keeps complete sessions together
- Extra Trees, Random Forest, robust linear, and soft-voting ensemble comparison
- Cross-validated decision-threshold tuning instead of assuming 50% is optimal
- A robust personal good-posture profile for baseline-relative explanations
- Unfamiliar-pose detection to flag inputs outside the model's training range
- Full-data refitting after validation, so the final model learns from every session

### Front view versus side view

A front-facing webcam is sufficient for a useful personalized classifier. It is strongest at detecting sideways torso lean, uneven shoulders, head tilt/centering, and the overall visual pattern of slouching. MediaPipe's world landmarks provide some depth evidence as well.

A side view is recommended when forward-head posture or rounded-back slouch is the main target. Those movements are much less ambiguous in profile. Do not mix front and side frames casually in a small dataset: train with a consistent front view first, or collect several labeled sessions of each view so the model can learn both.

## Architecture

### Data Collection (collect.py)
- Webcam capture with OpenCV
- Parallel MediaPipe detection: 21-point hand landmarks + 468-point face landmarks
- Feature vector: 63 normalized hand coordinates + 72 depth-aware hand-to-head values + 1 global depth cue (136 total)
- Session ID tracking (YYYYMMDD_HHMMSS) for train/test grouping
- Optional: Live model predictions overlaid on camera feed
- Output: Appends to `landmarks.csv`

### Training (train.py)
- Loads and normalizes feature matrix
- Detects multiple sessions; uses `GroupShuffleSplit` (80/20 split) if available
- Falls back to stratified random split for single-session data
- Trains three pipelines:
  - Random Forest: 200 trees, max_depth=10, balanced class weights
  - Gradient Boosting: 200 estimators, max_depth=4, subsample=0.8
  - Logistic Regression: StandardScaler preprocessing, C=0.1
- Reports accuracy, confusion matrix, classification report per model
- Saves best model (by test accuracy) to `quell_model.pkl`

### Feature Extraction (features.py)
- Hand points: [0, 4, 8, 12, 16, 20] (fingertips + wrist)
- Head anchors: [10, 151, 234, 454, 127, 356] (forehead, skull, temples, and head sides)
- Hand normalization: wrist-centered and scaled by palm size
- Depth features: 3D distance and z-offset for each hand point/head anchor pair
- 136 total features

## Installation

```bash
pip install opencv-python mediapipe scikit-learn pandas numpy

python collect.py
```

First run auto-downloads MediaPipe task files (~40MB total).

## Usage

### Collect Data

```bash
python collect.py
```

**Controls:**
- **T** — Record "touching hair" (label: 1)
- **N** — Record "not touching" (label: 0)
- **C** — Clear session data (doesn't affect saved CSV)
- **S** — Save and quit
- **Q** — Quit without saving

**Output:**
- `landmarks.csv`: Appends rows with 136 features + session ID + label
- Displays live predictions if `quell_model.pkl` exists
- Shows sample counts at bottom

### Train Model

```bash
python train.py
```

**Output:**
- Accuracy metrics (train/test %)
- Confusion matrix
- Classification report per model
- `quell_model.pkl`: Best performing pipeline

Example:
```
Train: 450   Test: 112

Training Random Forest …
  Train: 94.2%   Test: 88.9%   Gap: 5.3%

Training Gradient Boosting …
  Train: 93.8%   Test: 91.2%   Gap: 2.6%

Best: Gradient Boosting (test 91.2% / train 93.8%)
```

### Use Model

```python
import pickle
from features import build_feature_vector

model = pickle.load(open("quell_model.pkl", "rb"))["pipeline"]

# Single prediction
feat = build_feature_vector(hand_landmarks, face_landmarks)
pred = model.predict(feat.reshape(1, -1))[0]
conf = model.predict_proba(feat.reshape(1, -1))[0][int(pred)]
```

### Collect and Train Posture

Keep your head and both shoulders visible. Hips are optional: when they are in frame the model adds torso alignment and depth cues; when they are below a normal desk-camera crop those inputs are masked and the head/shoulder model continues working. Record both classes in varied but realistic positions, lighting, and clothing. Several shorter sessions are more valuable than one long session.

```bash
# Camera index defaults to 1 (the physical webcam on this setup).
python collect_posture.py --view front

# Repeat collection in at least 3 sessions, then train.
python train_posture.py

# Run posture by itself.
python posture.py
```

Camera index 0 is OBS Virtual Camera on this setup. To use OBS intentionally,
pass `--camera 0`. The combined monitor can be overridden in the same way with
`QUELL_CAMERA=0 python webcam.py`.

Posture collection controls:

- **G** — hold to record good posture
- **B** — hold to record bad posture
- **C** — clear samples from the current run
- **S** — save and quit
- **Q** — discard this run

After `posture_model.pkl` exists, `webcam.py` automatically enables the posture layer alongside hair-touch detection.

For a reliable first model, collect at least three sessions on different days or after moving naturally between runs. In each session, capture normal variation—not only one rigid "good" pose and one exaggerated "bad" pose. Collection is rate-limited to roughly eight samples per second to reduce highly correlated duplicate frames. Training reports cross-validated balanced accuracy, bad-posture F1, pose diversity, a tuned threshold, and the most useful features. The output is behavioral feedback, not a medical diagnosis.

## Data Format

### landmarks.csv
```csv
image-hand and depth feature columns...,session,label
12.3,14.5,...,35.2,20250707_143022,1
11.2,13.8,...,34.1,20250707_143022,1
45.6,48.2,...,62.3,20250707_143022,0
```

Column count: 138 (136 features + session + label)

### quell_model.pkl
Python pickle containing:
```python
{"pipeline": <sklearn.pipeline.Pipeline>}
```

## Design Notes

**Session Awareness**: Each `collect.py` run generates a unique session ID. `train.py` detects multiple sessions and uses `GroupShuffleSplit` to prevent data leakage (same person, same pose in both train and test). Single-session datasets fall back to stratified random split.

**Feature Normalization**: Distance features normalized by face height for scale invariance. Random Forest doesn't require feature scaling; Logistic Regression uses StandardScaler in pipeline.

**Class Imbalance**: Random Forest and Logistic Regression use `class_weight="balanced"` to handle label imbalance.

## Common Issues

**Low accuracy (<80%)**
- Collect more samples (aim for 500+)
- Collect across multiple sessions and lighting conditions
- Verify labels are accurate during collection

**High train/test gap (>10%)**
- Reduce model complexity (e.g., max_depth=8 for RF)
- Collect more varied data
- Check for systematic labeling errors

**"No hand detected"**
- Ensure full hand visibility in frame
- Improve lighting
- Move closer to camera

## Requirements

- Python 3.8+
- OpenCV 4.5+
- MediaPipe 0.8+
- scikit-learn 1.0+
- pandas 1.3+, numpy 1.20+

## Performance

- Collection: ~30 FPS (GPU optional)
- Feature extraction: ~5ms per frame
- Training: ~2–5 seconds per model (450 training samples)
- Inference: <1ms per prediction
