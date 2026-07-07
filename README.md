# Habits

Hand-to-face contact classifier using MediaPipe landmarks. Detects habit patterns (specifically hair-touching) via machine learning from webcam feed. Includes data collection pipeline, multi-model training, and live inference visualization.

## Overview

Habits implements an end-to-end ML workflow:
1. **Collection**: Real-time hand and face landmark detection, feature extraction, session-based labeling
2. **Training**: Multi-model evaluation (Random Forest, Gradient Boosting, Logistic Regression) with session-aware train/test splitting
3. **Inference**: Live prediction display during data collection with model confidence scoring

The system extracts 120 distance-based features from hand keypoints to head anchors, normalized by face height. Session IDs enable honest cross-validation by grouping frames temporally.

## Architecture

### Data Collection (collect.py)
- Webcam capture with OpenCV
- Parallel MediaPipe detection: 21-point hand landmarks + 468-point face landmarks
- Feature vector: Euclidean distances from 10 hand keypoints to 12 head anchors (120 total)
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
- Head anchors: [10, 152, 6, 234, 454, 116, 345, 172, 397, 176, 400, 9]
- Distance normalization: divide by face height to achieve scale invariance
- 120 total features (10 hand × 12 head)

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
- `landmarks.csv`: Appends rows with 120 features + session ID + label
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

## Data Format

### landmarks.csv
```csv
hand_0_head_0,hand_0_head_1,...,hand_9_head_11,session,label
12.3,14.5,...,35.2,20250707_143022,1
11.2,13.8,...,34.1,20250707_143022,1
45.6,48.2,...,62.3,20250707_143022,0
```

Column count: 122 (120 features + session + label)

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
