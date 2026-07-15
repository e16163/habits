# Quell

Quell is a local webcam app for tracking hair-touching habits and sitting posture. It shows live feedback, session stats, trends, and intervention prompts. Camera frames stay on your computer.

## Setup

Requires Python 3.10 or newer.

```bash
git clone <repo-url>
cd habits
python -m pip install -e .
```

## Run the app

```bash
quell --camera 1
```

Or run it without installing the command:

```bash
python quell_app.py --camera 1
```

Use `--demo` to preview the interface without trained models:

```bash
python quell_app.py --demo
```

Controls:

- `P` — pause
- `D` — toggle details
- `M` — mirror camera
- `F` — fullscreen
- `Q` or `Esc` — quit

## Train the hair-touching model

Collect examples:

```bash
python collect.py --camera 1
```

- `T` — touching hair
- `N` — not touching hair
- `C` — clear samples
- `S` — save
- `Q` — quit

Then train:

```bash
python train.py
```

## Train the posture model

Collect front-view examples:

```bash
python collect_posture.py --camera 1 --view front
```

- `G` — good posture
- `B` — bad posture
- `C` — clear samples
- `S` — save
- `Q` — quit

Record several sessions with different clothing, lighting, chair positions, and natural good/bad posture. Then train:

```bash
python train_posture.py --view front
```

Keep your head and both shoulders visible. Hips improve the reading but are optional for normal desk framing.

## Camera selection

Camera numbers can change on macOS. If the wrong camera opens or initialization fails, try another index:

```bash
python quell_app.py --camera 0
python quell_app.py --camera 1
python quell_app.py --camera 2
```

An iPhone may appear as a camera through Continuity Camera. Disconnect it or choose a different index if you want the Mac camera.

## Local files

Training and history files are created locally and ignored by Git:

- `landmarks.csv`
- `quell_model.pkl`
- `posture_landmarks.csv`
- `posture_model.pkl`
- `quell_history.json`

## Troubleshooting

- **Not in frame:** move back slightly and keep your head and both shoulders visible.
- **Poor predictions:** collect more balanced examples and retrain.
- **Old or incompatible data:** remove the relevant CSV/model files, recollect, and retrain.
- **MediaPipe feedback warnings:** these are usually harmless if the camera feed and landmarks appear.

Quell is a personal awareness tool, not a medical device.
