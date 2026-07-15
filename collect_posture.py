"""Collect personalized good/bad posture examples from a webcam.

Controls:
  G — hold to record good posture
  B — hold to record bad posture
  C — clear samples collected in this run
  S — save and quit
  Q — quit without saving
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

from posture_features import (
    build_posture_feature_vector,
    feature_names,
    heuristic_bad_probability,
    pose_quality,
    posture_issues,
    posture_metrics,
)
from posture_runtime import (
    assess_posture,
    create_pose_detector,
    draw_upper_body_pose,
    load_posture_model,
)


GOOD, BAD = 0, 1
MIN_QUALITY = 0.55
SAMPLE_INTERVAL_SECONDS = 0.12  # ~8 Hz avoids thousands of near-identical frames


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="OpenCV camera index (1 on this setup; use 0 for OBS Virtual Camera)",
    )
    parser.add_argument(
        "--view",
        choices=("front", "side"),
        default="front",
        help="camera view stored with this training session",
    )
    parser.add_argument("--output", default="posture_landmarks.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = create_pose_detector()
    try:
        model, model_bundle = load_posture_model()
        if model is not None:
            print("Loaded posture_model.pkl — live preview enabled.")
    except Exception as error:
        model = None
        model_bundle = None
        print(f"Could not load posture model: {error}")

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1.0)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Try --camera 1 for the physical "
            "camera or --camera 0 for OBS Virtual Camera."
        )

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[list] = []
    counts = {GOOD: 0, BAD: 0}
    save_requested = False
    last_sample_time = 0.0
    print(f"Session: {session_id} ({args.view} view)")
    print("Hold G = good | B = bad | C = clear | S = save | Q = discard")

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = detector.detect(mp_image)

        key = cv2.waitKey(1) & 0xFF
        label = None
        if key in (ord("g"), ord("G")):
            label = GOOD
        elif key in (ord("b"), ord("B")):
            label = BAD
        elif key in (ord("c"), ord("C")):
            rows.clear()
            counts = {GOOD: 0, BAD: 0}
            print("Session samples cleared.")
        elif key in (ord("s"), ord("S")):
            save_requested = True
            break
        elif key in (ord("q"), ord("Q")):
            break

        probability = None
        assessment = None
        issues: list[str] = []
        metrics: dict[str, float] = {}
        quality = 0.0
        pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        world_landmarks = (
            result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        )

        if pose_landmarks is not None:
            quality = pose_quality(pose_landmarks)
            draw_upper_body_pose(
                frame,
                pose_landmarks,
                color=(90, 215, 110) if quality >= MIN_QUALITY else (80, 100, 180),
            )
            if quality >= MIN_QUALITY:
                features = build_posture_feature_vector(
                    pose_landmarks, world_landmarks
                )
                metrics = posture_metrics(features)
                if model is not None:
                    assessment = assess_posture(model, model_bundle, features)
                    probability = assessment["probability"]
                    issues = posture_issues(metrics, assessment["good_profile"])
                    if assessment["is_bad"] and not issues:
                        issues = ["combined posture pattern differs from baseline"]
                else:
                    probability = heuristic_bad_probability(metrics)
                    issues = posture_issues(metrics)
                sample_time = time.monotonic()
                if (
                    label is not None
                    and sample_time - last_sample_time >= SAMPLE_INTERVAL_SECONDS
                ):
                    rows.append(
                        features.tolist() + [session_id, args.view, int(label)]
                    )
                    counts[label] += 1
                    last_sample_time = sample_time

        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 82), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        if label is None:
            status = "G good  |  B bad  |  C clear  |  S save"
            color = (235, 235, 235)
        elif quality < MIN_QUALITY:
            status = "POSE NOT CLEAR - keep head and both shoulders visible"
            color = (80, 120, 245)
        else:
            status = "RECORDING GOOD POSTURE" if label == GOOD else "RECORDING BAD POSTURE"
            color = (100, 225, 120) if label == GOOD else (80, 100, 245)
        cv2.putText(frame, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2)
        cv2.putText(
            frame,
            f"good {counts[GOOD]}   bad {counts[BAD]}   view {args.view}   pose quality {quality:.0%}",
            (18, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 205, 205),
            1,
        )

        if probability is not None:
            threshold = assessment["threshold"] if assessment else 0.5
            model_label = "BAD" if probability >= threshold else "GOOD"
            model_color = (
                (60, 80, 245) if probability >= threshold else (90, 220, 110)
            )
            novelty = "  UNFAMILIAR POSE" if assessment and assessment["is_unfamiliar"] else ""
            text = (
                f"preview: {model_label}  {probability:.0%} bad  "
                f"threshold {threshold:.0%}{novelty}"
            )
            cv2.putText(
                frame,
                text,
                (18, height - 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                model_color,
                2,
            )
            torso_text = (
                f"{metrics['torso_lean_deg']:+.1f} deg"
                if metrics["hip_visibility"] >= 0.35
                else "n/a (hips optional)"
            )
            metric_text = (
                f"shoulders {metrics['shoulder_slope_deg']:+.1f} deg   "
                f"torso {torso_text}   "
                f"neck {metrics['neck_tilt_deg']:+.1f} deg   "
                f"head-depth {metrics['head_forward_world']:+.2f}"
            )
            cv2.putText(
                frame,
                metric_text,
                (18, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
            )
            issue_text = ", ".join(issues[:3]) if issues else "no obvious front-view issue"
            cv2.putText(
                frame,
                issue_text,
                (18, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                model_color,
                1,
            )
        elif pose_landmarks is None:
            cv2.putText(
                frame,
                "No pose - keep head and both shoulders visible",
                (18, height - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (100, 120, 235),
                2,
            )

        cv2.imshow("Quell - Posture Data Collection", frame)

    capture.release()
    detector.close()
    cv2.destroyAllWindows()

    if not save_requested or not rows:
        print("No posture data saved.")
        return

    exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    with open(args.output, "a", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(feature_names() + ["session", "view", "label"])
        writer.writerows(rows)
    print(f"Appended {len(rows)} samples to {args.output}")
    print(f"  Good: {counts[GOOD]}  Bad: {counts[BAD]}")
    print("Run python train_posture.py after collecting several sessions.")


if __name__ == "__main__":
    main()
