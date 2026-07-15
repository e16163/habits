"""Run the live posture monitor.

The trained personal classifier is preferred.  Before training, the monitor
uses conservative geometric guidance so collection and camera setup can still
be tested.
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp

from posture_features import (
    build_posture_feature_vector,
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
    smoother_for_bundle,
)


MIN_QUALITY = 0.55


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="OpenCV camera index (1 on this setup; use 0 for OBS Virtual Camera)",
    )
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = create_pose_detector()
    try:
        model, bundle = load_posture_model()
    except Exception as error:
        raise RuntimeError(f"Could not load posture model: {error}") from error
    mode = "personal model" if model is not None else "geometry preview"
    print(f"Posture monitor: {mode}. Q to quit.")

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1.0)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Try --camera 1 for the physical "
            "camera or --camera 0 for OBS Virtual Camera."
        )

    smoother = smoother_for_bundle(bundle)
    bad_started = None
    total_bad_seconds = 0.0
    previous_time = time.monotonic()

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            continue
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        now = time.monotonic()
        delta = now - previous_time
        previous_time = now

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = detector.detect(image)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

        quality = pose_quality(landmarks) if landmarks is not None else 0.0
        probability = None
        assessment = None
        metrics: dict[str, float] = {}
        issues: list[str] = []
        if landmarks is not None:
            draw_upper_body_pose(frame, landmarks)
            if quality >= MIN_QUALITY:
                features = build_posture_feature_vector(landmarks, world)
                metrics = posture_metrics(features)
                if model is not None:
                    assessment = assess_posture(model, bundle, features)
                    probability = assessment["probability"]
                    issues = posture_issues(metrics, assessment["good_profile"])
                    if assessment["is_bad"] and not issues:
                        issues = ["combined posture pattern differs from baseline"]
                else:
                    probability = heuristic_bad_probability(metrics)
                    issues = posture_issues(metrics)

        is_bad = smoother.update(probability, now)
        if probability is not None and is_bad:
            total_bad_seconds += delta
            if bad_started is None:
                bad_started = now
        else:
            bad_started = None

        height, width = frame.shape[:2]
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (width, 102), (18, 18, 18), -1)
        cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)

        if probability is None:
            title = "POSITION YOUR UPPER BODY"
            subtitle = "Keep your head and both shoulders visible"
            color = (90, 150, 245)
        elif assessment and assessment["is_unfamiliar"]:
            title = "UNFAMILIAR POSTURE"
            subtitle = "outside training range - collect this pose if it is normal"
            color = (70, 170, 245)
        elif is_bad:
            title = "RESET YOUR POSTURE"
            subtitle = ", ".join(issues[:3]) or "return to your trained good posture"
            color = (70, 90, 245)
        else:
            title = "POSTURE GOOD"
            subtitle = "personal model active" if model is not None else "geometry preview - train for personalization"
            color = (95, 225, 115)

        cv2.circle(frame, (24, 30), 7, color, -1)
        cv2.putText(frame, title, (43, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)
        cv2.putText(
            frame,
            subtitle,
            (20, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )
        if metrics:
            panel = frame.copy()
            cv2.rectangle(panel, (width - 330, 118), (width - 18, 282), (18, 18, 18), -1)
            cv2.addWeighted(panel, 0.76, frame, 0.24, 0, frame)
            cv2.putText(
                frame,
                "LIVE BIOMECHANICS",
                (width - 312, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (225, 225, 225),
                1,
            )
            readings = [
                ("shoulder slope", metrics["shoulder_slope_deg"], "deg"),
                (
                    "torso lean",
                    metrics["torso_lean_deg"],
                    "deg" if metrics["hip_visibility"] >= 0.35 else "hidden",
                ),
                ("neck tilt", metrics["neck_tilt_deg"], "deg"),
                ("head offset", metrics["head_offset_x"], "ratio"),
                ("torso depth", metrics["torso_forward_world"], "ratio"),
            ]
            for row, (label, value, unit) in enumerate(readings):
                y = 174 + row * 25
                cv2.putText(
                    frame,
                    label,
                    (width - 312, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (175, 175, 175),
                    1,
                )
                if unit == "deg":
                    value_text = f"{value:+.1f} deg"
                elif unit == "hidden":
                    value_text = "n/a"
                else:
                    value_text = f"{value:+.2f}"
                cv2.putText(
                    frame,
                    value_text,
                    (width - 112, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    color,
                    1,
                )
        ema = smoother.ema if smoother.ema is not None else 0.0
        cv2.putText(
            frame,
            f"bad probability {ema:.0%}   pose quality {quality:.0%}   "
            f"bad time {int(total_bad_seconds // 60):02d}:{int(total_bad_seconds % 60):02d}",
            (20, height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
        )
        cv2.imshow("Quell - Posture", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    capture.release()
    detector.close()
    cv2.destroyAllWindows()
    print(f"Bad-posture time: {total_bad_seconds:.1f} seconds")


if __name__ == "__main__":
    main()
