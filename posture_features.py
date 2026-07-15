"""Feature extraction for the upper-body posture classifier.

The feature vector intentionally combines camera-normalized image landmarks,
MediaPipe world landmarks, and interpretable angles.  Image features make the
model useful with an ordinary front-facing webcam; world features add depth
evidence when MediaPipe can estimate it reliably.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


# MediaPipe Pose landmark indices.
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_HIP, RIGHT_HIP = 23, 24

UPPER_BODY_POINTS = [
    NOSE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_HIP,
    RIGHT_HIP,
]

POINT_LABELS = {
    NOSE: "nose",
    LEFT_EAR: "left_ear",
    RIGHT_EAR: "right_ear",
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow",
    RIGHT_ELBOW: "right_elbow",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
}

ENGINEERED_NAMES = [
    "shoulder_slope_deg",
    "hip_slope_deg",
    "torso_lean_deg",
    "neck_tilt_deg",
    "head_offset_x",
    "nose_height_ratio",
    "left_ear_shoulder_distance",
    "right_ear_shoulder_distance",
    "head_width_ratio",
    "shoulder_torso_ratio",
    "shoulder_depth_rotation",
    "hip_depth_rotation",
    "head_forward_world",
    "torso_forward_world",
    "hip_visibility",
    "mean_visibility",
    "core_visibility",
]

N_FEATURES = len(UPPER_BODY_POINTS) * 3 * 2 + len(ENGINEERED_NAMES)
FEATURE_SCHEMA_VERSION = 2  # v2 supports head-and-shoulders-only webcam crops


def _xyz(landmark) -> np.ndarray:
    return np.array(
        [float(landmark.x), float(landmark.y), float(landmark.z)],
        dtype=np.float32,
    )


def _visibility(landmark) -> float:
    return float(getattr(landmark, "visibility", 1.0))


def _midpoint(points: np.ndarray, left: int, right: int) -> np.ndarray:
    return (points[left] + points[right]) / 2.0


def _distance(a: np.ndarray, b: np.ndarray, dimensions: int = 3) -> float:
    return float(np.linalg.norm(a[:dimensions] - b[:dimensions]))


def _slope_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Signed axial line slope in image coordinates, normalized to ±90°."""
    delta = b[:2] - a[:2]
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    # Left/right landmark order reverses in a mirrored selfie view. A shoulder
    # line is axial, so 180° and 0° must describe the same horizontal posture.
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def _vertical_degrees(origin: np.ndarray, upper: np.ndarray) -> float:
    """Signed deviation from upright; positive means leaning screen-right."""
    delta = upper[:2] - origin[:2]
    return math.degrees(math.atan2(float(delta[0]), float(-delta[1])))


def feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("image", "world"):
        for index in UPPER_BODY_POINTS:
            label = POINT_LABELS[index]
            names.extend(
                [f"{prefix}_{label}_x", f"{prefix}_{label}_y", f"{prefix}_{label}_z"]
            )
    names.extend(ENGINEERED_NAMES)
    return names


def pose_quality(pose_landmarks: Iterable) -> float:
    """Visibility gate for desk-camera posture scoring.

    Hips are deliberately optional: a normal seated webcam crop often ends at
    the chest. When hips are visible, the feature extractor adds torso cues;
    otherwise it masks those cues and still scores head/shoulder posture.
    """
    landmarks = list(pose_landmarks)
    if len(landmarks) <= RIGHT_HIP:
        return 0.0
    required = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
    return min(_visibility(landmarks[index]) for index in required)


def build_posture_feature_vector(
    pose_landmarks: Iterable,
    world_landmarks: Iterable | None = None,
) -> np.ndarray:
    """Return a fixed-size, translation- and scale-normalized feature vector."""
    pose_landmarks = list(pose_landmarks)
    if len(pose_landmarks) <= RIGHT_HIP:
        raise ValueError("Pose result does not contain the required upper-body landmarks")

    image = np.array([_xyz(lm) for lm in pose_landmarks], dtype=np.float32)
    shoulder_mid = _midpoint(image, LEFT_SHOULDER, RIGHT_SHOULDER)
    shoulder_width = _distance(image[LEFT_SHOULDER], image[RIGHT_SHOULDER], 2)
    image_scale = max(shoulder_width, 1e-6)
    hip_visibility = min(
        _visibility(pose_landmarks[LEFT_HIP]),
        _visibility(pose_landmarks[RIGHT_HIP]),
    )
    hips_visible = hip_visibility >= 0.35
    hip_mid = _midpoint(image, LEFT_HIP, RIGHT_HIP)
    torso_length = (
        _distance(shoulder_mid, hip_mid, 2) if hips_visible else 0.0
    )

    image_features = np.concatenate(
        [
            (image[index] - shoulder_mid) / image_scale
            if _visibility(pose_landmarks[index]) >= 0.25
            else np.zeros(3, dtype=np.float32)
            for index in UPPER_BODY_POINTS
        ]
    )

    world_features = np.zeros(len(UPPER_BODY_POINTS) * 3, dtype=np.float32)
    world = None
    world_scale = 1.0
    if world_landmarks is not None:
        world_landmarks = list(world_landmarks)
        if len(world_landmarks) > RIGHT_HIP:
            world = np.array([_xyz(lm) for lm in world_landmarks], dtype=np.float32)
            world_scale = max(
                _distance(world[LEFT_SHOULDER], world[RIGHT_SHOULDER]), 1e-6
            )
            world_shoulder_mid = _midpoint(
                world, LEFT_SHOULDER, RIGHT_SHOULDER
            )
            world_features = np.concatenate(
                [
                    (world[index] - world_shoulder_mid) / world_scale
                    if _visibility(pose_landmarks[index]) >= 0.25
                    else np.zeros(3, dtype=np.float32)
                    for index in UPPER_BODY_POINTS
                ]
            )

    ear_mid = _midpoint(image, LEFT_EAR, RIGHT_EAR)
    ears_visible = min(
        _visibility(pose_landmarks[LEFT_EAR]),
        _visibility(pose_landmarks[RIGHT_EAR]),
    ) >= 0.25
    neck_tilt = (
        _vertical_degrees(shoulder_mid, ear_mid)
        if ears_visible
        else _vertical_degrees(shoulder_mid, image[NOSE])
    )
    visibilities = [_visibility(pose_landmarks[index]) for index in UPPER_BODY_POINTS]

    head_forward_world = 0.0
    torso_forward_world = 0.0
    if world is not None:
        world_shoulder_mid = _midpoint(world, LEFT_SHOULDER, RIGHT_SHOULDER)
        head_forward_world = float(
            (world[NOSE, 2] - world_shoulder_mid[2]) / world_scale
        )
        if hips_visible:
            world_hip_mid = _midpoint(world, LEFT_HIP, RIGHT_HIP)
            torso_forward_world = float(
                (world_shoulder_mid[2] - world_hip_mid[2]) / world_scale
            )

    hip_slope = (
        _slope_degrees(image[LEFT_HIP], image[RIGHT_HIP])
        if hips_visible
        else 0.0
    )
    torso_lean = (
        _vertical_degrees(hip_mid, shoulder_mid) if hips_visible else 0.0
    )
    left_ear_shoulder = (
        _distance(image[LEFT_EAR], image[LEFT_SHOULDER], 2) / image_scale
        if ears_visible
        else 0.0
    )
    right_ear_shoulder = (
        _distance(image[RIGHT_EAR], image[RIGHT_SHOULDER], 2) / image_scale
        if ears_visible
        else 0.0
    )
    head_width_ratio = (
        _distance(image[LEFT_EAR], image[RIGHT_EAR], 2) / image_scale
        if ears_visible
        else 0.0
    )

    engineered = np.array(
        [
            _slope_degrees(image[LEFT_SHOULDER], image[RIGHT_SHOULDER]),
            hip_slope,
            torso_lean,
            neck_tilt,
            float((image[NOSE, 0] - shoulder_mid[0]) / image_scale),
            float((shoulder_mid[1] - image[NOSE, 1]) / image_scale),
            left_ear_shoulder,
            right_ear_shoulder,
            head_width_ratio,
            shoulder_width / max(torso_length, 1e-6) if hips_visible else 0.0,
            float(
                (image[RIGHT_SHOULDER, 2] - image[LEFT_SHOULDER, 2])
                / max(shoulder_width, 1e-6)
            ),
            float((image[RIGHT_HIP, 2] - image[LEFT_HIP, 2]) / image_scale)
            if hips_visible
            else 0.0,
            head_forward_world,
            torso_forward_world,
            hip_visibility,
            float(np.mean(visibilities)),
            pose_quality(pose_landmarks),
        ],
        dtype=np.float32,
    )

    vector = np.concatenate([image_features, world_features, engineered]).astype(
        np.float32
    )
    if vector.shape != (N_FEATURES,) or not np.all(np.isfinite(vector)):
        raise ValueError("Could not produce a finite posture feature vector")
    return vector


def posture_metrics(feature_vector: np.ndarray) -> dict[str, float]:
    """Return the interpretable tail of a posture feature vector by name."""
    vector = np.asarray(feature_vector).reshape(-1)
    if vector.size != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} posture features, got {vector.size}")
    values = vector[-len(ENGINEERED_NAMES) :]
    return {name: float(value) for name, value in zip(ENGINEERED_NAMES, values)}


def heuristic_bad_probability(metrics: Mapping[str, float]) -> float:
    """Conservative front-view fallback used before a personal model exists.

    These thresholds are UI guidance, not a medical diagnosis.  A trained
    personal model should be preferred because body proportions and camera
    placement vary substantially.
    """
    components = [
        min(abs(metrics["torso_lean_deg"]) / 18.0, 1.0),
        min(abs(metrics["shoulder_slope_deg"]) / 14.0, 1.0),
        min(abs(metrics["neck_tilt_deg"]) / 20.0, 1.0),
        min(abs(metrics["head_offset_x"]) / 0.35, 1.0),
        min(abs(metrics["torso_forward_world"]) / 0.45, 1.0),
    ]
    # Emphasize the two strongest cues instead of punishing normal variation
    # across every measurement.
    components.sort(reverse=True)
    score = 0.7 * components[0] + 0.3 * components[1]
    return float(np.clip(score, 0.0, 1.0))


def posture_issues(
    metrics: Mapping[str, float],
    good_profile: Mapping[str, float] | None = None,
) -> list[str]:
    """Explain the strongest posture deviations.

    When a trained good-posture profile is supplied, thresholds are measured
    relative to that person's normal alignment and camera position.
    """
    issues: list[str] = []
    baseline = good_profile or {}

    def delta(name: str) -> float:
        return metrics[name] - float(baseline.get(name, 0.0))

    hips_available = metrics.get("hip_visibility", 1.0) >= 0.35
    if hips_available and abs(delta("torso_lean_deg")) > (7 if good_profile else 10):
        issues.append("torso leaning")
    if abs(delta("shoulder_slope_deg")) > (5 if good_profile else 8):
        issues.append("shoulders uneven")
    if abs(delta("neck_tilt_deg")) > (8 if good_profile else 12):
        issues.append("head tilted")
    if abs(delta("head_offset_x")) > (0.14 if good_profile else 0.22):
        issues.append("head off-centre")
    if abs(delta("head_forward_world")) > (0.18 if good_profile else 0.30):
        issues.append("head forward/back")
    if hips_available and abs(delta("torso_forward_world")) > (0.18 if good_profile else 0.30):
        issues.append("torso slouch/depth shift")
    return issues
