"""Shared MediaPipe and temporal helpers for posture applications."""

from __future__ import annotations

import os
import pickle
import time
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from posture_features import FEATURE_SCHEMA_VERSION, N_FEATURES, feature_names


POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

POSE_CONNECTIONS = [
    (7, 8),
    (7, 11),
    (8, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]


def ensure_pose_model(path: str = POSE_MODEL_PATH) -> str:
    if not os.path.exists(path):
        print("Downloading MediaPipe full pose model (~10–30 MB)…")
        urllib.request.urlretrieve(POSE_MODEL_URL, path)
        print("  Done.")
    return path


def create_pose_detector(model_path: str = POSE_MODEL_PATH):
    ensure_pose_model(model_path)
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def load_posture_model(path: str = "posture_model.pkl"):
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as file:
        bundle = pickle.load(file)
    pipeline = bundle.get("pipeline") if isinstance(bundle, dict) else bundle
    stored_names = bundle.get("feature_names") if isinstance(bundle, dict) else None
    if stored_names is not None and list(stored_names) != feature_names():
        raise ValueError(
            "posture_model.pkl uses a different feature schema; run train_posture.py again"
        )
    if getattr(pipeline, "n_features_in_", N_FEATURES) != N_FEATURES:
        raise ValueError(
            f"posture_model.pkl expects {pipeline.n_features_in_} features; expected {N_FEATURES}"
        )
    stored_schema = (
        bundle.get("feature_schema_version") if isinstance(bundle, dict) else None
    )
    if stored_schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "posture_model.pkl predates adaptive upper-body features; "
            "run train_posture.py again"
        )
    return pipeline, bundle


def bad_probability(model, feature_vector: np.ndarray) -> float:
    probabilities = model.predict_proba(feature_vector.reshape(1, -1))[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)])


def assess_posture(model, bundle, feature_vector: np.ndarray) -> dict:
    """Return probability, calibrated decision, and novelty information."""
    probability = bad_probability(model, feature_vector)
    bundle = bundle if isinstance(bundle, dict) else {}
    threshold = float(bundle.get("decision_threshold", 0.5))
    profile = bundle.get("profile", {})
    center = profile.get("feature_center")
    scale = profile.get("feature_scale")
    ood_limit = float(profile.get("ood_limit", float("inf")))
    deviation = 0.0
    if center is not None and scale is not None:
        center = np.asarray(center, dtype=np.float32)
        scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)
        deviation = float(
            np.median(np.abs((np.asarray(feature_vector) - center) / scale))
        )
    return {
        "probability": probability,
        "threshold": threshold,
        "is_bad": probability >= threshold,
        "deviation": deviation,
        "is_unfamiliar": deviation > ood_limit,
        "good_profile": profile.get("good_profile"),
    }


def smoother_for_bundle(bundle) -> "PostureSmoother":
    bundle = bundle if isinstance(bundle, dict) else {}
    threshold = float(bundle.get("decision_threshold", 0.5))
    return PostureSmoother(
        bad_threshold=min(0.88, threshold + 0.12),
        good_threshold=max(0.12, threshold - 0.12),
    )


class PostureSmoother:
    """EMA + asymmetric dwell times to prevent flickering posture alerts."""

    def __init__(
        self,
        alpha: float = 0.22,
        bad_threshold: float = 0.65,
        good_threshold: float = 0.42,
        bad_hold_seconds: float = 1.25,
        good_hold_seconds: float = 0.75,
    ):
        self.alpha = alpha
        self.bad_threshold = bad_threshold
        self.good_threshold = good_threshold
        self.bad_hold_seconds = bad_hold_seconds
        self.good_hold_seconds = good_hold_seconds
        self.ema: float | None = None
        self.is_bad = False
        self._candidate_since: float | None = None

    def reset(self) -> None:
        self.ema = None
        self.is_bad = False
        self._candidate_since = None

    def update(self, probability: float | None, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if probability is None:
            self.reset()
            return False

        self.ema = (
            probability
            if self.ema is None
            else self.alpha * probability + (1.0 - self.alpha) * self.ema
        )
        wants_change = (
            self.ema >= self.bad_threshold
            if not self.is_bad
            else self.ema <= self.good_threshold
        )
        if not wants_change:
            self._candidate_since = None
            return self.is_bad

        if self._candidate_since is None:
            self._candidate_since = now
            return self.is_bad

        hold = self.bad_hold_seconds if not self.is_bad else self.good_hold_seconds
        if now - self._candidate_since >= hold:
            self.is_bad = not self.is_bad
            self._candidate_since = None
        return self.is_bad


def draw_upper_body_pose(
    frame,
    landmarks,
    color=(90, 190, 240),
    visibility_threshold: float = 0.45,
) -> None:
    height, width = frame.shape[:2]
    for start, end in POSE_CONNECTIONS:
        a, b = landmarks[start], landmarks[end]
        if (
            float(getattr(a, "visibility", 1.0)) < visibility_threshold
            or float(getattr(b, "visibility", 1.0)) < visibility_threshold
        ):
            continue
        cv2.line(
            frame,
            (int(a.x * width), int(a.y * height)),
            (int(b.x * width), int(b.y * height)),
            color,
            2,
            cv2.LINE_AA,
        )
    for index in {point for connection in POSE_CONNECTIONS for point in connection}:
        landmark = landmarks[index]
        if float(getattr(landmark, "visibility", 1.0)) >= visibility_threshold:
            cv2.circle(
                frame,
                (int(landmark.x * width), int(landmark.y * height)),
                3,
                color,
                -1,
                cv2.LINE_AA,
            )
