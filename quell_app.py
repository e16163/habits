"""Quell unified hair-habit and posture desktop dashboard."""

from __future__ import annotations

import argparse
import os
import pickle
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

from features import (
    HEAD_ANCHORS,
    N_FEATURES as HAIR_FEATURE_COUNT,
    build_feature_vector,
)
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
from quell_stats import HistoryStore, Intervention, InterventionEngine, SessionStats


APP_DIR = Path(__file__).resolve().parent
WINDOW_TITLE = "Quell - Focus and Alignment"
CANVAS_WIDTH, CANVAS_HEIGHT = 1440, 900
INFERENCE_WIDTH, INFERENCE_HEIGHT = 640, 360

HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index")
    parser.add_argument("--history", default="quell_history.json")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render animated sample data without camera or model initialization",
    )
    parser.add_argument(
        "--screenshot",
        help="save one dashboard frame and exit (normally used with --demo)",
    )
    return parser.parse_args()


def ensure_asset(path: str, url: str, label: str) -> None:
    if not os.path.exists(path):
        print(f"Downloading {label}...")
        urllib.request.urlretrieve(url, path)
        print("  Done.")


def load_hair_model(path: str = "quell_model.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as file:
        bundle = pickle.load(file)
    model = bundle.get("pipeline") if isinstance(bundle, dict) else bundle
    expected = getattr(model, "n_features_in_", HAIR_FEATURE_COUNT)
    if expected != HAIR_FEATURE_COUNT:
        raise ValueError(
            f"{path} expects {expected} features; retrain it with the current train.py"
        )
    return model


def probability_for_label(model, features: np.ndarray, label: int) -> float:
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(label)])


class SignalLatch:
    """EMA signal with asymmetric thresholds and short dwell times."""

    def __init__(
        self,
        alpha: float = 0.32,
        on_threshold: float = 0.66,
        off_threshold: float = 0.38,
        on_hold: float = 0.14,
        off_hold: float = 0.20,
    ):
        self.alpha = alpha
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.on_hold = on_hold
        self.off_hold = off_hold
        self.ema = 0.0
        self.active = False
        self._candidate_since: float | None = None

    def update(self, probability: float | None, now: float) -> bool:
        value = 0.0 if probability is None else float(probability)
        self.ema = self.alpha * value + (1.0 - self.alpha) * self.ema
        wants_change = (
            self.ema >= self.on_threshold
            if not self.active
            else self.ema <= self.off_threshold
        )
        if not wants_change:
            self._candidate_since = None
            return self.active
        if self._candidate_since is None:
            self._candidate_since = now
            return self.active
        hold = self.on_hold if not self.active else self.off_hold
        if now - self._candidate_since >= hold:
            self.active = not self.active
            self._candidate_since = None
        return self.active


def format_duration(seconds: float, compact: bool = False) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    if compact and minutes == 0:
        return f"{secs}s"
    return f"{minutes:02d}:{secs:02d}"


def cover_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        frame,
        (int(source_width * scale), int(source_height * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y = max(0, (resized.shape[0] - height) // 2)
    x = max(0, (resized.shape[1] - width) // 2)
    return resized[y : y + height, x : x + width]


def draw_hand_overlay(frame: np.ndarray, hand_landmarks, color) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for landmarks in hand_landmarks:
        for start, end in HAND_CONNECTIONS:
            a, b = landmarks[start], landmarks[end]
            cv2.line(
                overlay,
                (int(a.x * width), int(a.y * height)),
                (int(b.x * width), int(b.y * height)),
                color,
                2,
                cv2.LINE_AA,
            )
        for index in [0, 4, 8, 12, 16, 20]:
            landmark = landmarks[index]
            cv2.circle(
                overlay,
                (int(landmark.x * width), int(landmark.y * height)),
                3,
                color,
                -1,
                cv2.LINE_AA,
            )
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)


@dataclass
class RuntimeState:
    hair_model_loaded: bool = False
    posture_model_loaded: bool = False
    hand_visible: bool = False
    face_visible: bool = False
    pose_visible: bool = False
    touching: bool = False
    hair_probability: float | None = None
    posture_bad: bool = False
    posture_probability: float | None = None
    posture_unfamiliar: bool = False
    posture_metrics: dict[str, float] | None = None
    posture_issues: list[str] | None = None
    fps: float = 0.0
    camera_index: int = 1


class DashboardRenderer:
    BG = (8, 12, 20)
    SURFACE = (16, 22, 34)
    SURFACE_2 = (22, 29, 43)
    BORDER = (39, 49, 68)
    TEXT = (239, 244, 252)
    MUTED = (145, 157, 181)
    SUBTLE = (91, 104, 129)
    MINT = (73, 224, 172)
    ROSE = (255, 104, 139)
    AMBER = (255, 189, 93)
    BLUE = (95, 148, 255)
    VIOLET = (163, 128, 255)

    def __init__(self):
        self.fonts = {
            (size, bold): self._load_font(size, bold)
            for size, bold in [
                (11, False),
                (12, True),
                (13, False),
                (14, True),
                (16, False),
                (18, True),
                (22, True),
                (28, True),
                (38, True),
                (48, True),
            ]
        }
        self.background = self._create_background()

    @classmethod
    def _create_background(cls) -> Image.Image:
        """Pre-render a quiet ambient gradient so per-frame UI stays fast."""
        height, width = CANVAS_HEIGHT, CANVAS_WIDTH
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        top = np.array([7, 11, 19], dtype=np.float32)[None, None, :]
        bottom = np.array([10, 16, 27], dtype=np.float32)[None, None, :]
        pixels = np.broadcast_to(top * (1 - y) + bottom * y, (height, width, 3)).copy()
        yy, xx = np.mgrid[0:height, 0:width]
        glows = [
            (170, 10, np.array(cls.MINT), 0.052, 360),
            (1290, 300, np.array(cls.BLUE), 0.070, 470),
            (760, 930, np.array(cls.VIOLET), 0.035, 520),
        ]
        for cx, cy, color, strength, radius in glows:
            distance = ((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)
            alpha = (np.exp(-distance) * strength)[..., None]
            pixels = pixels * (1 - alpha) + color[None, None, :] * alpha
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _load_font(size: int, bold: bool):
        candidates = [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size=size, index=1 if bold and path.endswith(".ttc") else 0)
            except (OSError, ValueError):
                continue
        return ImageFont.load_default()

    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = self._load_font(size, bold)
        return self.fonts[key]

    def text(
        self,
        draw: ImageDraw.ImageDraw,
        xy,
        value: str,
        size: int,
        color=None,
        bold: bool = False,
        anchor: str | None = None,
    ) -> None:
        draw.text(
            xy,
            value,
            font=self.font(size, bold),
            fill=color or self.TEXT,
            anchor=anchor,
        )

    def card(self, draw, box, radius: int = 22, fill=None, outline=None):
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(
            (x1 + 2, y1 + 5, x2 + 2, y2 + 5),
            radius=radius,
            fill=(0, 0, 0, 72),
        )
        draw.rounded_rectangle(
            box,
            radius=radius,
            fill=fill or (*self.SURFACE, 238),
            outline=outline or self.BORDER,
            width=1,
        )

    def logo_mark(self, draw, x: int, y: int, size: int) -> None:
        """Draw the vector Q/reset mark used by the SVG brand assets."""
        pad = max(3, int(size * 0.15))
        box = (x + pad, y + pad, x + size - pad, y + size - pad)
        width = max(4, int(size * 0.13))
        draw.ellipse(
            (x, y, x + size, y + size),
            fill=(*self.MINT, 20),
        )
        draw.arc(box, start=-44, end=236, fill=self.MINT, width=width)
        draw.arc(box, start=236, end=300, fill=self.BLUE, width=width)
        tail_start = (x + int(size * 0.64), y + int(size * 0.66))
        tail_end = (x + int(size * 0.88), y + int(size * 0.90))
        draw.line((tail_start, tail_end), fill=self.BLUE, width=width)
        inner_y = y + int(size * 0.49)
        draw.arc(
            (x + int(size * 0.31), inner_y - 5, x + int(size * 0.72), inner_y + 14),
            start=195,
            end=345,
            fill=self.TEXT,
            width=max(2, width // 3),
        )
        dot = max(2, int(size * 0.045))
        dot_x, dot_y = x + int(size * 0.22), y + int(size * 0.70)
        draw.ellipse((dot_x - dot, dot_y - dot, dot_x + dot, dot_y + dot), fill=self.MINT)

    def pill(self, draw, xy, label: str, color, active: bool = True):
        x, y = xy
        font = self.font(11, True)
        bbox = draw.textbbox((0, 0), label, font=font)
        width = bbox[2] - bbox[0] + 30
        fill = (*color, 42) if active else (31, 39, 54, 255)
        draw.rounded_rectangle((x, y, x + width, y + 28), radius=14, fill=fill)
        draw.ellipse((x + 10, y + 10, x + 18, y + 18), fill=color if active else self.SUBTLE)
        draw.text((x + 23, y + 7), label, font=font, fill=self.TEXT if active else self.MUTED)
        return width

    def progress(self, draw, box, value: float, color):
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(box, radius=(y2 - y1) // 2, fill=(37, 45, 61))
        ratio = max(0.0, min(1.0, value))
        if ratio > 0:
            draw.rounded_rectangle(
                (x1, y1, x1 + max(y2 - y1, (x2 - x1) * ratio), y2),
                radius=(y2 - y1) // 2,
                fill=color,
            )

    def sparkline(self, draw, box, values: list[float], color):
        x1, y1, x2, y2 = box
        if len(values) < 2:
            draw.line((x1, (y1 + y2) / 2, x2, (y1 + y2) / 2), fill=self.BORDER, width=2)
            return
        low, high = min(values), max(values)
        span = max(1e-6, high - low)
        points = []
        for index, value in enumerate(values):
            x = x1 + index * (x2 - x1) / (len(values) - 1)
            y = y2 - 5 - (value - low) / span * (y2 - y1 - 10)
            points.append((x, y))
        draw.line(points, fill=color, width=3, joint="curve")
        for x, y in points[-3:]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    def metric(self, draw, x, y, value: str, label: str, color=None):
        self.text(draw, (x, y), value, 22, color or self.TEXT, bold=True)
        self.text(draw, (x, y + 30), label.upper(), 11, self.MUTED, bold=True)

    def render(
        self,
        camera_bgr: np.ndarray,
        state: RuntimeState,
        stats: SessionStats,
        history: HistoryStore,
        intervention: Intervention | None,
        paused: bool,
    ) -> np.ndarray:
        image = self.background.copy()
        draw = ImageDraw.Draw(image, "RGBA")

        # Header
        draw.rectangle((0, 0, CANVAS_WIDTH, 71), fill=(7, 11, 19, 212))
        draw.line((0, 71, CANVAS_WIDTH, 71), fill=(48, 61, 84, 180), width=1)
        self.logo_mark(draw, 22, 10, 48)
        self.text(draw, (82, 14), "Quell", 22, bold=True)
        self.text(draw, (83, 42), "AWARENESS, QUIETLY.", 10, self.MUTED, bold=True)

        chip_x = 300
        chip_x += self.pill(
            draw,
            (chip_x, 22),
            "HAIR MODEL" if state.hair_model_loaded else "TRAIN HAIR MODEL",
            self.MINT if state.hair_model_loaded else self.AMBER,
            state.hair_model_loaded,
        ) + 10
        self.pill(
            draw,
            (chip_x, 22),
            "POSTURE MODEL" if state.posture_model_loaded else "POSTURE GEOMETRY",
            self.BLUE if state.posture_model_loaded else self.VIOLET,
            True,
        )
        draw.ellipse((1221, 31, 1231, 41), fill=self.AMBER if paused else self.MINT)
        self.text(draw, (1242, 25), "PAUSED" if paused else "LIVE", 12, bold=True)
        self.text(draw, (1408, 20), format_duration(stats.active_seconds), 18, bold=True, anchor="ra")
        self.text(draw, (1408, 43), f"CAM {state.camera_index}  ·  {state.fps:.0f} FPS", 11, self.MUTED, bold=True, anchor="ra")

        # Camera card
        camera_box = (24, 92, 924, 692)
        draw.rounded_rectangle((22, 90, 926, 694), radius=29, fill=(0, 0, 0, 90))
        camera = cover_frame(camera_bgr, 900, 600)
        camera_rgb = Image.fromarray(cv2.cvtColor(camera, cv2.COLOR_BGR2RGB))
        mask = Image.new("L", (900, 600), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, 899, 599), radius=26, fill=255)
        image.paste(camera_rgb, (24, 92), mask)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle(camera_box, radius=26, outline=(68, 86, 117), width=2)
        draw.line((76, 92, 210, 92), fill=self.MINT, width=3)
        draw.line((738, 692, 872, 692), fill=self.BLUE, width=3)

        # Camera glass overlays
        draw.rounded_rectangle((46, 114, 164, 146), radius=16, fill=(8, 12, 20, 185))
        draw.ellipse((60, 125, 68, 133), fill=self.MINT if not paused else self.AMBER)
        self.text(draw, (76, 122), "CAMERA LIVE" if not paused else "PAUSED", 11, bold=True)

        status_y = 629
        hair_color = self.ROSE if state.touching else self.MINT
        posture_color = self.AMBER if state.posture_bad else self.BLUE
        draw.rounded_rectangle((46, status_y, 900, 674), radius=18, fill=(8, 12, 20, 210))
        self.text(
            draw,
            (66, status_y + 13),
            "TOUCH DETECTED" if state.touching else ("HANDS CLEAR" if state.hair_model_loaded else "HAIR MODEL NEEDED"),
            12,
            hair_color if state.hair_model_loaded else self.MUTED,
            bold=True,
        )
        draw.line((256, status_y + 10, 256, status_y + 35), fill=self.BORDER, width=1)
        posture_label = (
            "POSE NOT VISIBLE"
            if not state.pose_visible
            else "UNFAMILIAR POSE"
            if state.posture_unfamiliar
            else "ALIGNMENT RESET"
            if state.posture_bad
            else "POSTURE ALIGNED"
        )
        self.text(draw, (276, status_y + 13), posture_label, 12, posture_color, bold=True)
        if state.posture_issues and state.posture_bad:
            self.text(draw, (486, status_y + 13), " · ".join(state.posture_issues[:2]), 11, self.MUTED)

        # Under-camera insight strip
        self.card(draw, (24, 708, 924, 770), radius=20, fill=(14, 21, 33, 228))
        reduction = history.reduction_percent(stats.touch_rate_per_hour) if stats.active_seconds >= 60 else None
        best = max(stats.best_touch_free_seconds, stats.current_touch_free_seconds, history.historical_best_streak)
        self.text(draw, (46, 724), "REDUCTION SIGNAL", 11, self.MUTED, bold=True)
        reduction_text = "Building baseline" if reduction is None else f"{reduction:+.0f}% vs baseline"
        reduction_color = self.MUTED if reduction is None else self.MINT if reduction >= 0 else self.ROSE
        self.text(draw, (46, 744), reduction_text, 16, reduction_color, bold=True)
        self.text(draw, (312, 724), "BEST TOUCH-FREE", 11, self.MUTED, bold=True)
        self.text(draw, (312, 744), format_duration(best), 16, self.TEXT, bold=True)
        self.text(draw, (552, 724), "RECENT POSTURE", 11, self.MUTED, bold=True)
        recent_posture = history.recent_good_posture
        self.text(draw, (552, 744), "No history" if recent_posture is None else f"{recent_posture:.0f}% good", 16, self.BLUE, bold=True)
        self.text(draw, (770, 724), "CONTROLS", 11, self.MUTED, bold=True)
        self.text(draw, (770, 744), "P pause  ·  D detail  ·  Q quit", 11, self.TEXT)

        # Right: session overview
        right_x, right_w = 944, 472
        self.card(draw, (right_x, 92, right_x + right_w, 246), fill=(15, 21, 33, 236))
        draw.ellipse((right_x + 22, 116, right_x + 30, 124), fill=self.VIOLET)
        self.text(draw, (right_x + 40, 112), "SESSION PULSE", 12, self.MUTED, bold=True)
        success = stats.intervention_success_percent
        good = stats.good_posture_percent
        self.metric(draw, right_x + 22, 142, str(stats.touch_count), "touches", self.ROSE)
        self.metric(draw, right_x + 133, 142, "--" if good is None else f"{good:.0f}%", "posture good", self.BLUE)
        self.metric(draw, right_x + 278, 142, str(stats.posture_corrections), "corrections", self.AMBER)
        self.metric(draw, right_x + 392, 142, "--" if success is None else f"{success:.0f}%", "prompt success", self.MINT)

        # Hair card
        self.card(draw, (right_x, 262, right_x + right_w, 474), fill=(13, 25, 32, 236))
        draw.ellipse((right_x + 22, 286, right_x + 30, 294), fill=self.MINT)
        self.text(draw, (right_x + 40, 282), "HAIR HABIT", 12, self.MUTED, bold=True)
        hair_signal = state.hair_probability or 0.0
        self.text(draw, (right_x + 450, 280), f"{hair_signal:.0%}", 14, self.ROSE if state.touching else self.MINT, bold=True, anchor="ra")
        self.progress(draw, (right_x + 22, 310, right_x + 450, 320), hair_signal, self.ROSE if state.touching else self.MINT)
        self.metric(draw, right_x + 22, 340, f"{stats.touch_rate_per_hour:.1f}", "touches / hour")
        self.metric(draw, right_x + 175, 340, format_duration(stats.current_touch_free_seconds), "current clear")
        reduction_value = history.reduction_percent(stats.touch_rate_per_hour) if stats.active_seconds >= 60 else None
        self.metric(
            draw,
            right_x + 338,
            340,
            "--" if reduction_value is None else f"{reduction_value:+.0f}%",
            "reduction",
            self.MINT if reduction_value is not None and reduction_value >= 0 else self.ROSE,
        )
        self.text(draw, (right_x + 22, 417), "TOUCH-RATE TREND", 10, self.MUTED, bold=True)
        self.sparkline(
            draw,
            (right_x + 160, 408, right_x + 450, 453),
            history.touch_rate_series(stats.touch_rate_per_hour if stats.active_seconds >= 60 else None),
            self.MINT,
        )

        # Posture card
        self.card(draw, (right_x, 490, right_x + right_w, 770), fill=(14, 21, 38, 238))
        draw.ellipse((right_x + 22, 514, right_x + 30, 522), fill=self.BLUE)
        self.text(draw, (right_x + 40, 510), "POSTURE ANALYSIS", 12, self.MUTED, bold=True)
        posture_probability = state.posture_probability
        score = None if posture_probability is None else max(0.0, min(100.0, (1.0 - posture_probability) * 100.0))
        gauge_center = (right_x + 84, 589)
        draw.arc((gauge_center[0] - 50, gauge_center[1] - 50, gauge_center[0] + 50, gauge_center[1] + 50), 140, 400, fill=self.BORDER, width=10)
        if score is not None:
            draw.arc(
                (gauge_center[0] - 50, gauge_center[1] - 50, gauge_center[0] + 50, gauge_center[1] + 50),
                140,
                140 + 260 * score / 100.0,
                fill=self.AMBER if state.posture_bad else self.BLUE,
                width=10,
            )
        self.text(draw, gauge_center, "--" if score is None else f"{score:.0f}", 28, bold=True, anchor="mm")
        self.text(draw, (gauge_center[0], gauge_center[1] + 31), "ALIGNMENT", 10, self.MUTED, bold=True, anchor="mm")

        metrics = state.posture_metrics or {}
        hip_visible = metrics.get("hip_visibility", 0) >= 0.35
        posture_values = [
            ("SHOULDERS", f"{metrics.get('shoulder_slope_deg', 0):+.1f}°" if metrics else "--"),
            ("NECK TILT", f"{metrics.get('neck_tilt_deg', 0):+.1f}°" if metrics else "--"),
            ("HEAD OFFSET", f"{metrics.get('head_offset_x', 0):+.2f}" if metrics else "--"),
            ("HEAD DEPTH", f"{metrics.get('head_forward_world', 0):+.2f}" if metrics else "--"),
            ("TORSO LEAN", f"{metrics.get('torso_lean_deg', 0):+.1f}°" if metrics and hip_visible else "n/a"),
            ("POSE QUALITY", f"{metrics.get('core_visibility', 0):.0%}" if metrics else "--"),
        ]
        for index, (label, value) in enumerate(posture_values):
            column = index % 2
            row = index // 2
            x = right_x + 160 + column * 150
            y = 542 + row * 58
            self.text(draw, (x, y), label, 10, self.MUTED, bold=True)
            self.text(draw, (x, y + 20), value, 16, self.TEXT, bold=True)
        good_label = "No pose yet" if good is None else f"{good:.0f}% of visible time aligned"
        self.text(draw, (right_x + 22, 735), good_label, 12, self.BLUE, bold=True)

        # Intervention / coaching strip
        banner_box = (24, 790, 1416, 876)
        tone_colors = {
            "rose": self.ROSE,
            "amber": self.AMBER,
            "mint": self.MINT,
            "blue": self.BLUE,
        }
        if intervention:
            accent = tone_colors.get(intervention.tone, self.BLUE)
            self.card(draw, banner_box, radius=22, fill=(18, 26, 39, 242), outline=accent)
            draw.rounded_rectangle((24, 790, 32, 876), radius=4, fill=accent)
            self.text(draw, (52, 806), intervention.eyebrow, 11, accent, bold=True)
            self.text(draw, (52, 829), intervention.title, 22, self.TEXT, bold=True)
            self.text(draw, (380, 810), intervention.body, 13, self.MUTED)
            self.text(draw, (380, 838), intervention.action, 14, self.TEXT, bold=True)
            remaining = max(0.0, intervention.expires_at - time.monotonic())
            duration = max(0.1, intervention.expires_at - intervention.created_at)
            self.progress(draw, (1182, 839, 1388, 847), remaining / duration, accent)
        else:
            self.card(draw, banner_box, radius=22)
            self.text(draw, (52, 806), "QUIET COACHING", 11, self.MINT, bold=True)
            self.text(draw, (52, 829), "Small resets beat rigid perfection", 22, self.TEXT, bold=True)
            self.text(draw, (588, 810), "Hands neutral. Shoulders soft. Screen comes to you.", 13, self.MUTED)
            self.text(draw, (588, 838), "Quell intervenes only when a pattern persists.", 13, self.TEXT, bold=True)

        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class QuellApp:
    def __init__(self, args):
        self.args = args
        self.renderer = DashboardRenderer()
        self.stats = SessionStats()
        self.history = HistoryStore(args.history)
        self.interventions = InterventionEngine()
        self.state = RuntimeState(camera_index=args.camera)
        self.mirror = not args.no_mirror
        self.debug_overlay = args.debug_overlay
        self.paused = False
        self.fullscreen = False
        self.hair_latch = SignalLatch()
        self.posture_smoother = smoother_for_bundle(None)
        self.hair_model = None
        self.posture_model = None
        self.posture_bundle = None
        self.hand_detector = None
        self.face_detector = None
        self.pose_detector = None
        self.capture = None
        self.last_frame: np.ndarray | None = None
        self._last_time = time.monotonic()
        self._fps_ema = 0.0
        if not args.demo:
            self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        try:
            self.hair_model = load_hair_model()
        except Exception as error:
            print(f"Hair model disabled: {error}")
        self.state.hair_model_loaded = self.hair_model is not None
        if self.hair_model is not None:
            ensure_asset(HAND_MODEL_PATH, HAND_MODEL_URL, "hand landmarker")
            ensure_asset(FACE_MODEL_PATH, FACE_MODEL_URL, "face landmarker")
            self.hand_detector = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )

        try:
            self.posture_model, self.posture_bundle = load_posture_model()
        except Exception as error:
            print(f"Posture model disabled; geometry guidance remains available: {error}")
        self.state.posture_model_loaded = self.posture_model is not None
        self.posture_smoother = smoother_for_bundle(self.posture_bundle)
        self.pose_detector = create_pose_detector()

        self.capture = cv2.VideoCapture(self.args.camera)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(0.8)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.args.camera}. Try --camera 0 or "
                "--camera 1 after disconnecting unwanted virtual/Continuity cameras."
            )

    def _infer(self, frame: np.ndarray, now: float) -> np.ndarray:
        inference_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB),
        )

        hand_result = self.hand_detector.detect(mp_image) if self.hand_detector else None
        face_result = self.face_detector.detect(mp_image) if self.face_detector else None
        pose_result = self.pose_detector.detect(mp_image) if self.pose_detector else None

        hand_landmarks = hand_result.hand_landmarks if hand_result else []
        face_landmarks = (
            face_result.face_landmarks[0]
            if face_result and face_result.face_landmarks
            else None
        )
        self.state.hand_visible = bool(hand_landmarks)
        self.state.face_visible = face_landmarks is not None

        raw_hair_probability = None
        if self.hair_model is not None and hand_landmarks:
            probabilities = []
            for landmarks in hand_landmarks:
                features = build_feature_vector(landmarks, face_landmarks)
                probabilities.append(probability_for_label(self.hair_model, features, 1))
            raw_hair_probability = max(probabilities)
        self.state.touching = self.hair_latch.update(raw_hair_probability, now)
        self.state.hair_probability = self.hair_latch.ema if self.hair_model else None

        pose_landmarks = (
            pose_result.pose_landmarks[0]
            if pose_result and pose_result.pose_landmarks
            else None
        )
        world_landmarks = (
            pose_result.pose_world_landmarks[0]
            if pose_result and pose_result.pose_world_landmarks
            else None
        )
        quality = pose_quality(pose_landmarks) if pose_landmarks else 0.0
        self.state.pose_visible = quality >= 0.55
        self.state.posture_metrics = None
        self.state.posture_issues = None
        self.state.posture_unfamiliar = False
        posture_probability = None
        if self.state.pose_visible:
            features = build_posture_feature_vector(pose_landmarks, world_landmarks)
            metrics = posture_metrics(features)
            self.state.posture_metrics = metrics
            if self.posture_model is not None:
                assessment = assess_posture(
                    self.posture_model, self.posture_bundle, features
                )
                posture_probability = assessment["probability"]
                self.state.posture_unfamiliar = assessment["is_unfamiliar"]
                issues = posture_issues(metrics, assessment["good_profile"])
                if assessment["is_bad"] and not issues:
                    issues = ["combined pattern differs from baseline"]
            else:
                posture_probability = heuristic_bad_probability(metrics)
                issues = posture_issues(metrics)
            self.state.posture_issues = issues
        self.state.posture_bad = self.posture_smoother.update(
            posture_probability, now
        )
        self.state.posture_probability = (
            self.posture_smoother.ema if self.state.pose_visible else None
        )

        # Subtle, confidence-colored landmark overlay.
        if pose_landmarks is not None:
            color = (78, 151, 255) if not self.state.posture_bad else (70, 170, 255)
            overlay = frame.copy()
            draw_upper_body_pose(overlay, pose_landmarks, color=color)
            cv2.addWeighted(overlay, 0.56 if self.debug_overlay else 0.34, frame, 0.44 if self.debug_overlay else 0.66, 0, frame)
        if hand_landmarks:
            draw_hand_overlay(
                frame,
                hand_landmarks,
                (115, 110, 255) if self.state.touching else (172, 224, 73),
            )
        if self.debug_overlay and face_landmarks:
            height, width = frame.shape[:2]
            for index in HEAD_ANCHORS:
                landmark = face_landmarks[index]
                cv2.circle(
                    frame,
                    (int(landmark.x * width), int(landmark.y * height)),
                    3,
                    (255, 180, 90),
                    -1,
                    cv2.LINE_AA,
                )
        return frame

    @staticmethod
    def _demo_camera(now: float) -> np.ndarray:
        height, width = 720, 1280
        yy, xx = np.mgrid[0:height, 0:width]
        phase = (np.sin(now * 0.35) + 1.0) / 2.0
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.clip(35 + 70 * xx / width + 20 * phase, 0, 255)
        frame[:, :, 1] = np.clip(25 + 38 * yy / height, 0, 255)
        frame[:, :, 2] = np.clip(18 + 26 * (1 - xx / width), 0, 255)
        center = (width // 2, height // 2 + 40)
        cv2.circle(frame, (center[0], center[1] - 170), 72, (82, 63, 49), -1, cv2.LINE_AA)
        cv2.ellipse(frame, center, (165, 230), 0, 200, 340, (82, 63, 49), -1, cv2.LINE_AA)
        cv2.line(frame, (center[0] - 145, center[1] - 35), (center[0] + 145, center[1] - 35), (174, 140, 88), 4, cv2.LINE_AA)
        return frame

    def _prepare_demo(self, now: float) -> tuple[np.ndarray, Intervention | None]:
        self.stats.active_seconds = 18 * 60 + 42
        self.stats.touch_count = 4
        self.stats.touch_seconds = 7.8
        self.stats.posture_visible_seconds = 1040
        self.stats.posture_bad_seconds = 186
        self.stats.posture_corrections = 6
        self.stats.interventions = 7
        self.stats.successful_interventions = 6
        self.stats.recovery_seconds = [2.8, 4.1, 3.4, 5.0, 2.2, 3.7]
        self.stats.best_touch_free_seconds = 384
        self.stats._clear_streak_started = self.stats.active_seconds - 132
        self.state = RuntimeState(
            hair_model_loaded=True,
            posture_model_loaded=True,
            hand_visible=True,
            face_visible=True,
            pose_visible=True,
            touching=False,
            hair_probability=0.14,
            posture_bad=False,
            posture_probability=0.17,
            posture_metrics={
                "shoulder_slope_deg": 1.8,
                "neck_tilt_deg": -2.4,
                "head_offset_x": 0.04,
                "head_forward_world": -0.12,
                "torso_lean_deg": 2.1,
                "hip_visibility": 0.12,
                "core_visibility": 0.97,
            },
            posture_issues=[],
            fps=24.0,
            camera_index=self.args.camera,
        )
        if not self.history.sessions:
            self.history.sessions = [
                {"active_seconds": 1200, "touch_rate_per_hour": 19.0, "good_posture_percent": 72, "best_touch_free_seconds": 240},
                {"active_seconds": 1500, "touch_rate_per_hour": 16.5, "good_posture_percent": 76, "best_touch_free_seconds": 305},
                {"active_seconds": 1300, "touch_rate_per_hour": 14.0, "good_posture_percent": 79, "best_touch_free_seconds": 330},
                {"active_seconds": 1400, "touch_rate_per_hour": 12.2, "good_posture_percent": 82, "best_touch_free_seconds": 384},
            ]
        intervention = Intervention(
            kind="posture",
            eyebrow="ALIGNMENT INSIGHT",
            title="Your resets are getting faster",
            body="Six corrections this session · 3.5s average recovery.",
            action="Keep the correction small enough to stay relaxed.",
            created_at=now,
            expires_at=now + 10,
            tone="mint",
        )
        return self._demo_camera(now), intervention

    def run(self) -> None:
        if not self.args.screenshot:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, CANVAS_WIDTH, CANVAS_HEIGHT)
        save_history = not self.args.demo
        try:
            while True:
                now = time.monotonic()
                delta = now - self._last_time
                self._last_time = now
                if delta > 0:
                    instant_fps = 1.0 / delta
                    self._fps_ema = instant_fps if self._fps_ema == 0 else 0.12 * instant_fps + 0.88 * self._fps_ema
                self.state.fps = self._fps_ema

                if self.args.demo:
                    frame, intervention = self._prepare_demo(now)
                else:
                    ok, frame = self.capture.read()
                    if not ok:
                        continue
                    if self.mirror:
                        frame = cv2.flip(frame, 1)
                    if not self.paused:
                        frame = self._infer(frame, now)
                        events = self.stats.update(
                            delta,
                            self.state.touching,
                            self.state.posture_bad,
                            self.state.pose_visible,
                        )
                        intervention = self.interventions.update(now, self.stats, events)
                    else:
                        intervention = self.interventions.active
                    self.last_frame = frame.copy()

                dashboard = self.renderer.render(
                    frame,
                    self.state,
                    self.stats,
                    self.history,
                    intervention,
                    self.paused,
                )
                if self.args.screenshot:
                    cv2.imwrite(self.args.screenshot, dashboard)
                    print(f"Saved dashboard screenshot to {self.args.screenshot}")
                    break

                cv2.imshow(WINDOW_TITLE, dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("p"), ord("P")):
                    self.paused = not self.paused
                elif key in (ord("d"), ord("D")):
                    self.debug_overlay = not self.debug_overlay
                elif key in (ord("m"), ord("M")):
                    self.mirror = not self.mirror
                elif key in (ord("f"), ord("F")):
                    self.fullscreen = not self.fullscreen
                    cv2.setWindowProperty(
                        WINDOW_TITLE,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
                    )
        finally:
            if save_history:
                try:
                    self.history.add(self.stats)
                except OSError as error:
                    print(f"Could not save session history: {error}")
            self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        for detector in [self.hand_detector, self.face_detector, self.pose_detector]:
            if detector is not None:
                detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    os.chdir(APP_DIR)
    app = QuellApp(args)
    app.run()


if __name__ == "__main__":
    main()
