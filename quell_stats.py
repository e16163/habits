"""Local session statistics and behavior-change interventions for Quell."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_VERSION = 1


@dataclass
class SessionStats:
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    active_seconds: float = 0.0
    touch_count: int = 0
    touch_seconds: float = 0.0
    posture_visible_seconds: float = 0.0
    posture_bad_seconds: float = 0.0
    posture_corrections: int = 0
    interventions: int = 0
    successful_interventions: int = 0
    recovery_seconds: list[float] = field(default_factory=list)
    best_touch_free_seconds: float = 0.0

    _touching: bool = field(default=False, repr=False)
    _posture_bad: bool = field(default=False, repr=False)
    _clear_streak_started: float = field(default=0.0, repr=False)
    _bad_episode_started: float | None = field(default=None, repr=False)

    def update(
        self,
        delta_seconds: float,
        touching: bool,
        posture_bad: bool,
        posture_visible: bool,
    ) -> dict[str, Any]:
        delta = max(0.0, min(float(delta_seconds), 0.25))
        self.active_seconds += delta
        events: dict[str, Any] = {}

        if touching:
            self.touch_seconds += delta
        if touching and not self._touching:
            self.touch_count += 1
            self.best_touch_free_seconds = max(
                self.best_touch_free_seconds,
                self.active_seconds - self._clear_streak_started,
            )
            events["touch_started"] = True
        elif not touching and self._touching:
            self._clear_streak_started = self.active_seconds
            events["touch_ended"] = True
        self._touching = touching

        if posture_visible:
            self.posture_visible_seconds += delta
            if posture_bad:
                self.posture_bad_seconds += delta
        if posture_visible and posture_bad and not self._posture_bad:
            self._bad_episode_started = self.active_seconds
            events["posture_bad_started"] = True
        elif self._posture_bad and posture_visible and not posture_bad:
            if self._bad_episode_started is not None:
                recovery = self.active_seconds - self._bad_episode_started
                events["posture_recovered"] = recovery
                self.posture_corrections += 1
            self._bad_episode_started = None
        elif not posture_visible and self._posture_bad:
            # Losing tracking is not evidence that posture was corrected.
            self._bad_episode_started = None
            events["posture_lost"] = True
        self._posture_bad = posture_bad if posture_visible else False
        return events

    @property
    def current_touch_free_seconds(self) -> float:
        if self._touching:
            return 0.0
        return max(0.0, self.active_seconds - self._clear_streak_started)

    @property
    def current_bad_posture_seconds(self) -> float:
        if not self._posture_bad or self._bad_episode_started is None:
            return 0.0
        return max(0.0, self.active_seconds - self._bad_episode_started)

    @property
    def touch_rate_per_hour(self) -> float:
        if self.active_seconds <= 1.0:
            return 0.0
        return self.touch_count * 3600.0 / self.active_seconds

    @property
    def good_posture_percent(self) -> float | None:
        if self.posture_visible_seconds < 1.0:
            return None
        good = 1.0 - self.posture_bad_seconds / self.posture_visible_seconds
        return max(0.0, min(100.0, good * 100.0))

    @property
    def intervention_success_percent(self) -> float | None:
        if self.interventions == 0:
            return None
        return 100.0 * self.successful_interventions / self.interventions

    @property
    def average_recovery_seconds(self) -> float | None:
        if not self.recovery_seconds:
            return None
        return sum(self.recovery_seconds) / len(self.recovery_seconds)

    def register_intervention(self) -> None:
        self.interventions += 1

    def register_success(self, recovery_seconds: float) -> None:
        self.successful_interventions += 1
        self.recovery_seconds.append(max(0.0, float(recovery_seconds)))

    def export(self) -> dict[str, Any]:
        self.best_touch_free_seconds = max(
            self.best_touch_free_seconds, self.current_touch_free_seconds
        )
        data = asdict(self)
        for key in [
            "_touching",
            "_posture_bad",
            "_clear_streak_started",
            "_bad_episode_started",
        ]:
            data.pop(key, None)
        data["touch_rate_per_hour"] = self.touch_rate_per_hour
        data["good_posture_percent"] = self.good_posture_percent
        data["intervention_success_percent"] = self.intervention_success_percent
        data["average_recovery_seconds"] = self.average_recovery_seconds
        return data


class HistoryStore:
    def __init__(self, path: str | Path = "quell_history.json"):
        self.path = Path(path)
        self.sessions: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") == HISTORY_VERSION:
                self.sessions = list(payload.get("sessions", []))[-100:]
        except (OSError, ValueError, TypeError):
            self.sessions = []

    def add(self, stats: SessionStats) -> None:
        if stats.active_seconds < 10.0:
            return
        self.sessions.append(stats.export())
        self.sessions = self.sessions[-100:]
        payload = {"version": HISTORY_VERSION, "sessions": self.sessions}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def _qualified(self) -> list[dict[str, Any]]:
        return [s for s in self.sessions if float(s.get("active_seconds", 0)) >= 60]

    @property
    def baseline_touch_rate(self) -> float | None:
        sessions = self._qualified()[:3]
        rates = [float(s.get("touch_rate_per_hour", 0)) for s in sessions]
        return sum(rates) / len(rates) if rates and any(rate > 0 for rate in rates) else None

    @property
    def recent_touch_rate(self) -> float | None:
        rates = [
            float(s.get("touch_rate_per_hour", 0))
            for s in self._qualified()[-5:]
        ]
        return sum(rates) / len(rates) if rates else None

    @property
    def recent_good_posture(self) -> float | None:
        values = [
            float(s["good_posture_percent"])
            for s in self._qualified()[-5:]
            if s.get("good_posture_percent") is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def historical_best_streak(self) -> float:
        return max(
            [float(s.get("best_touch_free_seconds", 0)) for s in self.sessions]
            or [0.0]
        )

    def reduction_percent(self, current_rate: float) -> float | None:
        baseline = self.baseline_touch_rate
        if baseline is None or baseline <= 0:
            return None
        return max(-999.0, min(100.0, (1.0 - current_rate / baseline) * 100.0))

    def touch_rate_series(self, current_rate: float | None = None) -> list[float]:
        rates = [
            float(s.get("touch_rate_per_hour", 0))
            for s in self._qualified()[-9:]
        ]
        if current_rate is not None:
            rates.append(float(current_rate))
        return rates[-10:]


@dataclass
class Intervention:
    kind: str
    eyebrow: str
    title: str
    body: str
    action: str
    created_at: float
    expires_at: float
    tone: str


class InterventionEngine:
    def __init__(self):
        self.active: Intervention | None = None
        self._last_touch_prompt = -999.0
        self._posture_prompted = False
        self._pending_touch_at: float | None = None
        self._pending_posture_at: float | None = None
        self._break_prompted = False

    def _show(
        self,
        now: float,
        kind: str,
        eyebrow: str,
        title: str,
        body: str,
        action: str,
        tone: str,
        duration: float = 8.0,
    ) -> None:
        self.active = Intervention(
            kind=kind,
            eyebrow=eyebrow,
            title=title,
            body=body,
            action=action,
            created_at=now,
            expires_at=now + duration,
            tone=tone,
        )

    def update(
        self,
        now: float,
        stats: SessionStats,
        events: dict[str, Any],
    ) -> Intervention | None:
        if self.active and now >= self.active.expires_at:
            self.active = None

        if events.get("touch_started") and now - self._last_touch_prompt >= 10.0:
            self._last_touch_prompt = now
            self._pending_touch_at = stats.active_seconds
            stats.register_intervention()
            self._show(
                now,
                "hair",
                "HABIT INTERRUPT",
                "Hands back to neutral",
                "Break the loop before it becomes automatic.",
                "Exhale slowly, then place both hands flat on the desk.",
                "rose",
            )

        if events.get("touch_ended") and self._pending_touch_at is not None:
            recovery = stats.active_seconds - self._pending_touch_at
            if recovery <= 12.0:
                stats.register_success(recovery)
                self._show(
                    now,
                    "success",
                    "RESET COMPLETE",
                    "Pattern interrupted",
                    f"You reset in {recovery:.1f} seconds.",
                    "Return attention to the task, not the urge.",
                    "mint",
                    duration=4.5,
                )
            self._pending_touch_at = None

        if stats.current_bad_posture_seconds >= 10.0 and not self._posture_prompted:
            self._posture_prompted = True
            self._pending_posture_at = stats.active_seconds
            stats.register_intervention()
            self._show(
                now,
                "posture",
                "ALIGNMENT RESET",
                "Rebuild from the chair up",
                "A small reset is more sustainable than holding a rigid pose.",
                "Feet supported · soften shoulders · bring the screen to you.",
                "amber",
                duration=10.0,
            )

        recovery = events.get("posture_recovered")
        if recovery is not None:
            self._posture_prompted = False
            if self._pending_posture_at is not None:
                prompted_recovery = stats.active_seconds - self._pending_posture_at
                stats.register_success(prompted_recovery)
                self._show(
                    now,
                    "success",
                    "ALIGNMENT RESTORED",
                    "Nice correction",
                    f"Recovered in {prompted_recovery:.1f} seconds after the prompt.",
                    "Let the new position stay relaxed.",
                    "mint",
                    duration=4.5,
                )
            self._pending_posture_at = None

        if events.get("posture_lost"):
            self._posture_prompted = False
            self._pending_posture_at = None

        if stats.active_seconds >= 25 * 60 and not self._break_prompted:
            self._break_prompted = True
            self._show(
                now,
                "break",
                "MICROBREAK",
                "Move away from the screen",
                "Sustained stillness is its own posture risk.",
                "Stand, look far away, and move for sixty seconds.",
                "blue",
                duration=12.0,
            )

        return self.active
