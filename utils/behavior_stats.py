"""Per-behavior facts (how long it has been active, how often it started).

No scoring here on purpose: risk evaluation (weights, combinations, levels)
lives in the separate AvaliadorRisco container, which reads eventos.csv.
This module only measures; it replaced utils/risk.py + utils/classifier.py.
"""

from dataclasses import dataclass

TRACKED_BEHAVIORS = (
    "phone_call", "phone_use", "head_turn", "head_down", "eye_rub",
    "one_hand_raised", "two_hands_raised", "hands_busy_object",
    "no_seatbelt", "smoking",
)
FREQUENCY_WINDOW_SECS = 300.0  # episode starts counted over the last 5 minutes


@dataclass(frozen=True)
class BehaviorStat:
    active_secs: float   # duration of the current episode (0 when inactive)
    recent_count: int    # episode starts within FREQUENCY_WINDOW_SECS


class EpisodeTimer:
    """Measures one behavior: current episode duration + recent episode starts.

    Example: t = EpisodeTimer(); t.update(True, dt=0.1, now=10.0); t.stat().active_secs == 0.1
    """

    def __init__(self, window_secs: float = FREQUENCY_WINDOW_SECS):
        self._window_secs = window_secs
        self._active_secs = 0.0
        self._was_active = False
        self._recent_starts: list[float] = []

    def update(self, active: bool, dt: float, now: float) -> None:
        if active and not self._was_active:
            self._recent_starts.append(now)
        self._was_active = active
        cutoff = now - self._window_secs
        self._recent_starts = [t for t in self._recent_starts if t > cutoff]
        self._active_secs = self._active_secs + dt if active else 0.0

    def stat(self) -> BehaviorStat:
        return BehaviorStat(round(self._active_secs, 1), len(self._recent_starts))


class BehaviorStatsTracker:
    """Keeps one EpisodeTimer per behavior in TRACKED_BEHAVIORS.

    Example: tracker.update({"no_seatbelt": True}, now=time.time()); tracker.stats()["no_seatbelt"]
    """

    def __init__(self):
        self._timers = {k: EpisodeTimer() for k in TRACKED_BEHAVIORS}
        self._last_time: float | None = None

    def update(self, events: dict[str, bool], now: float) -> None:
        unknown = set(events) - set(TRACKED_BEHAVIORS)
        if unknown:
            raise ValueError(f"unknown behaviors {sorted(unknown)}, expected keys of {TRACKED_BEHAVIORS}")
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        for key, timer in self._timers.items():
            timer.update(events.get(key, False), dt, now)

    def stats(self) -> dict[str, BehaviorStat]:
        return {k: t.stat() for k, t in self._timers.items()}
