class BehaviorTracker:
    """
    Tracks a single behavior over time and returns its risk contribution (0..max_pts).

    While active:
      - score ramps linearly from 0 to max_pts over ramp_secs (duration effect)
      - a frequency bonus is added for every episode start in the last 5 minutes

    While inactive:
      - active episode timer resets to zero (next episode starts fresh)
      - score decays at decay_per_sec until it reaches 0
    """

    FREQUENCY_WINDOW = 300.0  # seconds (5 minutes)

    def __init__(self, max_pts, ramp_secs, decay_per_sec, freq_bonus=3.0):
        self.max_pts = max_pts
        self.ramp_secs = ramp_secs
        self.decay_per_sec = decay_per_sec
        self.freq_bonus = freq_bonus    # extra points per recent episode start
        self._active_secs = 0.0
        self._score = 0.0
        self._was_active = False
        self._recent_starts = []        # timestamps of recent episode starts

    def update(self, active: bool, dt: float, now: float) -> float:
        if active and not self._was_active:
            self._recent_starts.append(now)
        self._was_active = active

        cutoff = now - self.FREQUENCY_WINDOW
        self._recent_starts = [t for t in self._recent_starts if t > cutoff]

        if active:
            self._active_secs += dt
            duration_score = self.max_pts * min(1.0, self._active_secs / self.ramp_secs)
            freq_score = min(self.max_pts * 0.5, self.freq_bonus * len(self._recent_starts))
            self._score = min(self.max_pts, duration_score + freq_score)
        else:
            self._active_secs = 0.0
            self._score = max(0.0, self._score - self.decay_per_sec * dt)

        return self._score

    @property
    def active_secs(self) -> float:
        return self._active_secs if self._was_active else 0.0

    @property
    def recent_count(self) -> int:
        return len(self._recent_starts)


class RiskTracker:
    """
    Aggregates per-behavior risk into a single safety score (100=attentive, 0=critical).

    Behavior parameters:
      max_pts       — maximum risk points this behavior can contribute
      ramp_secs     — seconds of continuous activity to reach max_pts
      decay_per_sec — points lost per second after the behavior stops
      freq_bonus    — extra points per episode start in the last 5 minutes
    """

    _CONFIG = {
        #                          max_pts  ramp_s  decay/s  freq_bonus
        "phone_call":        dict(max_pts=40, ramp_secs=30,  decay_per_sec=0.30, freq_bonus=5.0),
        "phone_use":         dict(max_pts=35, ramp_secs=30,  decay_per_sec=0.30, freq_bonus=5.0),
        "head_turn":         dict(max_pts=25, ramp_secs=20,  decay_per_sec=0.80, freq_bonus=3.0),
        "eye_rub":           dict(max_pts=20, ramp_secs=15,  decay_per_sec=1.00, freq_bonus=3.0),
        "one_hand_raised":   dict(max_pts=15, ramp_secs=60,  decay_per_sec=0.10, freq_bonus=2.0),
        "two_hands_raised":  dict(max_pts=40, ramp_secs=60,  decay_per_sec=0.10, freq_bonus=5.0),
        "hands_busy_object": dict(max_pts=25, ramp_secs=30,  decay_per_sec=0.30, freq_bonus=3.0),
    }

    def __init__(self):
        self._trackers = {k: BehaviorTracker(**v) for k, v in self._CONFIG.items()}
        self._last_time = None

    def update(self, events: dict, now: float) -> float:
        """
        events — dict mapping behavior key -> bool (is the behavior currently active?)
        Returns safety score 0-100 (100 = no risk, 0 = critical).
        """
        if self._last_time is None:
            self._last_time = now
            return 100.0

        dt = now - self._last_time
        self._last_time = now

        risk = sum(
            t.update(events.get(k, False), dt, now)
            for k, t in self._trackers.items()
        )
        return max(0.0, 100.0 - min(100.0, risk))

    def debug_info(self) -> dict:
        """Returns per-behavior score, active duration and recent event count."""
        return {
            k: {
                "active_secs": round(t.active_secs, 1),
                "recent_count": t.recent_count,
            }
            for k, t in self._trackers.items()
        }
