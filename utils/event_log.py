"""Snapshot + CSV event logging for triggered alerts."""

import csv
import os
import time

import numpy as np
import cv2

from utils.behavior_stats import TRACKED_BEHAVIORS, BehaviorStat
from utils.daily_dir import daily_log_dir

# No score/state columns: risk is evaluated by the AvaliadorRisco container,
# which reads columns 0-1 (timestamp, alerts) — keep those two first.
LOG_HEADER = [
    "timestamp", "alerts", "snapshot",
    "phone_call_secs", "phone_call_cnt",
    "phone_use_secs",  "phone_use_cnt",
    "head_turn_secs",  "head_turn_cnt",
    "head_down_secs",  "head_down_cnt",
    "eye_rub_secs",    "eye_rub_cnt",
    "one_hand_secs",   "one_hand_cnt",
    "two_hands_secs",  "two_hands_cnt",
    "hands_obj_secs",  "hands_obj_cnt",
    "no_seatbelt_secs", "no_seatbelt_cnt",
    "smoking_secs",     "smoking_cnt",
]


class SnapshotLogger:
    """Saves a JPEG + appends a CSV row the first time a given alert
    combination appears, at most once every `cooldown_secs`.

    Debounces so a sustained alert doesn't spam disk I/O every frame.
    """

    def __init__(self, snapshot_dir: str, cooldown_secs: float = 5.0):
        self._snapshot_dir = snapshot_dir
        self._cooldown_secs = cooldown_secs
        self._last_snapshot_time = 0.0
        self._last_alerts: frozenset = frozenset()

    def maybe_log(
        self,
        frame: np.ndarray,
        alerts: list[str],
        behavior_stats: dict[str, BehaviorStat],
    ) -> None:
        current_alerts = frozenset(alerts)
        if not alerts:
            self._last_alerts = frozenset()  # reset so next occurrence is saved
            return
        if not self._should_save(current_alerts):
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = self._save_snapshot(frame, ts, alerts)
        self._append_csv_row(ts, alerts, filename, behavior_stats)

        print(f"📸 Snapshot salvo: {filename}  [{', '.join(alerts)}]")
        self._last_snapshot_time = time.time()
        self._last_alerts = current_alerts

    def _should_save(self, current_alerts: frozenset) -> bool:
        is_new_alert_set = current_alerts != self._last_alerts
        cooldown_elapsed = (time.time() - self._last_snapshot_time) >= self._cooldown_secs
        return is_new_alert_set and cooldown_elapsed

    def _save_snapshot(self, frame: np.ndarray, ts: str, alerts: list[str]) -> str:
        tag = alerts[0].replace(" ", "_")
        filename = os.path.join(daily_log_dir(self._snapshot_dir), f"{ts}_{tag}.jpg")
        cv2.imwrite(filename, frame)
        return filename

    def _append_csv_row(
        self,
        ts: str,
        alerts: list[str],
        filename: str,
        behavior_stats: dict[str, BehaviorStat],
    ) -> None:
        log_file = os.path.join(daily_log_dir(self._snapshot_dir), "eventos.csv")
        write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(LOG_HEADER)
            writer.writerow([
                ts,
                "; ".join(alerts),
                os.path.basename(filename),
                *_behavior_columns(behavior_stats),
            ])


def _behavior_columns(behavior_stats: dict[str, BehaviorStat]) -> list:
    cols = []
    for behavior in TRACKED_BEHAVIORS:
        stat = behavior_stats[behavior]
        cols += [stat.active_secs, stat.recent_count]
    return cols
