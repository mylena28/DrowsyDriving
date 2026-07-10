"""Snapshot + CSV event logging for triggered alerts."""

import csv
import os
import time

import numpy as np
import cv2

from utils.daily_dir import daily_log_dir

RISK_BEHAVIORS = [
    "phone_call", "phone_use", "head_turn", "eye_rub",
    "one_hand_raised", "two_hands_raised", "hands_busy_object",
    "no_seatbelt", "smoking",
]
LOG_HEADER = [
    "timestamp", "alerts", "score", "state", "snapshot",
    "phone_call_secs", "phone_call_cnt",
    "phone_use_secs",  "phone_use_cnt",
    "head_turn_secs",  "head_turn_cnt",
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
        stable_score: float,
        state: str,
        risk_debug: dict,
    ) -> None:
        current_alerts = frozenset(alerts)
        if not alerts:
            self._last_alerts = frozenset()  # reset so next occurrence is saved
            return
        if not self._should_save(current_alerts):
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = self._save_snapshot(frame, ts, alerts)
        self._append_csv_row(ts, alerts, stable_score, state, filename, risk_debug)

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
        stable_score: float,
        state: str,
        filename: str,
        risk_debug: dict,
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
                f"{stable_score:.1f}",
                state,
                os.path.basename(filename),
                *_risk_columns(risk_debug),
            ])


def _risk_columns(risk_debug: dict) -> list:
    cols = []
    for behavior in RISK_BEHAVIORS:
        cols += [risk_debug[behavior]["active_secs"], risk_debug[behavior]["recent_count"]]
    return cols
