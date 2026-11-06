import csv

import numpy as np

from utils.behavior_stats import TRACKED_BEHAVIORS, BehaviorStat
from utils.event_log import LOG_HEADER, SnapshotLogger


def _stats() -> dict[str, BehaviorStat]:
    stats = {k: BehaviorStat(0.0, 0) for k in TRACKED_BEHAVIORS}
    stats["no_seatbelt"] = BehaviorStat(2.5, 3)
    return stats


def _rows(logs_dir) -> list[list[str]]:
    (csv_path,) = logs_dir.glob("*/eventos.csv")
    with open(csv_path, newline="") as f:
        return list(csv.reader(f))


def test_csv_row_has_no_score_or_state_and_matches_header(tmp_path):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    SnapshotLogger(str(tmp_path)).maybe_log(frame, ["NO SEATBELT", "SMOKING"], _stats())
    header, row = _rows(tmp_path)
    assert header == LOG_HEADER
    assert "score" not in header and "state" not in header
    assert len(row) == len(header)


def test_csv_keeps_timestamp_and_alerts_first_for_avaliador_risco(tmp_path):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    SnapshotLogger(str(tmp_path)).maybe_log(frame, ["NO SEATBELT"], _stats())
    _, row = _rows(tmp_path)
    assert len(row[0]) == len("20261106_120000") and row[1] == "NO SEATBELT"
    assert row[header_index("no_seatbelt_secs")] == "2.5"
    assert row[header_index("no_seatbelt_cnt")] == "3"


def header_index(column: str) -> int:
    return LOG_HEADER.index(column)
