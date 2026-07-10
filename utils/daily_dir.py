"""Per-day subfolder resolution shared by event/video logging."""

import os
import time


def daily_log_dir(base_dir: str) -> str:
    """Return today's subfolder inside base_dir, creating it if needed.

    Rolls over automatically at midnight since the date is read fresh on
    every call, so a long-running process doesn't need a restart to start
    writing into the next day's folder.

    Example: daily_log_dir("logs") -> "logs/2026-07-10"
    """
    day_dir = os.path.join(base_dir, time.strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)
    return day_dir
