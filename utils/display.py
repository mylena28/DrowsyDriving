import cv2
import numpy as np

from utils.classifier import DEFAULT_STATE

# Keyed by the same state labels classify_from_score() returns, so the
# score breakpoints only live in one place (utils/classifier).
_STATE_COLORS = {
    "ATENTO": (0, 255, 0),
    "MULTIPLAS TAREFAS": (0, 255, 255),
    "DESATENCAO": (0, 165, 255),
    "RISCO": (0, 0, 255),
    DEFAULT_STATE: (0, 0, 150),
}


def get_color(state: str) -> tuple[int, int, int]:
    return _STATE_COLORS.get(state, _STATE_COLORS[DEFAULT_STATE])


def draw_status(
    frame: np.ndarray,
    state: str,
    alerts: list[str] | None = None,
) -> np.ndarray:

    x, y = 20, 40
    color = get_color(state)

    cv2.putText(frame, state,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)

    if alerts:
        for i, alert in enumerate(alerts):
            cv2.putText(frame, f"  {alert}",
                        (x, y + 40 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 255), 2)

    return frame
