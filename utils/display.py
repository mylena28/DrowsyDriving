import cv2
import numpy as np

# Only shows what was detected. No state label / color bands: risk levels are
# decided by the AvaliadorRisco container, not here (see utils/behavior_stats).
_HEADER_COLOR = (255, 255, 255)
_ALERT_COLOR = (0, 200, 255)


def draw_status(frame: np.ndarray, alerts: list[str] | None = None) -> np.ndarray:
    """Writes the active detections on the debug window frame.

    Example: draw_status(frame, ["SEM CINTO DE SEGURANCA"])
    """
    x, y = 20, 40
    header = "DETECCOES" if alerts else "SEM DETECCOES"
    cv2.putText(frame, header, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _HEADER_COLOR, 2)

    for i, alert in enumerate(alerts or []):
        cv2.putText(frame, f"  {alert}",
                    (x, y + 40 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _ALERT_COLOR, 2)

    return frame
