from typing import Optional

import numpy as np

Point = Optional[np.ndarray]


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))


def lower_face_crop(frame: np.ndarray, nose: Point, eye_l: Point, eye_r: Point) -> np.ndarray:
    """Crop the lower-face region (nose down to chin) for small-object detection.

    Uses inter-eye distance as the face scale. Falls back to the full frame if
    any keypoint is missing or the face is too small to be reliable.
    """
    if nose is None or eye_l is None or eye_r is None:
        return frame

    eye_dist = distance(eye_l, eye_r)
    if eye_dist < 10:
        return frame

    x1, x2, y1, y2 = _crop_bounds(frame, nose, eye_dist)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _crop_bounds(frame: np.ndarray, nose: np.ndarray, eye_dist: float) -> tuple[int, int, int, int]:
    cx, cy = int(nose[0]), int(nose[1])
    h, w = frame.shape[:2]

    # Horizontal: 1.2x eye-distance either side of nose
    # Vertical: 0.5x above nose (to include nose tip context), 1.8x below (chin)
    margin_x = int(eye_dist * 1.2)
    margin_up = int(eye_dist * 0.5)
    margin_down = int(eye_dist * 1.8)

    x1 = max(0, cx - margin_x)
    x2 = min(w, cx + margin_x)
    y1 = max(0, cy - margin_up)
    y2 = min(h, cy + margin_down)
    return x1, x2, y1, y2
