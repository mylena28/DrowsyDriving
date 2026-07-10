from typing import Optional

import numpy as np

Point = Optional[np.ndarray]


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Mouth sits below the nose, roughly proportional to inter-eye distance —
# COCO pose has no mouth keypoint, so this is a coarse proxy scaled by it.
MOUTH_OFFSET_FACTOR = 0.4


def approx_mouth_point(nose: Point, eye_l: Point, eye_r: Point) -> Point:
    """Approximate mouth position from nose + eyes (no mouth keypoint in
    COCO pose). Used to corroborate the smoking model with a hand-to-mouth
    gesture check (see behaviors/smoking_gesture.py).
    """
    if nose is None or eye_l is None or eye_r is None:
        return None
    d_eye = distance(eye_l, eye_r)
    return np.array(nose) + np.array([0.0, d_eye * MOUTH_OFFSET_FACTOR])


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


def torso_crop(frame: np.ndarray, shoulder_l: Point, shoulder_r: Point) -> np.ndarray:
    """Crop the chest/torso region (shoulders down to waist-ish) for
    seatbelt detection.

    Uses shoulder width as the body scale, same idea as lower_face_crop's
    use of inter-eye distance. Falls back to the full frame if either
    shoulder keypoint is missing or too close together to be reliable.
    """
    if shoulder_l is None or shoulder_r is None:
        return frame

    shoulder_dist = distance(shoulder_l, shoulder_r)
    if shoulder_dist < 10:
        return frame

    x1, x2, y1, y2 = _torso_bounds(frame, shoulder_l, shoulder_r, shoulder_dist)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _torso_bounds(
    frame: np.ndarray, shoulder_l: np.ndarray, shoulder_r: np.ndarray, shoulder_dist: float,
) -> tuple[int, int, int, int]:
    mid = (np.array(shoulder_l) + np.array(shoulder_r)) / 2
    cx, cy = int(mid[0]), int(mid[1])
    h, w = frame.shape[:2]

    # Horizontal: 1.5x shoulder-width either side (seatbelt strap crosses diagonally)
    # Vertical: 0.3x above shoulder line, 3x below (down to roughly the buckle)
    margin_x = int(shoulder_dist * 1.5)
    margin_up = int(shoulder_dist * 0.3)
    margin_down = int(shoulder_dist * 3.0)

    x1 = max(0, cx - margin_x)
    x2 = min(w, cx + margin_x)
    y1 = max(0, cy - margin_up)
    y2 = min(h, cy + margin_down)
    return x1, x2, y1, y2
