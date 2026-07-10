"""Hand-near-mouth corroboration for the cigarette YOLO detector.

The `smoking_detection.onnx` model alone throws frequent false positives
(lighters, fingers, pens near the face read as a cigarette), so a hit only
counts when a wrist is also near the mouth. Pose-only, no facial landmarks.
"""

import numpy as np

from utils.geometry import Point, approx_mouth_point, distance

HAND_MOUTH_THRESHOLD = 1.2  # eye-distances; tighter than HandBusyDetector's 3.0


def hand_near_mouth(hands: list[np.ndarray], nose: Point, eye_l: Point, eye_r: Point) -> bool:
    mouth = approx_mouth_point(nose, eye_l, eye_r)
    if mouth is None or not hands:
        return False

    d_eye = distance(eye_l, eye_r)
    if d_eye == 0:
        return False

    return any(distance(hand, mouth) / d_eye < HAND_MOUTH_THRESHOLD for hand in hands)
