from typing import Optional

import numpy as np

from detectors.yolo_onnx import YOLOOnnx

Point = Optional[np.ndarray]

THRESH_FACE = 3.0   # mais permissivo
THRESH_HAND = 1.5   # mais restritivo


def detect_objects(
    frame: np.ndarray,
    model: YOLOOnnx,
    nose_point: Point,
    eye_l: Point,
    eye_r: Point,
    hands: list[np.ndarray],
) -> tuple[Optional[tuple], bool]:
    """Detects a phone (position returned) and whether any object of
    interest (phone/bottle/cup) is close to both the face and a wrist."""
    phone_center = None
    hands_busy = False

    results = model(frame, imgsz=320, verbose=False)
    for r in results:
        for box in r.boxes:
            label, obj_center = _labeled_center(box, model)
            if label not in ("cell phone", "bottle", "cup"):
                continue

            if label == "cell phone":
                phone_center = tuple(obj_center)

            if _object_busies_hands(obj_center, nose_point, eye_l, eye_r, hands):
                hands_busy = True

    return phone_center, hands_busy


def _labeled_center(box, model: YOLOOnnx) -> tuple[str, np.ndarray]:
    label = model.names[int(box.cls[0])]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
    return label, center


def _object_busies_hands(
    obj_center: np.ndarray, nose_point: Point, eye_l: Point, eye_r: Point, hands: list[np.ndarray],
) -> bool:
    """True if obj_center is close to both the face and some wrist,
    normalized by inter-eye distance.

    Uses 'is not None' rather than truthy checks — a bare `if array:` on a
    multi-element numpy array raises ValueError.
    """
    if nose_point is None or eye_l is None or eye_r is None or not hands:
        return False

    d_eye = np.linalg.norm(np.array(eye_l) - np.array(eye_r))
    if d_eye == 0:
        return False

    close_to_face = np.linalg.norm(obj_center - np.array(nose_point)) / d_eye < THRESH_FACE
    close_to_hand = any(
        np.linalg.norm(obj_center - np.array(hand)) / d_eye < THRESH_HAND
        for hand in hands
    )
    return close_to_face and close_to_hand
