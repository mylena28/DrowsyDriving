from typing import Optional

import numpy as np

Point = Optional[np.ndarray]

# Wrist within this many eye-distances from the face center = hand raised/busy.
# 3.0 is permissive enough to catch hand-to-face gestures while ignoring
# hands resting on a steering wheel (which sit well below the face).
HAND_FACE_THRESHOLD = 3.0


class HandBusyDetector:
    def update(
        self, hand_l: Point, hand_r: Point, eye_l: Point, eye_r: Point, nose: Point,
    ) -> tuple[bool, bool]:
        """
        Returns (left_busy, right_busy) booleans based purely on wrist
        proximity to the face center. No object detection required.
        """
        if eye_l is None or eye_r is None or nose is None:
            return False, False

        eye_l_np, eye_r_np = np.array(eye_l), np.array(eye_r)
        d_eye = np.linalg.norm(eye_l_np - eye_r_np)
        if d_eye == 0:
            return False, False

        face_center = (eye_l_np + eye_r_np) / 2
        return (
            self._is_hand_busy(hand_l, face_center, d_eye),
            self._is_hand_busy(hand_r, face_center, d_eye),
        )

    @staticmethod
    def _is_hand_busy(hand: Point, face_center: np.ndarray, d_eye: float) -> bool:
        if hand is None:
            return False
        d = np.linalg.norm(np.array(hand) - face_center) / d_eye
        return d < HAND_FACE_THRESHOLD
