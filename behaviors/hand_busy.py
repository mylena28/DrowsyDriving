import numpy as np

# Wrist within this many eye-distances from the face center = hand raised/busy.
# 3.0 is permissive enough to catch hand-to-face gestures while ignoring
# hands resting on a steering wheel (which sit well below the face).
HAND_FACE_THRESHOLD = 3.0


class HandBusyDetector:
    def update(self, hand_l, hand_r, eye_l, eye_r, nose):
        """
        Returns (left_busy, right_busy) booleans based purely on wrist
        proximity to the face center. No object detection required.
        """
        left_busy = False
        right_busy = False

        if eye_l is None or eye_r is None or nose is None:
            return left_busy, right_busy

        eye_l_np = np.array(eye_l)
        eye_r_np = np.array(eye_r)
        d_eye = np.linalg.norm(eye_l_np - eye_r_np)
        if d_eye == 0:
            return left_busy, right_busy

        face_center = (eye_l_np + eye_r_np) / 2

        if hand_l is not None:
            d = np.linalg.norm(np.array(hand_l) - face_center) / d_eye
            left_busy = d < HAND_FACE_THRESHOLD

        if hand_r is not None:
            d = np.linalg.norm(np.array(hand_r) - face_center) / d_eye
            right_busy = d < HAND_FACE_THRESHOLD

        return left_busy, right_busy
