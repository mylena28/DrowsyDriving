"""Head-down / possible-dozing detector, pose-only (no facial landmarks —
a companion project already does EAR/PERCLOS/yawn from facial landmarks;
this stays pose-only by design so the two projects don't overlap in scope).
"""

from typing import Optional

import numpy as np

from utils.geometry import distance

Point = Optional[np.ndarray]


class HeadDownDetector:
    """Flags a sustained head-down posture from YOLOv8-Pose keypoints only.

    Two independent signals, either sufficient on its own:
      - pitch: nose sits too close to the shoulder line, normalized by
        shoulder width — the head has tipped forward/down. Requires
        shoulders + nose + eyes all present.
      - face_lost: nose/eyes vanish from pose for a long sustained window
        while shoulders stay tracked (driver still seated in frame, just not
        facing the camera) — a coarse dozing/slump proxy.

    Caveat: face_lost can't fully distinguish a genuine head-down from a
    sustained mirror-check/dashboard-glance/hand-occluding-face — a full
    head-turn also drops nose/eyes. Mitigated, not eliminated, by requiring
    face_lost to persist far longer than the pitch signal.

    Thresholds below are frame counts, not calibrated seconds (Pi 5
    CPU-only inference likely runs under the ~30 FPS cap) — untested
    starting guesses, tune against real on-device footage.
    """

    def __init__(
        self,
        pitch_ratio_threshold: float = 0.5,
        pitch_frames_threshold: int = 6,
        face_lost_frames_threshold: int = 28,
    ):
        self.pitch_ratio_threshold = pitch_ratio_threshold
        self.pitch_frames_threshold = pitch_frames_threshold
        self.face_lost_frames_threshold = face_lost_frames_threshold
        self._pitch_counter = 0
        self._face_lost_counter = 0

    def update(
        self,
        nose: Point, eye_l: Point, eye_r: Point,
        shoulder_l: Point, shoulder_r: Point,
    ) -> bool:
        if shoulder_l is None or shoulder_r is None:
            self._pitch_counter = 0
            self._face_lost_counter = 0
            return False
        if nose is None or eye_l is None or eye_r is None:
            self._pitch_counter = 0
            return self._update_face_lost()

        self._face_lost_counter = 0
        return self._update_pitch(nose, shoulder_l, shoulder_r)

    def _update_pitch(self, nose: Point, shoulder_l: Point, shoulder_r: Point) -> bool:
        shoulder_width = distance(shoulder_l, shoulder_r)
        if shoulder_width == 0:
            self._pitch_counter = 0
            return False

        shoulder_mid_y = (shoulder_l[1] + shoulder_r[1]) / 2
        pitch_ratio = (shoulder_mid_y - nose[1]) / shoulder_width
        if pitch_ratio < self.pitch_ratio_threshold:
            self._pitch_counter += 1
        else:
            self._pitch_counter = 0
        return self._pitch_counter >= self.pitch_frames_threshold

    def _update_face_lost(self) -> bool:
        self._face_lost_counter += 1
        return self._face_lost_counter >= self.face_lost_frames_threshold
