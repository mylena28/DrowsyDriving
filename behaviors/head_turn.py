from typing import Optional

import numpy as np

from utils.geometry import distance

Point = Optional[np.ndarray]


class HeadTurnDetector:
    def __init__(self, yaw_threshold: float = 0.4, frames_threshold: int = 5):
        """
        yaw_threshold: sensibilidade para rotação da cabeça (diferença normalizada)
        frames_threshold: frames consecutivos para ativação
        """
        self.yaw_threshold = yaw_threshold
        self.frames_threshold = frames_threshold
        self.counter = 0

    def update(self, nose: Point, left_eye: Point, right_eye: Point) -> bool:
        """
        Retorna True se o motorista virou a cabeça por tempo suficiente.
        """
        if nose is None or left_eye is None or right_eye is None:
            self.counter = 0
            return False

        d_left = distance(nose, left_eye)
        d_right = distance(nose, right_eye)
        d_eye = distance(left_eye, right_eye)

        if d_eye == 0:
            return False

        yaw = (d_left - d_right) / d_eye

        if abs(yaw) > self.yaw_threshold:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter > self.frames_threshold

    def reset(self) -> None:
        self.counter = 0
