from typing import Optional

import numpy as np
from utils.geometry import distance

Point = Optional[np.ndarray]


class EyeRubDetector:
    def __init__(self, proximity_factor: float = 0.8, min_movement: float = 2,
                 max_movement: float = 15, history_size: int = 5,
                 frames_threshold: int = 5):
        """
        proximity_factor: distância mão-olho em relação à distância interocular
        min_movement / max_movement: movimento médio (pixels) para caracterizar esfregação
        history_size: número de movimentos recentes para média
        frames_threshold: frames consecutivos de esfregação para ativar
        """
        self.proximity_factor = proximity_factor
        self.min_movement = min_movement
        self.max_movement = max_movement
        self.history_size = history_size
        self.frames_threshold = frames_threshold
        self.counter = 0
        self.prev_hand: Point = None
        self.movement_history: list[float] = []

    def update(self, hand_points: list[np.ndarray], eye_left: Point, eye_right: Point) -> bool:
        """
        hand_points: lista de pontos da mão (cada ponto é (x, y)).
        Considera apenas a primeira mão da lista.
        """
        if not hand_points or eye_left is None or eye_right is None:
            self._reset_state()
            return False

        hand = hand_points[0]  # primeira mão
        d_eye = distance(eye_left, eye_right)
        if d_eye == 0 or not self._is_close_to_eye(hand, eye_left, eye_right, d_eye):
            self._reset_state()
            return False

        self._track_movement(hand)
        self.counter = self.counter + 1 if self._is_oscillating() else 0
        return self.counter > self.frames_threshold

    def _is_close_to_eye(self, hand: np.ndarray, eye_left: Point, eye_right: Point, d_eye: float) -> bool:
        d_left = distance(hand, eye_left)
        d_right = distance(hand, eye_right)
        return d_left < self.proximity_factor * d_eye or d_right < self.proximity_factor * d_eye

    def _track_movement(self, hand: np.ndarray) -> None:
        if self.prev_hand is not None:
            movement = np.linalg.norm(np.array(hand) - np.array(self.prev_hand))
            self.movement_history.append(movement)
            if len(self.movement_history) > self.history_size:
                self.movement_history.pop(0)
        self.prev_hand = hand

    def _is_oscillating(self) -> bool:
        """True once enough history has accumulated and the average hand
        movement sits in the rubbing range (not still, not a swipe)."""
        if len(self.movement_history) < self.history_size:
            return False
        avg_movement = np.mean(self.movement_history)
        return self.min_movement < avg_movement < self.max_movement

    def _reset_state(self) -> None:
        self.counter = 0
        self.movement_history = []
        self.prev_hand = None

    def reset(self) -> None:
        self._reset_state()
