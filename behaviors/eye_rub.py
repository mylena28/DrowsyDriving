import numpy as np
from utils.geometry import distance

class EyeRubDetector:
    def __init__(self, proximity_factor=0.8, min_movement=2, max_movement=15,
                 history_size=5, frames_threshold=5):
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
        self.prev_hand = None
        self.movement_history = []

    def update(self, hand_points, eye_left, eye_right):
        """
        hand_points: lista de pontos da mão (cada ponto é (x, y)).
        Considera apenas a primeira mão da lista.
        """
        if not hand_points or eye_left is None or eye_right is None:
            self._reset_state()
            return False

        hand = hand_points[0]  # primeira mão
        d_eye = distance(eye_left, eye_right)
        if d_eye == 0:
            self._reset_state()
            return False

        # Proximidade normalizada
        d_left = distance(hand, eye_left)
        d_right = distance(hand, eye_right)
        close_to_eye = (d_left < self.proximity_factor * d_eye) or \
                       (d_right < self.proximity_factor * d_eye)

        if not close_to_eye:
            self._reset_state()
            return False

        # Movimento da mão (se houver histórico)
        if self.prev_hand is not None:
            movement = np.linalg.norm(np.array(hand) - np.array(self.prev_hand))
            self.movement_history.append(movement)
            if len(self.movement_history) > self.history_size:
                self.movement_history.pop(0)

        self.prev_hand = hand

        # Verifica oscilação (média dos movimentos recentes)
        if len(self.movement_history) >= self.history_size:
            avg_movement = np.mean(self.movement_history)
            if self.min_movement < avg_movement < self.max_movement:
                self.counter += 1
            else:
                self.counter = 0
        else:
            # Ainda coletando histórico, não incrementa contador
            self.counter = 0

        return self.counter > self.frames_threshold

    def _reset_state(self):
        self.counter = 0
        self.movement_history = []
        self.prev_hand = None

    def reset(self):
        self._reset_state()
