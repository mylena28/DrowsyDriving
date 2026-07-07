from typing import Optional

import numpy as np

Point = Optional[np.ndarray]


class PhoneStateDetector:
    def __init__(self, lateral_factor: float = 0.5, distance_norm_threshold: float = 1.8,
                 call_frames_threshold: int = 8, vertical_phone_factor: float = 0.0):
        """
        lateral_factor: multiplicador para considerar posição lateral (0.5 típico)
        distance_norm_threshold: distância celular-nariz normalizada (1.8)
        call_frames_threshold: frames consecutivos em posição de ligação
        vertical_phone_factor: se >0, exige que o celular esteja abaixo deste fator * face_center_y
        """
        self.lateral_factor = lateral_factor
        self.distance_norm_threshold = distance_norm_threshold
        self.call_frames_threshold = call_frames_threshold
        self.vertical_phone_factor = vertical_phone_factor
        self.counter = 0

    def update(self, phone_center: Optional[tuple], nose: Point, eye_left: Point, eye_right: Point) -> str:
        """
        Retorna um string: "ATENTO", "USANDO CELULAR" ou "EM LIGACAO"
        """
        if phone_center is None or nose is None or eye_left is None or eye_right is None:
            self.counter = 0  # sem celular detectado, reset contador
            return "ATENTO"

        phone, nose_np = np.array(phone_center), np.array(nose)
        eye_l, eye_r = np.array(eye_left), np.array(eye_right)
        d_eye = np.linalg.norm(eye_l - eye_r)
        if d_eye == 0:
            return "ATENTO"

        # Detecção de ligação (telefone ao ouvido)
        if self._is_call_position(phone, nose_np, eye_l, eye_r, d_eye):
            self.counter += 1
        else:
            self.counter = 0
        if self.counter > self.call_frames_threshold:
            return "EM LIGACAO"

        if self._is_below_face(phone, eye_left, eye_right):
            return "USANDO CELULAR"
        return "ATENTO"

    def _is_call_position(
        self, phone: np.ndarray, nose_np: np.ndarray, eye_l: np.ndarray, eye_r: np.ndarray, d_eye: float,
    ) -> bool:
        d_norm = np.linalg.norm(phone - nose_np) / d_eye
        face_vec = eye_r - eye_l
        phone_vec = phone - nose_np
        # Produto escalar - positivo indica alinhamento com a face
        alignment = np.dot(phone_vec, face_vec)
        lateral = abs(alignment) > self.lateral_factor * (np.linalg.norm(face_vec) ** 2)
        close = d_norm < self.distance_norm_threshold
        return lateral and close

    def _is_below_face(self, phone: np.ndarray, eye_left: Point, eye_right: Point) -> bool:
        """Uso normal (olhando para baixo): celular abaixo do centro do rosto."""
        face_center_y = (eye_left[1] + eye_right[1]) / 2
        if self.vertical_phone_factor == 0:
            return phone[1] > face_center_y
        return phone[1] > face_center_y * self.vertical_phone_factor

    def reset(self) -> None:
        self.counter = 0
