import numpy as np

class PhoneStateDetector:
    def __init__(self, lateral_factor=0.5, distance_norm_threshold=1.8,
                 call_frames_threshold=8, vertical_phone_factor=0.0):
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

    def update(self, phone_center, nose, eye_left, eye_right):
        """
        Retorna um string: "ATENTO", "USANDO CELULAR" ou "EM LIGACAO"
        """
        state = "ATENTO"

        if phone_center is not None and nose is not None and \
           eye_left is not None and eye_right is not None:

            phone = np.array(phone_center)
            nose_np = np.array(nose)
            eye_l = np.array(eye_left)
            eye_r = np.array(eye_right)

            d_face = np.linalg.norm(phone - nose_np)
            d_eye = np.linalg.norm(eye_l - eye_r)
            if d_eye == 0:
                return state
            d_norm = d_face / d_eye

            face_vec = eye_r - eye_l
            phone_vec = phone - nose_np
            # Produto escalar - positivo indica alinhamento com a face
            alignment = np.dot(phone_vec, face_vec)
            lateral = abs(alignment) > self.lateral_factor * (np.linalg.norm(face_vec) ** 2)
            close = d_norm < self.distance_norm_threshold

            # Detecção de ligação (telefone ao ouvido)
            if lateral and close:
                self.counter += 1
            else:
                self.counter = 0

            if self.counter > self.call_frames_threshold:
                return "EM LIGACAO"

            # Uso normal (olhando para baixo)
            face_center_y = (eye_left[1] + eye_right[1]) / 2
            # Se phone[1] (y) > face_center_y, o celular está abaixo do centro do rosto
            if self.vertical_phone_factor == 0:
                below_face = phone[1] > face_center_y
            else:
                below_face = phone[1] > face_center_y * self.vertical_phone_factor

            if below_face:
                state = "USANDO CELULAR"
        else:
            # Sem celular detectado, reset contador
            self.counter = 0

        return state

    def reset(self):
        self.counter = 0
