from utils.geometry import distance

class YawnDetector:
    def __init__(self, mouth_threshold=0.35, frames_threshold=5):
        """
        mouth_threshold: razão abertura boca / distância interocular
        frames_threshold: número de frames consecutivos para ativar
        """
        self.mouth_threshold = mouth_threshold
        self.frames_threshold = frames_threshold
        self.counter = 0

    def update(self, face_landmarks, eye_left, eye_right, frame_shape):
        """
        Retorna True se bocejo detectado, False caso contrário.
        """
        if face_landmarks is None or eye_left is None or eye_right is None:
            self.counter = 0
            return False

        h, w, _ = frame_shape

        # Pontos da boca (landmarks 13 e 14 do MediaPipe FaceMesh)
        upper = face_landmarks.landmark[13]
        lower = face_landmarks.landmark[14]

        p1 = (int(upper.x * w), int(upper.y * h))
        p2 = (int(lower.x * w), int(lower.y * h))

        d_mouth = distance(p1, p2)
        d_eye = distance(eye_left, eye_right)

        if d_eye == 0:
            return False

        ratio = d_mouth / d_eye

        if ratio > self.mouth_threshold:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter > self.frames_threshold

    def reset(self):
        self.counter = 0
