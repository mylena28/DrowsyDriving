import cv2
import os
import numpy as np

# Caminhos automáticos para os modelos (colocar os arquivos na mesma pasta)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO_PATH = os.path.join(BASE_DIR, 'pose_deploy.prototxt')
MODEL_PATH = os.path.join(BASE_DIR, 'pose_iter_102000.caffemodel')

class HandDetector:
    def __init__(self, proto_path=PROTO_PATH, model_path=MODEL_PATH):
        if not os.path.exists(proto_path):
            raise FileNotFoundError(f"Arquivo prototxt não encontrado: {proto_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo caffemodel não encontrado: {model_path}")

        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1

    def update(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0/255, (self.inWidth, self.inHeight),
                                     (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()

        H = output.shape[2]
        W = output.shape[3]
        points = []
        # O modelo OpenPose mãos tem 22 saídas (21 pontos + fundo)
        for i in range(22):
            prob_map = output[0, i, :, :]
            prob_map = cv2.resize(prob_map, (w, h))
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            if prob > self.threshold:
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)
        return points

# Instância global (singleton) para ser usada pela função detect_hands
_detector = None

def detect_hands(frame, rgb):
    """
    Função compatível com a interface esperada pelo main.py.
    Retorna uma lista de pontos (x, y) da primeira mão detectada,
    ignorando pontos None. O parâmetro rgb não é usado (mantido por compatibilidade).
    """
    global _detector
    if _detector is None:
        _detector = HandDetector()
    hand_points = _detector.update(frame)
    # Filtra pontos válidos (ignora None)
    valid_points = [p for p in hand_points if p is not None]
    return valid_points
