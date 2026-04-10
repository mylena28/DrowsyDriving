import cv2
import numpy as np

class OpenCVFaceDetector:
    def __init__(self, model_path='detectors/face_detection_yunet_2023mar.onnx', input_size=(640, 480)):
        # Inicializa o detector usando a API específica para YuNet
        # Isso substitui o antigo cv2.dnn.readNet que usava o .caffemodel
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=1  # Otimização: foca apenas na face principal do motorista
        )

        # Configura o backend para usar a CPU do Raspberry Pi da forma mais eficiente
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def update(self, frame):
        # O YuNet precisa que o input_size coincida com o frame atual
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))

        # Realiza a detecção
        result, faces = self.detector.detect(frame)

        if faces is not None:
            # O YuNet retorna um array onde a primeira linha [0] é a face mais provável
            f = faces[0]

            # Extração dos pontos (landmarks) conforme a documentação do YuNet:
            # f[0:4]   -> Bounding box (x, y, w, h)
            # f[4:6]   -> Olho Direito
            # f[6:8]   -> Olho Esquerdo
            # f[8:10]  -> Nariz
            # f[10:12] -> Canto Direito da Boca
            # f[12:14] -> Canto Esquerdo da Boca

            face_box = f[0:4].astype(int)
            eye_r = f[4:6].astype(int)
            eye_l = f[6:8].astype(int)
            nose = f[8:10].astype(int)

            return face_box, nose, eye_l, eye_r

        return None, None, None, None
