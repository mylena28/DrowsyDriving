import cv2
import numpy as np

class OpenCVFaceDetector:
    def __init__(self, model_path='detectors/face_detection_yunet_2023mar.onnx', input_size=(320, 320), conf_threshold=0.9, nms_threshold=0.3, top_k=5000, backend_id=0, target_id=0):
        """
        Inicializa o detector de faces e landmarks baseado no YuNet.

        :param model_path: Caminho para o arquivo do modelo ONNX do YuNet.
        :param input_size: Tamanho da imagem de entrada para o modelo (largura, altura).
        :param conf_threshold: Limiar de confiança para detecções.
        :param nms_threshold: Limiar para o Non-Maximum Suppression.
        :param top_k: Número máximo de detecções a serem mantidas.
        :param backend_id: ID do backend de inferência do OpenCV.
        :param target_id: ID do dispositivo de destino (ex: CPU).
        """
        self.model = cv2.FaceDetectorYN_create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=backend_id,
            target_id=target_id
        )
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.backend_id = backend_id
        self.target_id = target_id

    def update(self, frame):
        """
        Processa um frame para detectar rosto e landmarks.

        :param frame: Imagem BGR (formato do OpenCV).
        :return: Uma tupla (face_landmarks, nose_point, eye_left, eye_right).
                 - face_landmarks: None (para compatibilidade com código antigo).
                 - nose_point: Tupla (x, y) ou None.
                 - eye_left: Tupla (x, y) ou None.
                 - eye_right: Tupla (x, y) ou None.
        """
        height, width, _ = frame.shape
        # Redimensiona o detector para o tamanho do frame atual
        self.model.setInputSize((width, height))

        # Executa a detecção
        _, faces = self.model.detect(frame)

        # Se não encontrou nenhuma face, retorna None
        if faces is None or len(faces) == 0:
            return None, None, None, None

        # Pega a primeira face detectada
        face = faces[0]

        # O modelo YuNet retorna 15 valores: [x1, y1, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rc, y_rc, x_lc, y_lc, score]
        # As coordenadas são normalizadas entre 0 e 1. Vamos convertê-las para pixels.
        # Convertendo para inteiros para usar como índices
        coords = face[:14].astype(np.int32)

        # Desempacotando as coordenadas
        x_le, y_le = coords[6], coords[7]  # olho esquerdo
        x_re, y_re = coords[4], coords[5]  # olho direito
        x_n, y_n = coords[8], coords[9]    # nariz

        # Converte os pontos para o formato esperado pelo resto do código
        eye_left = (x_le, y_le)
        eye_right = (x_re, y_re)
        nose_point = (x_n, y_n)

        # Retorna um placeholder para 'face_landmarks' para manter compatibilidade
        # com as funções que esperam esse argumento (ex: yawn)
        return None, nose_point, eye_left, eye_right
