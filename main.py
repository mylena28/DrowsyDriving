import cv2
import sys
import os

# Tentativa segura de importar picamera2 apenas se o hardware existir
PICAMERA_AVAILABLE = False
# Checa se é um ambiente Raspberry Pi para evitar erros em containers genéricos [cite: 263]
if os.path.exists('/usr/bin/libcamera-hello'):
    try:
        from picamera2 import Picamera2
        PICAMERA_AVAILABLE = True
        print("Picamera2 detectado.")
    except (ImportError, RuntimeError):
        print("Erro ao carregar Picamera2. Usando fallback.")
else:
    print("Ambiente sem suporte a Picamera2 (Notebook/Docker). Usando webcam.")

from detectors.object_detector import detect_objects
# YuNet removido para simplificação e correção de erros [cite: 171, 191]
from detectors.pose_detector import PoseDetector

from behaviors.head_turn import HeadTurnDetector
from behaviors.eye_rub import EyeRubDetector
from behaviors.phone_usage import PhoneStateDetector

from utils.display import draw_status
from utils.stabilizer import StateStabilizer
from utils.risk import compute_score
from utils.classifier import classify_from_score
from ultralytics import YOLO

# Instâncias dos detectores
head_turn_detector = HeadTurnDetector(frames_threshold=5)
eye_rub_detector = EyeRubDetector(frames_threshold=5)
phone_detector = PhoneStateDetector(call_frames_threshold=8)

# Modelos YOLO (YOLOv8n para celular e YOLOv8n-pose para esqueleto) [cite: 17, 126]
# Carregados globalmente para evitar reinicializações custosas [cite: 343]
model = YOLO("yolov8n.pt")
pose_detector = PoseDetector("yolov8n-pose.pt")

def main():
    # Inicialização da câmera [cite: 4, 182]
    if PICAMERA_AVAILABLE:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"format": 'BGR888', "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        get_frame = lambda: picam2.capture_array()
        print("Câmera PiCamera2 iniciada.")
    else:
        # Fallback para webcam em Notebooks ou containers com acesso a /dev/video0 [cite: 271]
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro: não foi possível abrir a webcam.")
            return
        get_frame = lambda: cap.read()[1]
        print("Webcam aberta com sucesso.")

    stabilizer = StateStabilizer(window_size=15, change_threshold=5)

    # Lógica de exibição condicional (Modo Headless) [cite: 381]
    # Garante que SHOW_DISPLAY seja Falso se estiver no Rasp ou se o DISPLAY estiver vazio/ausente [cite: 209, 385]
    SHOW_DISPLAY = False
    if not PICAMERA_AVAILABLE:
        env_display = os.environ.get("DISPLAY", "")
        if env_display.strip(): # Só abre janela se houver um servidor X11 configurado [cite: 365, 377]
            SHOW_DISPLAY = True
            print("Interface gráfica detectada. Janela de monitoramento ativa.")

    frame_count = 0
    while True:
        frame = get_frame()
        if frame is None:
            break
        frame_count += 1

        # DETECÇÕES UNIFICADAS VIA YOLO-POSE [cite: 221]
        # Extrai face e mãos em um único processamento para ganhar performance [cite: 189, 227]
        pose_data = pose_detector.update(frame)

        nose = pose_data["nose"]
        eye_l = pose_data["eye_l"]
        eye_r = pose_data["eye_r"]
        hands = pose_data["hands"]

        face = None # O YuNet não é mais necessário [cite: 189]

        # Detecção de objetos com correção de erro NumPy para evitar ambiguidade de arrays [cite: 198, 221]
        phone, hands_busy = detect_objects(frame, model, nose, eye_l, eye_r, hands)

        # COMPORTAMENTOS (usam os pontos vindos do YOLOv8-pose) [cite: 173]
        head_turned = head_turn_detector.update(nose, eye_l, eye_r)
        eye_rubbing = eye_rub_detector.update(hands, eye_l, eye_r)
        phone_state = phone_detector.update(phone, nose, eye_l, eye_r)

        # EVENTOS
        events = {
            "phone_call": phone_state == "EM LIGACAO",
            "phone_use": phone_state == "USANDO CELULAR",
            "hands_busy": hands_busy,
            "eye_rub": eye_rubbing,
            "head_turn": head_turned
        }

        score = compute_score(events)
        stable_score = stabilizer.update(score)
        state = classify_from_score(stable_score)

        # Diagnóstico no terminal (Sempre ativo para monitoramento em clusters) [cite: 210, 225]
        if frame_count % 30 == 0:
            print(f"\n--- DIAGNÓSTICO (frame {frame_count}) ---")
            print(f"Face (via Pose): {nose is not None}")
            print(f"Mãos detectadas: {len(hands)}")
            print(f"Celular: {phone if phone is not None else 'não detectado'}")
            print(f"Estado: {state} | Score: {stable_score:.1f}")
            print("-------------------------------------\n")

        # Display condicional: só executa cv2.imshow se houver suporte gráfico [cite: 226, 381]
        if SHOW_DISPLAY:
            display_frame = draw_status(frame, state, stable_score)
            cv2.imshow("Monitoramento DrowsyDriving", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberação dos recursos para evitar processos zumbis [cite: 188, 226]
    if PICAMERA_AVAILABLE:
        picam2.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
