import cv2
from picamera2 import Picamera2

from detectors.object_detector import detect_objects
from detectors.face_detector import OpenCVFaceDetector
from detectors.hand_detector import detect_hands

from behaviors.yawn import YawnDetector
from behaviors.head_turn import HeadTurnDetector
from behaviors.eye_rub import EyeRubDetector
from behaviors.phone_usage import PhoneStateDetector

from utils.display import draw_status
from utils.stabilizer import StateStabilizer
from utils.risk import compute_score
from utils.classifier import classify_from_score
from ultralytics import YOLO

# Instâncias globais dos detectores de comportamento
yawn_detector = YawnDetector(frames_threshold=5)
head_turn_detector = HeadTurnDetector(frames_threshold=5)
eye_rub_detector = EyeRubDetector(frames_threshold=5)
phone_detector = PhoneStateDetector(call_frames_threshold=8)

# Modelo YOLO
model = YOLO("yolov8n.pt")

# Configuração da Picamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": 'BGR888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

def main():
    stabilizer = StateStabilizer(window_size=15, change_threshold=5)
    face_detector = OpenCVFaceDetector(model_path='face_detection_yunet_2023mar.onnx')  # caminho ajustado

    frame_count = 0
    while True:
        print("DEBUG: Iniciando main()", flush=True)
        # Captura o frame diretamente da picamera2 (já em formato BGR)
        frame = picam2.capture_array()
        frame_count += 1
        print("DEBUG: Iniciando Picamera2...", flush=True)


        # Converte para RGB (necessário apenas para o hand_detector? vamos manter)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # DETECÇÃO
        face, nose, eye_l, eye_r = face_detector.update(frame)
        hands = detect_hands(frame, rgb)
        phone, hands_busy = detect_objects(frame, model, nose, eye_l, eye_r, hands)

        # COMPORTAMENTOS
        yawning = yawn_detector.update(face, eye_l, eye_r, frame.shape)
        head_turned = head_turn_detector.update(nose, eye_l, eye_r)
        eye_rubbing = eye_rub_detector.update(hands, eye_l, eye_r)
        phone_state = phone_detector.update(phone, nose, eye_l, eye_r)

        # EVENTOS
        events = {
            "phone_call": phone_state == "EM LIGACAO",
            "phone_use": phone_state == "USANDO CELULAR",
            "hands_busy": hands_busy,
            "yawn": yawning,
            "eye_rub": eye_rubbing,
            "head_turn": head_turned
        }

        score = compute_score(events)
        stable_score = stabilizer.update(score)
        state = classify_from_score(stable_score)

        # Desenha o status no frame (opcional, se quiser ver a janela)
        frame = draw_status(frame, state, stable_score)
        # cv2.imshow("Sistema de Monitoramento", frame)  # descomente para ver a janela
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Diagnóstico a cada 30 frames
        if frame_count % 30 == 0:
            print(f"\n--- DIAGNÓSTICO (frame {frame_count}) ---")
            print(f"Face detectada: {nose is not None}")
            print(f"Mãos: {len(hands)} pontos detectados")
            print(f"Celular: {phone if phone else 'não detectado'}")
            print(f"Eventos ativos: { {k:v for k,v in events.items() if v} }")
            print(f"Score bruto: {score:.1f} | Score estabilizado: {stable_score:.1f}")
            print(f"Estado: {state}")
            print(f"Telefone: {phone_state}")
            print("-------------------------------------\n")

    # O loop é infinito; para encerrar, use Ctrl+C no terminal

if __name__ == "__main__":
    main()
