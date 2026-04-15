import cv2
import sys
import os
import time
import numpy as np

# --- DETECÇÃO DE HARDWARE ---
# No Raspberry Pi 5 com câmera CSI, NÃO use fallback para V4L2
# pois o OpenCV não consegue ler diretamente. Use APENAS Picamera2.
PICAMERA_AVAILABLE = False
picam2 = None

# Verifica se está em ambiente Raspberry Pi com libcamera
IS_RASPBERRY_PI = os.path.exists('/usr/bin/rpicam-hello') or os.path.exists('/usr/bin/libcamera-hello')

if IS_RASPBERRY_PI:
    try:
        from picamera2 import Picamera2
        PICAMERA_AVAILABLE = True
        print("✅ Ambiente Raspberry Pi 5 detectado. Usando Picamera2 (recomendado).")
    except ImportError as e:
        print(f"❌ Picamera2 não instalada: {e}")
        print("   Instale com: pip install picamera2")
        print("   Ou: sudo apt install python3-picamera2")
        PICAMERA_AVAILABLE = False
        sys.exit(1)
else:
    print("Ambiente não-RPi detectado. Usando OpenCV padrão (webcam USB).")

# Importação dos módulos locais (Detectores e Lógicas)
from detectors.object_detector import detect_objects
from detectors.pose_detector import PoseDetector
from behaviors.head_turn import HeadTurnDetector
from behaviors.eye_rub import EyeRubDetector
from behaviors.phone_usage import PhoneStateDetector
from utils.display import draw_status
from utils.stabilizer import StateStabilizer
from utils.risk import compute_score
from utils.classifier import classify_from_score
from detectors.yolo_onnx import YOLOOnnx

# Instâncias dos detectores corporais
head_turn_detector = HeadTurnDetector(frames_threshold=5)
eye_rub_detector = EyeRubDetector(frames_threshold=5)
phone_detector = PhoneStateDetector(call_frames_threshold=8)

# Modelos YOLO carregados globalmente para economizar memória no Pi 5
model = YOLOOnnx("yolov8n.onnx")
pose_detector = PoseDetector("yolov8n-pose.onnx")

def initialize_picamera2():
    """Inicializa a câmera usando Picamera2 (funciona no Pi 5)"""
    try:
        picam2 = Picamera2()

        # Configuração otimizada para o Pi 5
        config = picam2.create_video_configuration(
            main={
                "format": 'RGB888',      # Formato compatível com OpenCV
                "size": (640, 480)       # Resolução balanceada para performance
            },
            controls={
                "FrameRate": 30.0,       # Limita FPS para estabilidade
                "AwbMode": 0,            # Auto white balance
                "AnalogueGain": 1.0      # Ganho padrão
            }
        )

        picam2.configure(config)
        picam2.start()

        # Aguarda a câmera estabilizar (crítico para o Pi 5)
        time.sleep(2.0)

        # Testa se realmente captura frames
        test_frame = picam2.capture_array()
        if test_frame is None or test_frame.size == 0:
            raise Exception("Frame de teste vazio")

        print("✅ Picamera2 inicializada com sucesso!")
        print(f"   Resolução: 640x480 | Formato: RGB888 | FPS: 30")
        return picam2, True

    except Exception as e:
        print(f"❌ Erro ao inicializar Picamera2: {e}")
        print("\n📋 Diagnóstico rápido:")
        print("   1. Verifique o cabo da câmera:")
        print("      libcamera-hello --list-cameras")
        print("   2. Teste a câmera manualmente:")
        print("      libcamera-hello -t 5")
        print("   3. Reinicie o serviço de câmera:")
        print("      sudo systemctl restart rpicam-apps")
        return None, False

def initialize_opencv_fallback():
    """Fallback para OpenCV (webcams USB ou outros sistemas)"""
    try:
        # Tenta diferentes índices de câmera
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"✅ Câmera encontrada no índice {camera_index}")

                # Configurações otimizadas
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # Testa se realmente captura
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print("✅ Fallback OpenCV funcionando")
                    return cap, True
                else:
                    print(f"⚠️  Câmera índice {camera_index} não produz frames")
                    cap.release()

        print("❌ Nenhuma câmera encontrada no sistema")
        return None, False

    except Exception as e:
        print(f"❌ Erro no fallback OpenCV: {e}")
        return None, False

def main():
    global picam2, PICAMERA_AVAILABLE

    # --- INICIALIZAÇÃO DA CÂMERA ---
    cap_opencv = None
    get_frame = None

    if PICAMERA_AVAILABLE and IS_RASPBERRY_PI:
        # PRIORIDADE 1: Picamera2 (funciona com câmera CSI no Pi 5)
        picam2, success = initialize_picamera2()
        if success:
            # Função que retorna frame já em formato BGR para o OpenCV
            def get_frame_picamera2():
                frame_rgb = picam2.capture_array()
                # Converte RGB para BGR (OpenCV usa BGR)
                return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            get_frame = get_frame_picamera2
        else:
            print("\n⚠️  Picamera2 falhou. Tentando fallback...")
            PICAMERA_AVAILABLE = False

    if not PICAMERA_AVAILABLE or get_frame is None:
        # PRIORIDADE 2: OpenCV (para webcams USB ou ambiente não-RPi)
        cap_opencv, success = initialize_opencv_fallback()
        if success:
            get_frame = lambda: cap_opencv.read()[1]
        else:
            print("\n❌ ERRO CRÍTICO: Nenhuma câmera disponível.")
            print("   Soluções possíveis:")
            print("   1. Raspberry Pi 5 com câmera CSI:")
            print("      - Use Picamera2 (não use fallback V4L2)")
            print("      - Verifique: libcamera-hello --list-cameras")
            print("   2. Webcam USB:")
            print("      - Verifique conexão: lsusb")
            print("      - Teste: cheese ou guvcview")
            return

    # --- INÍCIO DO MONITORAMENTO ---
    stabilizer = StateStabilizer(window_size=15, change_threshold=5)

    # Define se deve mostrar janela gráfica
    SHOW_DISPLAY = "DISPLAY" in os.environ and not IS_RASPBERRY_PI

    frame_count = 0
    fps_update_time = time.time()
    fps_counter = 0

    print("\n" + "="*50)
    print("🚀 MONITORAMENTO INICIADO")
    print("="*50)
    print("📊 Pressione Ctrl+C para encerrar\n")

    try:
        while True:
            frame_start = time.time()

            # Captura o frame
            frame = get_frame()

            if frame is None:
                print("⚠️  Frame perdido. Tentando recuperar...")
                time.sleep(0.05)
                continue

            frame_count += 1
            fps_counter += 1

            # --- PROCESSAMENTO (YOLOv8-POSE) ---
            pose_data = pose_detector.update(frame)
            nose = pose_data["nose"]
            eye_l = pose_data["eye_l"]
            eye_r = pose_data["eye_r"]
            hands = pose_data["hands"]

            # --- ANÁLISE DE COMPORTAMENTOS ---
            phone, hands_busy = detect_objects(frame, model, nose, eye_l, eye_r, hands)
            head_turned = head_turn_detector.update(nose, eye_l, eye_r)
            eye_rubbing = eye_rub_detector.update(hands, eye_l, eye_r)
            phone_state = phone_detector.update(phone, nose, eye_l, eye_r)

            # Placeholders (lógica de fadiga removida conforme solicitado)
            stable_score = 100.0
            state = "MONITORANDO"

            # --- LOG DE DIAGNÓSTICO (a cada 30 frames) ---
            if frame_count % 30 == 0:
                # Calcula FPS aproximado
                elapsed = time.time() - fps_update_time
                if elapsed > 0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_update_time = time.time()
                else:
                    current_fps = 0

                # Build active-alert list from each detector's output
                alerts = []
                if head_turned:
                    alerts.append("HEAD TURN")
                if eye_rubbing:
                    alerts.append("EYE RUBBING")
                if phone is not None:
                    alerts.append("PHONE DETECTED")
                if phone_state == "EM LIGACAO":
                    alerts.append("ON CALL")
                elif phone_state == "USANDO CELULAR":
                    alerts.append("PHONE IN USE")
                if hands_busy:
                    alerts.append("HANDS BUSY")

                alert_str = ", ".join(alerts) if alerts else "none"

                print(f"[{frame_count:06d}] FPS: {current_fps:.1f} | "
                      f"Face: {'YES' if nose is not None else 'NO ':>3} | "
                      f"Hands: {len(hands)} | "
                      f"Phone: {'YES' if phone is not None else 'NO ':>3} | "
                      f"Score: {stable_score:5.1f} | "
                      f"State: {state} | "
                      f"Alerts: {alert_str}")

            # --- EXIBIÇÃO GRÁFICA (opcional, apenas em desktop) ---
            if SHOW_DISPLAY:
                display_frame = draw_status(frame, state, stable_score)
                cv2.imshow("Monitoramento DrowsyDriving", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏸️  Interrompido pelo usuário (tecla Q)")
                    break

            # Controle de FPS para não sobrecarregar o Pi 5
            frame_time = time.time() - frame_start
            if frame_time < 0.033:  # Limita a ~30 FPS máximo
                time.sleep(0.033 - frame_time)

    except KeyboardInterrupt:
        print("\n Interrompido pelo usuário (Ctrl+C)")

    except Exception as e:
        print(f"\n Erro durante execução: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- LIMPEZA DE RECURSOS ---
        print("\n🧹 Limpando recursos...")
        if picam2 is not None:
            picam2.stop()
            print("   ✓ Picamera2 liberada")
        if cap_opencv is not None:
            cap_opencv.release()
            print("   ✓ OpenCV liberado")
        cv2.destroyAllWindows()
        print("\n✅ Programa encerrado com sucesso!")

if __name__ == "__main__":
    main()
