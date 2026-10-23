"""Entry point: camera bootstrap + per-frame detection/alerting loop."""

import argparse
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2
import numpy as np

from behaviors.eye_rub import EyeRubDetector
from behaviors.hand_busy import HandBusyDetector
from behaviors.head_down import HeadDownDetector
from behaviors.head_turn import HeadTurnDetector
from behaviors.phone_usage import PhoneStateDetector
from behaviors.smoking_gesture import hand_near_mouth
from detectors.object_detector import detect_objects
from detectors.pose_detector import PoseDetector
from detectors.seatbelt_detector import SeatbeltDetector
from detectors.smoking_detector import SmokingDetector
from detectors.yolo_onnx import YOLOOnnx
from utils.camera import (
    CameraSource, CameraUnavailableError, DEFAULT_CAMERA_ID_CONTAINS,
    IS_RASPBERRY_PI, open_camera,
)
from utils.classifier import classify_from_score
from utils.display import draw_status
from utils.event_log import SnapshotLogger
from utils.geometry import lower_face_crop
from utils.risk import RiskTracker
from utils.video_writer import DebouncedRecorder, RiskClipRecorder

# Processing rate, not camera capture rate. Running every ONNX model flat-out
# pins the Pi 5; processing a handful of frames per second is plenty for
# drowsiness/distraction (events last seconds) and frees the CPU. Override per
# run with --fps or the PROCESSING_FPS env var. Lowering it stretches the
# real-time meaning of the frame-count cadences below, so retune if changed a lot.
DEFAULT_PROCESSING_FPS = 10
DETECT_EVERY_N = 3             # object detection cadence (phone/hands)
DETECT_SLOW_N = 15             # seatbelt/smoking cadence (changes slowly)
SMOKING_CONFIRM_N = 2          # consecutive positive hits required before the alert fires
DIAGNOSTIC_LOG_EVERY_N = 30
SNAPSHOT_COOLDOWN_SECONDS = 5.0

# Recording is opt-in (see --record). "off" writes no video; "risk" records a
# fixed clip around confirmed risk events; "presence" records while a face is
# visible (the legacy behaviour).
RECORD_MODES = ("off", "risk", "presence")
DEFAULT_RECORD_MODE = os.environ.get("RECORD_MODE", "off")
RECORD_CLIP_SECONDS = 10.0     # length of a risk clip, extended by overlapping events
RECORD_CONFIRM_FRAMES = 3      # consecutive risk frames before a clip starts (anti false-positive)

Recorder = Union[DebouncedRecorder, RiskClipRecorder]


class Detectors:
    """Bundles every model/behavior instance the frame loop needs.

    Built once, after the camera is confirmed working, and passed down
    explicitly to FrameProcessor rather than read from module globals.
    """

    def __init__(self):
        self.object_model = YOLOOnnx("yolov8n.onnx")
        self.pose = PoseDetector("yolov8n-pose.onnx")
        self.seatbelt = SeatbeltDetector()
        self.smoking = SmokingDetector()
        self.head_turn = HeadTurnDetector(frames_threshold=5)
        self.head_down = HeadDownDetector()
        self.eye_rub = EyeRubDetector(frames_threshold=5)
        self.phone = PhoneStateDetector(call_frames_threshold=8)
        self.hand_busy = HandBusyDetector()


class SlowDetectionState:
    """Holds the seatbelt/smoking readings between their slow-cadence
    (DETECT_SLOW_N-frame) detection passes."""

    # Trailing slow-cadence checks to look back for a hand-near-mouth gesture.
    # Wider than 1 tick on purpose: a real puff/exhale cycle has the hand away
    # from the mouth for much of the window, so requiring the gesture on the
    # exact same tick as the model hit would suppress true positives too.
    _GESTURE_WINDOW = 3

    def __init__(self):
        self.seatbelt_on = True   # assume wearing until first detection
        self.smoking = False
        self._smoking_hit_count = 0
        self._recent_hand_near_mouth: list[bool] = []

    def update(self, frame: np.ndarray, nose, eye_l, eye_r, hands, detectors: Detectors) -> None:
        self.seatbelt_on = detectors.seatbelt.detect(frame)
        face_crop = lower_face_crop(frame, nose, eye_l, eye_r)
        cigarette_seen = detectors.smoking.detect(face_crop)
        self._track_gesture(hand_near_mouth(hands, nose, eye_l, eye_r))

        if cigarette_seen and any(self._recent_hand_near_mouth):
            self._smoking_hit_count += 1
        else:
            self._smoking_hit_count = 0
        self.smoking = self._smoking_hit_count >= SMOKING_CONFIRM_N

    def _track_gesture(self, seen: bool) -> None:
        self._recent_hand_near_mouth.append(seen)
        self._recent_hand_near_mouth = self._recent_hand_near_mouth[-self._GESTURE_WINDOW:]


@dataclass
class FrameResult:
    frame_count: int
    nose: Optional[np.ndarray]
    stable_score: float
    state: str
    alerts: list = field(default_factory=list)
    display_alerts: list = field(default_factory=list)
    risk_debug: dict = field(default_factory=dict)


# (log_label, display_label) pairs, in priority order, for each alert condition.
def _active_alert_specs(
    phone, phone_state: str, head_turned: bool, head_down: bool, eye_rubbing: bool,
    hands_busy: bool, busy_count: int, seatbelt_on: bool, smoking: bool,
) -> list[tuple[str, str]]:
    specs = []
    if head_down:
        specs.append(("HEAD DOWN", "CABECA BAIXA"))
    if head_turned:
        specs.append(("HEAD TURN", "VIRANDO A CABECA"))
    if eye_rubbing:
        specs.append(("EYE RUBBING", "ESFREGANDO OS OLHOS"))
    if phone is not None:
        specs.append(("PHONE DETECTED", "CELULAR DETECTADO"))
    if phone_state == "EM LIGACAO":
        specs.append(("ON CALL", "EM LIGACAO"))
    elif phone_state == "USANDO CELULAR":
        specs.append(("PHONE IN USE", "USANDO CELULAR"))
    if hands_busy:
        specs.append(("HANDS BUSY", "MAOS OCUPADAS"))
    if busy_count == 1:
        specs.append(("1 MAO OCUPADA", "1 MAO LEVANTADA"))
    elif busy_count == 2:
        specs.append(("2 MAOS OCUPADAS", "2 MAOS LEVANTADAS"))
    if not seatbelt_on:
        specs.append(("NO SEATBELT", "SEM CINTO DE SEGURANCA"))
    if smoking:
        specs.append(("SMOKING", "FUMANDO"))
    return specs


class FrameProcessor:
    """Runs one frame through pose/object/behavior detection and risk
    scoring. Owns the inter-frame state that used to live as loop-local
    variables in main(): the last object-detection result (reused between
    DETECT_EVERY_N cadence steps) and the slow seatbelt/smoking readings.
    """

    def __init__(self, detectors: Detectors):
        self._detectors = detectors
        self._risk_tracker = RiskTracker()
        self._slow_state = SlowDetectionState()
        self._last_phone = None
        self._last_hands_busy = False
        self.frame_count = 0

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        self.frame_count += 1
        pose_data = self._detectors.pose.update(frame)
        nose, eye_l, eye_r = pose_data["nose"], pose_data["eye_l"], pose_data["eye_r"]
        shoulder_l, shoulder_r = pose_data["shoulder_l"], pose_data["shoulder_r"]
        hands, hand_l, hand_r = pose_data["hands"], pose_data["hand_l"], pose_data["hand_r"]

        phone, hands_busy = self._detect_phone_and_hands(frame, nose, eye_l, eye_r, hands)
        seatbelt_on, smoking = self._detect_seatbelt_and_smoking(frame, nose, eye_l, eye_r, hands)

        head_turned = self._detectors.head_turn.update(nose, eye_l, eye_r)
        head_down = self._detectors.head_down.update(nose, eye_l, eye_r, shoulder_l, shoulder_r)
        eye_rubbing = self._detectors.eye_rub.update(hands, eye_l, eye_r)
        phone_state = self._detectors.phone.update(phone, nose, eye_l, eye_r)
        busy_count = self._count_busy_hands(hand_l, hand_r, eye_l, eye_r, nose)

        events = _risk_events(phone_state, head_turned, head_down, eye_rubbing,
                               busy_count, hands_busy, seatbelt_on, smoking)
        stable_score = self._risk_tracker.update(events, timestamp)
        state = classify_from_score(stable_score)

        specs = _active_alert_specs(phone, phone_state, head_turned, head_down, eye_rubbing,
                                     hands_busy, busy_count, seatbelt_on, smoking)
        return FrameResult(
            frame_count=self.frame_count,
            nose=nose,
            stable_score=stable_score,
            state=state,
            alerts=[log for log, _ in specs],
            display_alerts=[disp for _, disp in specs],
            risk_debug=self._risk_tracker.debug_info(),
        )

    def _detect_phone_and_hands(self, frame, nose, eye_l, eye_r, hands):
        """Object detection runs every DETECT_EVERY_N frames; phone/hand
        state changes slowly enough that stale reads are fine in between."""
        if self.frame_count % DETECT_EVERY_N == 0:
            self._last_phone, self._last_hands_busy = detect_objects(
                frame, self._detectors.object_model, nose, eye_l, eye_r, hands)
        return self._last_phone, self._last_hands_busy

    def _detect_seatbelt_and_smoking(self, frame, nose, eye_l, eye_r, hands):
        """Seatbelt/smoking are checked at a slower cadence, only when a
        face is visible; assume compliant/absent while no face is found."""
        if nose is not None and self.frame_count % DETECT_SLOW_N == 0:
            self._slow_state.update(frame, nose, eye_l, eye_r, hands, self._detectors)
        if nose is None:
            return True, False
        return self._slow_state.seatbelt_on, self._slow_state.smoking

    def _count_busy_hands(self, hand_l, hand_r, eye_l, eye_r, nose) -> int:
        hand_l_busy, hand_r_busy = self._detectors.hand_busy.update(
            hand_l, hand_r, eye_l, eye_r, nose)
        return int(hand_l_busy) + int(hand_r_busy)


def _risk_events(
    phone_state: str, head_turned: bool, head_down: bool, eye_rubbing: bool,
    busy_count: int, hands_busy: bool, seatbelt_on: bool, smoking: bool,
) -> dict:
    return {
        "phone_call": phone_state == "EM LIGACAO",
        "phone_use": phone_state == "USANDO CELULAR",
        "head_turn": head_turned,
        "head_down": head_down,
        "eye_rub": eye_rubbing,
        "one_hand_raised": busy_count == 1,
        "two_hands_raised": busy_count == 2,
        "hands_busy_object": hands_busy,
        "no_seatbelt": not seatbelt_on,
        "smoking": smoking,
    }


class FpsCounter:
    """Tracks frames-per-second averaged over a window, reset on sampling."""

    def __init__(self):
        self._count = 0
        self._window_start = time.time()

    def tick(self) -> None:
        self._count += 1

    def sample_and_reset(self) -> float:
        elapsed = time.time() - self._window_start
        fps = self._count / elapsed if elapsed > 0 else 0.0
        self._count = 0
        self._window_start = time.time()
        return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DrowsyDriving Monitor")
    parser.add_argument(
        "--venv", action="store_true",
        help="Modo venv: abre janela com câmera e diagnóstico em texto (sem caixas)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Índice explícito da câmera Picamera2. Se omitido, seleciona pela "
             "Id (--camera-id-contains).",
    )
    parser.add_argument(
        "--camera-id-contains", type=str, default=None,
        help=f"Trecho da Id da câmera CSI a usar (padrão: '{DEFAULT_CAMERA_ID_CONTAINS}', "
             "a do DrowsyDriving). Ignorado se --camera for passado.",
    )
    parser.add_argument(
        "--fps", type=float,
        default=float(os.environ.get("PROCESSING_FPS", DEFAULT_PROCESSING_FPS)),
        help=f"Quadros processados por segundo (padrão: {DEFAULT_PROCESSING_FPS}). "
             "Menor = menos CPU.",
    )
    parser.add_argument(
        "--record", choices=RECORD_MODES, default=DEFAULT_RECORD_MODE,
        help="Gravação de vídeo: off (padrão), risk (clipe fixo em evento de "
             "risco) ou presence (enquanto houver rosto).",
    )
    return parser.parse_args()


def _print_diagnostics(result: FrameResult, current_fps: float) -> None:
    alert_str = ", ".join(result.alerts) if result.alerts else "none"
    active_debug = {
        k: v for k, v in result.risk_debug.items()
        if v["active_secs"] > 0 or v["recent_count"] > 0
    }
    print(f"[{result.frame_count:06d}] FPS: {current_fps:.1f} | "
          f"Face: {'YES' if result.nose is not None else 'NO ':>3} | "
          f"Score: {result.stable_score:5.1f} | "
          f"State: {result.state} | "
          f"Alerts: {alert_str} | "
          f"Risk detail: {active_debug}")


def _show_frame(frame: np.ndarray, result: FrameResult) -> bool:
    """Renders the debug overlay window. Returns False if the user pressed 'q'."""
    display_frame = draw_status(frame, result.state, result.display_alerts)
    cv2.imshow("Monitoramento DrowsyDriving", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n⏸️  Interrompido pelo usuário (tecla Q)")
        return False
    return True


def _throttle_to_target_fps(frame_start: float, target_frame_seconds: float) -> None:
    frame_time = time.time() - frame_start
    if frame_time < target_frame_seconds:
        time.sleep(target_frame_seconds - frame_time)


def _run_one_frame(
    camera: CameraSource,
    frame_processor: FrameProcessor,
    fps_counter: FpsCounter,
    snapshot_logger: SnapshotLogger,
    recorder: Optional[Recorder],
    show_display: bool,
    target_frame_seconds: float,
) -> bool:
    """Processes one frame. Returns False if the loop should stop (the user
    pressed 'q' in the display window)."""
    frame_start = time.time()
    frame = camera.read()
    if frame is None:
        print("⚠️  Frame perdido. Tentando recuperar...")
        time.sleep(0.05)
        return True

    fps_counter.tick()
    result = frame_processor.process(frame, frame_start)
    _record_and_log(frame, result, recorder, snapshot_logger)

    if result.frame_count % DIAGNOSTIC_LOG_EVERY_N == 0:
        _print_diagnostics(result, fps_counter.sample_and_reset())

    if show_display and not _show_frame(frame, result):
        return False

    _throttle_to_target_fps(frame_start, target_frame_seconds)
    return True


def _record_and_log(
    frame: np.ndarray,
    result: FrameResult,
    recorder: Optional[Recorder],
    snapshot_logger: SnapshotLogger,
) -> None:
    if recorder is not None:
        # A risk event = any active alert this frame. The recorder decides what
        # to do with each signal (presence mode uses face_visible, risk mode
        # uses risk_active).
        recorder.update(
            frame,
            face_visible=result.nose is not None,
            risk_active=bool(result.alerts),
        )
    snapshot_logger.maybe_log(
        frame, result.alerts, result.stable_score, result.state, result.risk_debug)


def _cleanup(camera: CameraSource, recorder: Optional[Recorder]) -> None:
    print("\n🧹 Limpando recursos...")
    if recorder is not None and recorder.isOpened():
        video_path = recorder.video_path
        recorder.release()
        print(f"   ✓ Vídeo salvo: {video_path}")
    camera.release()
    print(f"   ✓ Câmera liberada ({camera.backend})")
    cv2.destroyAllWindows()
    print("\n✅ Programa encerrado com sucesso!")


def _build_session(
    args: argparse.Namespace,
) -> tuple[FrameProcessor, FpsCounter, SnapshotLogger, Optional[Recorder]]:
    frame_processor = FrameProcessor(Detectors())
    fps_counter = FpsCounter()

    snapshot_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_logger = SnapshotLogger(snapshot_dir, cooldown_secs=SNAPSHOT_COOLDOWN_SECONDS)
    recorder = _build_recorder(args.record, snapshot_dir, args.fps)
    return frame_processor, fps_counter, snapshot_logger, recorder


def _build_recorder(mode: str, snapshot_dir: str, fps: float) -> Optional[Recorder]:
    if mode == "risk":
        return RiskClipRecorder(
            snapshot_dir, fps=fps,
            clip_seconds=RECORD_CLIP_SECONDS, confirm_frames=RECORD_CONFIRM_FRAMES)
    if mode == "presence":
        return DebouncedRecorder(snapshot_dir, fps=fps)
    return None


def run_monitoring_loop(camera: CameraSource, args: argparse.Namespace) -> None:
    frame_processor, fps_counter, snapshot_logger, recorder = _build_session(args)
    show_display = args.venv or ("DISPLAY" in os.environ and not IS_RASPBERRY_PI)
    target_frame_seconds = 1.0 / args.fps if args.fps > 0 else 0.0
    if args.venv:
        print("🖥️  Modo venv ativado: exibindo câmera com diagnóstico em tela")

    print("\n" + "=" * 50)
    print("🚀 MONITORAMENTO INICIADO")
    print("=" * 50)
    print(f"⚙️  {args.fps:.0f} FPS alvo | gravação: {args.record}")
    print("📊 Pressione Ctrl+C para encerrar\n")

    try:
        while _run_one_frame(camera, frame_processor, fps_counter,
                             snapshot_logger, recorder, show_display,
                             target_frame_seconds):
            pass
    except KeyboardInterrupt:
        print("\n Interrompido pelo usuário (Ctrl+C)")
    except Exception as exc:
        print(f"\n Erro durante execução: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _cleanup(camera, recorder)


def main() -> None:
    args = parse_args()
    try:
        camera = open_camera(
            camera_num=args.camera, camera_id_contains=args.camera_id_contains)
    except CameraUnavailableError as exc:
        print(f"\n❌ ERRO CRÍTICO: {exc}")
        return

    print(f"📷 Câmera pronta (backend={camera.backend})")
    run_monitoring_loop(camera, args)


if __name__ == "__main__":
    main()
