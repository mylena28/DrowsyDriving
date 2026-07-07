"""Camera bootstrap: Picamera2 (Pi 5 CSI) with an OpenCV (USB webcam) fallback.

OpenCV cannot read a CSI camera directly on the Pi 5, so Picamera2 is
mandatory there — see README "Architecture / Key design decisions".
"""

import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

IS_RASPBERRY_PI = (
    os.path.exists("/usr/bin/rpicam-hello") or os.path.exists("/usr/bin/libcamera-hello")
)


class CameraUnavailableError(RuntimeError):
    """No capture backend (Picamera2 or OpenCV) could produce a frame."""


class CameraSource:
    """Uniform read()/release() over whichever backend actually opened."""

    def __init__(
        self,
        backend: str,
        read: Callable[[], Optional[np.ndarray]],
        release: Callable[[], None],
    ):
        self.backend = backend
        self._read = read
        self._release = release

    def read(self) -> Optional[np.ndarray]:
        return self._read()

    def release(self) -> None:
        self._release()


def open_camera(camera_num: int = 1) -> CameraSource:
    """Opens Picamera2 on a detected Raspberry Pi, else falls back to OpenCV.

    Raises CameraUnavailableError with a diagnostic message if neither
    backend can produce frames.
    """
    if IS_RASPBERRY_PI:
        picam2 = _open_picamera2(camera_num)
        if picam2 is not None:
            return CameraSource("picamera2", picam2.capture_array, picam2.stop)
        print("\n⚠️  Picamera2 falhou. Tentando fallback OpenCV...")

    cap = _open_opencv_fallback()
    if cap is not None:
        return CameraSource("opencv", lambda: cap.read()[1], cap.release)

    raise CameraUnavailableError(
        "nenhuma câmera disponível: Picamera2 (Pi) falhou/indisponível e nenhum "
        "índice OpenCV em [0, 1, 2] produziu frames. Verifique "
        "`libcamera-hello --list-cameras` (CSI) ou `lsusb` (webcam USB)."
    )


def _open_picamera2(camera_num: int):
    """Returns a started Picamera2 instance, or None if import/init fails."""
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        print(f"❌ Picamera2 não instalada: {exc}")
        print("   Instale com: sudo apt install python3-picamera2")
        return None

    try:
        picam2 = _configure_and_start(Picamera2, camera_num)
        _check_first_frame(picam2, camera_num)
    except Exception as exc:
        print(f"❌ Erro ao inicializar Picamera2 (camera_num={camera_num}): {exc}")
        print("\n📋 Diagnóstico rápido:")
        print("   1. Verifique o cabo da câmera: libcamera-hello --list-cameras")
        print("   2. Teste a câmera manualmente: libcamera-hello -t 5")
        print("   3. Reinicie o serviço de câmera: sudo systemctl restart rpicam-apps")
        return None

    print("✅ Picamera2 inicializada com sucesso!")
    print("   Resolução: 640x480 | Formato: BGR888 | FPS: 30")
    return picam2


def _configure_and_start(picamera2_cls, camera_num: int):
    picam2 = picamera2_cls(camera_num=camera_num)
    config = picam2.create_video_configuration(
        main={
            "format": "BGR888",  # matches OpenCV's native byte order
            "size": (640, 480),  # balanced resolution for Pi 5 performance
        },
        controls={
            "FrameRate": 30.0,   # cap FPS for stability
            "AwbMode": 0,        # auto white balance
            "AnalogueGain": 1.0, # default gain
        },
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)  # camera needs to settle before the first real capture
    return picam2


def _check_first_frame(picam2, camera_num: int) -> None:
    test_frame = picam2.capture_array()
    if test_frame is None or test_frame.size == 0:
        picam2.stop()
        raise CameraUnavailableError(
            f"Picamera2 capture_array() retornou frame vazio (camera_num={camera_num})"
        )


def _open_opencv_fallback() -> Optional[cv2.VideoCapture]:
    """Tries camera indices 0-2, returns the first that produces a frame."""
    for camera_index in [0, 1, 2]:
        cap = _try_camera_index(camera_index)
        if cap is not None:
            return cap
    print("❌ Nenhuma câmera encontrada no sistema")
    return None


def _try_camera_index(camera_index: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None

    print(f"✅ Câmera encontrada no índice {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        print("✅ Fallback OpenCV funcionando")
        return cap

    print(f"⚠️  Câmera índice {camera_index} não produz frames")
    cap.release()
    return None
