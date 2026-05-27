"""
Detects seatbelt presence using a custom YOLO .onnx model.

Any detection above the confidence threshold means the seatbelt is ON.
No detection (while a face is visible) triggers the NO SEATBELT alert.

Class names depend on training — we only check whether anything was detected,
so the COCO name table in YOLOOnnx is not used here.
"""

import os
from detectors.yolo_onnx import YOLOOnnx

_BASE = os.path.dirname(os.path.abspath(__file__))


class SeatbeltDetector:
    def __init__(
        self,
        model_path: str = os.path.join(_BASE, "seatbelt.onnx"),
        conf: float = 0.45,
    ):
        self.model = YOLOOnnx(model_path)
        self.conf = conf

    def detect(self, frame) -> bool:
        """
        Returns True if seatbelt is detected in the frame, False otherwise.
        """
        for r in self.model(frame, conf=self.conf):
            if len(r.boxes) > 0:
                return True
        return False
