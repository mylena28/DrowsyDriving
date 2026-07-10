import numpy as np

from detectors.yolo_onnx import YOLOOnnx


class PoseDetector:
    def __init__(self, model_path: str = "yolov8n-pose.onnx"):
        self.model = YOLOOnnx(model_path)

    def update(self, frame: np.ndarray) -> dict:
        results = self.model(frame, conf=0.5, verbose=False)  # modelo já exportado leve (320x320) para RPi 5
        for r in results:
            if r.keypoints is not None:
                kp = r.keypoints.xy.cpu().numpy()[0]  # primeiro esqueleto detectado (o motorista)
                if len(kp) > 10:
                    return _keypoints_to_data(kp)
        return _empty_pose_data()


def _empty_pose_data() -> dict:
    return {
        "nose": None, "eye_l": None, "eye_r": None,
        "shoulder_l": None, "shoulder_r": None,
        "hands": [],
        "hand_l": None, "hand_r": None,
    }


def _keypoints_to_data(kp: np.ndarray) -> dict:
    """Mapeamento YOLOv8-Pose: 0:Nariz, 1:Olho_E, 2:Olho_D, 5:Ombro_E, 6:Ombro_D, 9:Pulso_E, 10:Pulso_D"""
    data = _empty_pose_data()
    data["nose"] = kp[0] if kp[0].any() else None
    data["eye_l"] = kp[1] if kp[1].any() else None
    data["eye_r"] = kp[2] if kp[2].any() else None
    data["shoulder_l"] = kp[5] if kp[5].any() else None
    data["shoulder_r"] = kp[6] if kp[6].any() else None

    # Pulsos individualmente e como lista
    if kp[9].any():
        data["hand_l"] = kp[9]
        data["hands"].append(kp[9])
    if kp[10].any():
        data["hand_r"] = kp[10]
        data["hands"].append(kp[10])
    return data
