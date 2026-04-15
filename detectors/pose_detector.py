import cv2
from detectors.yolo_onnx import YOLOOnnx

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.onnx'):
        self.model = YOLOOnnx(model_path)

    def update(self, frame):
        # Inferência leve para RPi 5
        results = self.model(frame, imgsz=320, conf=0.5, verbose=False)

        data = {
            "nose": None, "eye_l": None, "eye_r": None,
            "hands": []
        }

        for r in results:
            if r.keypoints is not None:
                # Pegamos o primeiro esqueleto detectado (o motorista)
                kp = r.keypoints.xy.cpu().numpy()[0]

                # Mapeamento YOLOv8-Pose: 0:Nariz, 1:Olho_E, 2:Olho_D, 9:Pulso_E, 10:Pulso_D
                if len(kp) > 10:
                    data["nose"] = kp[0] if kp[0].any() else None
                    data["eye_l"] = kp[1] if kp[1].any() else None
                    data["eye_r"] = kp[2] if kp[2].any() else None

                    # Adiciona pulsos como mãos
                    if kp[9].any(): data["hands"].append(kp[9])
                    if kp[10].any(): data["hands"].append(kp[10])

        return data
