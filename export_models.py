from ultralytics import YOLO

# imgsz=320 cuts inference time roughly in half vs the default 640,
# with minimal accuracy loss for close-range driver monitoring.
print("Exportando yolov8n.pt para ONNX (320x320)...")
YOLO("yolov8n.pt").export(format="onnx", imgsz=320)

print("Exportando yolov8n-pose.pt para ONNX (320x320)...")
YOLO("yolov8n-pose.pt").export(format="onnx", imgsz=320)

print("Modelos ONNX exportados com sucesso!")
