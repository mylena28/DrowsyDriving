from ultralytics import YOLO

print("Exportando yolov8n.pt para ONNX...")
YOLO("yolov8n.pt").export(format="onnx")

print("Exportando yolov8n-pose.pt para ONNX...")
YOLO("yolov8n-pose.pt").export(format="onnx")

print("Modelos ONNX exportados com sucesso!")
