from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

# Load a model
model = YOLO("/home/coder/data/recap193/runs/detect/train2/weights/last.pt")
benchmark(model="./runs/detect/train10/weights/best.pt", data="dataset_path.yaml", imgsz=640, half=False,  format="onnx")