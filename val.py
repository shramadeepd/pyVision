from ultralytics import YOLO


model = YOLO("./runs/detect/train10/weights/best.pt")


metrics = model.val()
print(metrics.box.map)  # map50-95