from ultralytics import YOLO

model = YOLO("yolov8s.yaml")

model.train(data="data.yaml", epochs=2, batch = 32, workers=8, optimizer="Adam", imgsz = 416)
