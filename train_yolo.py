from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="defect_data/data.yaml",
    epochs=50,
    batch=8,
    workers=4,
    imgsz=640,
    device=0,
)
