"""Fine-tune YOLOv8n on the casting-defect dataset.

Outputs to runs/detect/defect_v1/ (best.pt, last.pt, results.csv, evaluation plots).
"""

from pathlib import Path

from ultralytics import YOLO, settings

REPO = Path(__file__).parent.resolve()
settings.update({"datasets_dir": str(REPO)})

model = YOLO("yolov8n.pt")

model.train(
    data=str(REPO / "defect_data" / "data.yaml"),
    epochs=100,
    patience=20,
    batch=8,
    workers=4,
    imgsz=640,
    device=0,
    seed=0,
    name="defect_v1",
    plots=True,
)
