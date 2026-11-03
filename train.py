from pathlib import Path
from ultralytics import YOLO

DATA = Path(__file__).resolve().parent / "data.yaml"

if not DATA.exists():
    raise FileNotFoundError(f"Missing dataset config: {DATA}")

model = YOLO("yolov8n.pt")
model.train(
    data=str(DATA),
    epochs=60,
    imgsz=640,
    batch=16,
    project="runs",
    name="50_rupees_detection",
    patience=15,
    pretrained=True,
)
print("Training finished. Best weights: runs/50_rupees_detection/weights/best.pt")
