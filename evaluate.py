from pathlib import Path
from ultralytics import YOLO

weights = Path("runs/50_rupees_detection/weights/best.pt")
if not weights.exists():
    raise FileNotFoundError("Train the model first: python train.py")

model = YOLO(str(weights))
metrics = model.val(data="data.yaml", split="test")
print(metrics)
