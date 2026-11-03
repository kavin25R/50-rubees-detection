import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Detect Indian ₹50 notes")
    parser.add_argument("--source", default="0", help="Image/video path or webcam index (default: 0)")
    parser.add_argument("--weights", default="runs/50_rupees_detection/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.50)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model not found: {weights}. Train first using: python train.py"
        )

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(str(weights))
    model.predict(source=source, conf=args.conf, show=True, save=True)

if __name__ == "__main__":
    main()
