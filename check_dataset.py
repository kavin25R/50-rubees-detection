from pathlib import Path

ROOT = Path("dataset")
SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

errors = []
for split in SPLITS:
    images_dir = ROOT / "images" / split
    labels_dir = ROOT / "labels" / split
    images = [p for p in images_dir.glob("*") if p.suffix.lower() in IMG_EXTS]

    if not images:
        errors.append(f"No images found in {images_dir}")
        continue

    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            errors.append(f"Missing label: {label}")
            continue
        for line_no, line in enumerate(label.read_text().splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label}:{line_no}: expected 5 values")
                continue
            try:
                cls, *coords = map(float, parts)
                if cls != 0 or any(v < 0 or v > 1 for v in coords):
                    errors.append(f"{label}:{line_no}: invalid class/coordinates")
            except ValueError:
                errors.append(f"{label}:{line_no}: non-numeric label")

if errors:
    print("Dataset check FAILED:")
    print("\n".join(errors))
    raise SystemExit(1)

print("Dataset structure and labels look valid.")
