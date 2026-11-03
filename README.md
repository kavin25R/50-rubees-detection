# ₹50 Indian Currency Note Detection (YOLOv8)

This is a complete **starter project** for training a YOLOv8 object detector that detects ₹50 notes only.

## Important
This ZIP contains the project code and folder structure, **not a ready-made image dataset**. I cannot honestly claim to include real note images that have not been downloaded and checked. Add your own legally obtained images and annotate each ₹50 note with a bounding box before training.

## 1. Install
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Add and annotate images
Put images into:
- `dataset/images/train`
- `dataset/images/val`
- `dataset/images/test`

Use a tool such as LabelImg or CVAT to draw a bounding box around each ₹50 note. Export labels in **YOLO format**. For every image `abc.jpg`, create `dataset/labels/<same-split>/abc.txt`.

Each label line must be:
```text
0 x_center y_center width height
```
Coordinates must be normalized from 0 to 1. Class `0` is `50_rupees_note`.

Keep all denominations other than ₹50 out of the positive class. If you want the model to reject other notes, include images of other denominations with empty label files.

## 3. Dataset layout
```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## 4. Train
```bash
python train.py
```
The best model is saved under `runs/50_rupees_detection/weights/best.pt`.

## 5. Detect an image
```bash
python detect.py --source path/to/image.jpg
```

## 6. Live webcam
```bash
python detect.py --source 0
```

## 7. Evaluate
```bash
python evaluate.py
```

## Limitations
- This detects the visual presence of a ₹50 note; it does **not** verify whether a note is genuine or counterfeit.
- Model performance depends on image variety, annotation quality, and separation of train/validation/test images.
- Do not put near-duplicate frames of the same photo in different splits.
