"""
Pin / Object Detection with YOLO (Ultralytics)
===============================================
Runs a YOLO or YOLO-seg checkpoint — your own trained model, or any
pretrained one — on an image and writes a JSON file with each detection's
class, confidence, center position, orientation angle, and bounding box.
Also saves an annotated copy of the image for a quick visual check.

Fully self-contained: nothing else from this repo is required, just:
    pip install ultralytics opencv-python numpy

Run:
    python pin_detect_json.py image.png
    python pin_detect_json.py folder/

Weights: set MODEL below to your trained checkpoint's path (e.g. the
best.pt you get after training/fine-tuning, from this app's Train tab or
the Colab notebook). A plain Ultralytics name like "yolov8n.pt" also works
and auto-downloads on first run.

Tuning:
    CONF   : confidence threshold (lower = more, weaker detections)
    IOU    : overlap threshold used to merge duplicate boxes
    IMGSZ  : inference resolution
"""

import glob
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "best.pt"  # your trained checkpoint, or e.g. "yolov8n.pt" to auto-download
CONF = 0.25
IOU = 0.7
IMGSZ = 640
OUTPUT_JSON = "detections.json"  # combined results for everything processed this run

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL)
    return _model


def _angle_from_points(points):
    """Long-axis orientation in degrees (0=horizontal, 90=vertical), fit
    from a mask's outline with a minimum-area rectangle. None for a plain
    box result — there's no outline to fit an angle from.
    """
    if not points or len(points) < 3:
        return None
    pts = np.array(points, dtype=np.float32)
    (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
    theta = angle if rw >= rh else angle + 90
    return round(float(theta % 180), 1)


def detect(path):
    img = cv2.imread(path)
    if img is None:
        print(f"cannot read {path}")
        return None
    h_img, w_img = img.shape[:2]

    model = get_model()
    res = model(path, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)[0]
    names = res.names or {}
    boxes = res.boxes
    masks = res.masks

    detections = []
    vis = img.copy()

    n = len(boxes) if boxes is not None else 0
    for i in range(n):
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
        conf = float(boxes.conf[i].item())
        class_idx = int(boxes.cls[i].item())
        class_name = names.get(class_idx, str(class_idx))
        points = masks.xy[i].tolist() if masks is not None and i < len(masks.xy) else []

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        angle = _angle_from_points(points)

        det = {
            "class_name": class_name,
            "shape": "mask" if angle is not None else "box",
            "confidence": round(conf, 4),
            "position": {"x": round(cx, 1), "y": round(cy, 1)},
            "angle_degrees": angle,
            "left": {"x": round(x1, 1), "y": round(cy, 1)},
            "right": {"x": round(x2, 1), "y": round(cy, 1)},
            "top": {"x": round(cx, 1), "y": round(y1, 1)},
            "bottom": {"x": round(cx, 1), "y": round(y2, 1)},
            "box": {"x": round(x1, 1), "y": round(y1, 1), "width": round(w, 1), "height": round(h, 1)},
        }
        detections.append(det)

        if points:
            cv2.polylines(vis, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
        else:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        label = f"{class_name} {conf * 100:.0f}%"
        cv2.putText(
            vis, label, (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    out_img = os.path.splitext(path)[0] + "_detections.png"
    cv2.imwrite(out_img, vis)

    print(f"\n{os.path.basename(path)} — {len(detections)} detection(s)")
    for d in detections:
        angle_txt = f"  angle={d['angle_degrees']}°" if d["angle_degrees"] is not None else ""
        print(
            f"  {d['class_name']} ({d['confidence'] * 100:.0f}%)  "
            f"center=({d['position']['x']:.0f},{d['position']['y']:.0f}){angle_txt}"
        )
    print(f"  annotated -> {out_img}")

    return {"image": path, "image_width": w_img, "image_height": h_img, "detections": detections}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "."

    if os.path.isdir(arg):
        entries = []
        for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
            for p in sorted(glob.glob(os.path.join(arg, f"*.{ext}"))):
                entry = detect(p)
                if entry:
                    entries.append(entry)
        output = {"model": MODEL, "task": get_model().task, "confidence_threshold": CONF, "images": entries}
    else:
        entry = detect(arg) or {"image": arg, "image_width": 0, "image_height": 0, "detections": []}
        output = {"model": MODEL, "task": get_model().task, "confidence_threshold": CONF, **entry}

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")
