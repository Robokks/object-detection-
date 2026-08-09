"""Standalone detection script — no server, no GUI.

Edit the settings below and run this file (PyCharm's ▶ button next to
`if __name__ == "__main__":` works, or `python scripts/detect_to_json.py`
from a terminal in `backend/`). It loads a YOLO/YOLO-seg/SAM checkpoint,
runs it on an image (or every image in a folder), and writes a JSON file
with the same detail as the desktop app's Detect results table: class,
confidence, center position, orientation angle, left/right/top/bottom edge
midpoints, and the bounding box.

Command-line flags override the settings below if given, e.g.:
    python scripts/detect_to_json.py --weights best.pt --image photo.jpg
Run with --help to see all of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- edit these, or override with command-line flags ----------------------
WEIGHTS_PATH = "best.pt"  # path to a .pt checkpoint (plain YOLO, YOLO-seg, or SAM)
IMAGE_PATH = "image.jpg"  # a single image, or a folder of images
TASK = "detect"  # "detect" (boxes only), "segment" (YOLO-seg, masks), or "sam"
CONFIDENCE = 0.25
OUTPUT_PATH = "detections.json"
# -----------------------------------------------------------------------

from app.services import detect_service, report_service  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_model(weights_path: Path, task: str):
    if task == "sam":
        from ultralytics import SAM

        return SAM(str(weights_path))
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _detect_one(model, task: str, image_path: Path, confidence: float) -> dict:
    result = detect_service.run_detection_with_model(model, task, image_path.read_bytes(), confidence)
    return {
        "image": str(image_path),
        "image_width": result.image_width,
        "image_height": result.image_height,
        "detections": [report_service.shape_to_report(b) for b in result.boxes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="path to a .pt checkpoint")
    parser.add_argument("--image", default=IMAGE_PATH, help="image file, or a folder of images")
    parser.add_argument("--task", default=TASK, choices=["detect", "segment", "sam"])
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--output", default=OUTPUT_PATH, help="where to write the JSON results")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    image_path = Path(args.image)
    if not weights_path.exists():
        sys.exit(f"Weights file not found: {weights_path}")
    if not image_path.exists():
        sys.exit(f"Image path not found: {image_path}")

    print(f"Loading {weights_path} as a '{args.task}' model...")
    model = _load_model(weights_path, args.task)

    if image_path.is_dir():
        images = sorted(p for p in image_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            sys.exit(f"No images found in {image_path}")
        output = {
            "model": {"weights": str(weights_path), "task": args.task},
            "confidence_threshold": args.confidence,
            "images": [],
        }
        for img in images:
            print(f"Detecting on {img.name}...")
            entry = _detect_one(model, args.task, img, args.confidence)
            output["images"].append(entry)
            print(f"  {len(entry['detections'])} object(s) found")
    else:
        print(f"Detecting on {image_path.name}...")
        entry = _detect_one(model, args.task, image_path, args.confidence)
        output = {
            "model": {"weights": str(weights_path), "task": args.task},
            "confidence_threshold": args.confidence,
            **entry,
        }
        print(f"{len(entry['detections'])} object(s) found")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
