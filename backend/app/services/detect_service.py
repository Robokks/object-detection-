import io

from PIL import Image

from app.schemas import DetectionResult, Shape
from app.services import model_service

# SAM's automatic "segment everything" mode can return dozens/hundreds of
# masks for a busy image; cap it so the response stays usable.
MAX_SAM_INSTANCES = 50


def _polygon_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def run_detection(model_id: str, image_bytes: bytes, confidence: float) -> DetectionResult:
    model = model_service.load_model(model_id)
    _, task = model_service.resolve(model_id)
    return run_detection_with_model(model, task, image_bytes, confidence)


def run_detection_with_model(model, task: str, image_bytes: bytes, confidence: float) -> DetectionResult:
    """Same as `run_detection`, but takes an already-loaded Ultralytics model
    and its task instead of a model id registered with `model_service` — for
    callers (e.g. the standalone `scripts/detect_to_json.py`) that load a
    checkpoint directly from a file path rather than through the app's model
    registry.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    results = model.predict(source=image, conf=confidence, verbose=False)
    result = results[0]
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    shapes: list[Shape] = []

    if masks is not None:
        polygons = [poly.tolist() for poly in masks.xy]
        for i, points in enumerate(polygons):
            if len(points) < 3:
                continue
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i].item())
                class_idx = int(boxes.cls[i].item())
            else:
                x1, y1, w, h = _polygon_bbox(points)
                x2, y2 = x1 + w, y1 + h
                conf, class_idx = None, None

            # SAM is class-agnostic (no object categories), unlike YOLO-seg.
            class_name = "object" if task == "sam" else names.get(class_idx, str(class_idx))
            shapes.append(
                Shape(
                    class_name=class_name,
                    confidence=conf,
                    shape="polygon",
                    points=points,
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                )
            )

        if task == "sam" and len(shapes) > MAX_SAM_INSTANCES:
            shapes.sort(key=lambda s: s.width * s.height, reverse=True)
            shapes = shapes[:MAX_SAM_INSTANCES]

    elif boxes is not None:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            class_idx = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            shapes.append(
                Shape(
                    class_name=names.get(class_idx, str(class_idx)),
                    confidence=conf,
                    shape="box",
                    points=[],
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                )
            )

    return DetectionResult(image_width=width, image_height=height, boxes=shapes)
