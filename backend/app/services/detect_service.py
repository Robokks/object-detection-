import io

from PIL import Image

from app.schemas import BoundingBox, DetectionResult
from app.services import model_service


def run_detection(model_id: str, image_bytes: bytes, confidence: float) -> DetectionResult:
    model = model_service.load_model(model_id)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    results = model.predict(source=image, conf=confidence, verbose=False)
    result = results[0]

    boxes: list[BoundingBox] = []
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        class_idx = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        boxes.append(
            BoundingBox(
                class_name=names.get(class_idx, str(class_idx)),
                confidence=conf,
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
            )
        )

    return DetectionResult(image_width=width, image_height=height, boxes=boxes)
