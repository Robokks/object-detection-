import json
import threading

from app.config import DEFAULT_PRETRAINED_WEIGHTS, WEIGHTS_DIR
from app.schemas import ModelInfo

PRETRAINED_MODELS = {
    "pretrained:yolov8n": "yolov8n.pt",
    "pretrained:yolov8s": "yolov8s.pt",
}

_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


class ModelError(ValueError):
    pass


def list_models() -> list[ModelInfo]:
    models = []
    for model_id, checkpoint in PRETRAINED_MODELS.items():
        models.append(
            ModelInfo(id=model_id, label=f"{checkpoint} (pretrained, COCO 80 classes)", source="pretrained")
        )

    for weights_file in sorted(WEIGHTS_DIR.glob("*.pt")):
        run_name = weights_file.stem
        classes_file = WEIGHTS_DIR / f"{run_name}.classes.json"
        classes = json.loads(classes_file.read_text()) if classes_file.exists() else []
        models.append(
            ModelInfo(id=f"trained:{run_name}", label=f"{run_name} (custom trained)", source="trained", classes=classes)
        )
    return models


def resolve_checkpoint(model_id: str) -> str:
    if model_id in PRETRAINED_MODELS:
        return PRETRAINED_MODELS[model_id]
    if model_id.startswith("trained:"):
        run_name = model_id.split(":", 1)[1]
        weights_file = WEIGHTS_DIR / f"{run_name}.pt"
        if not weights_file.exists():
            raise ModelError(f"Trained model '{run_name}' not found")
        return str(weights_file)
    raise ModelError(f"Unknown model '{model_id}'")


def load_model(model_id: str):
    from ultralytics import YOLO  # heavy import, deferred

    with _cache_lock:
        if model_id not in _model_cache:
            checkpoint = resolve_checkpoint(model_id)
            _model_cache[model_id] = YOLO(checkpoint)
        return _model_cache[model_id]
