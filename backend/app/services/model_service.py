import json
import re
import threading
import uuid
from pathlib import Path

from app.config import IMPORTED_WEIGHTS_DIR, TRAINED_WEIGHTS_DIR
from app.schemas import ModelInfo, ModelTask

# id -> (checkpoint name ultralytics will auto-download, label, task)
PRETRAINED_CATALOG: dict[str, tuple[str, str, ModelTask]] = {
    "pretrained:yolov8n": ("yolov8n.pt", "YOLOv8n (pretrained, COCO 80 classes, detection)", "detect"),
    "pretrained:yolov8s": ("yolov8s.pt", "YOLOv8s (pretrained, COCO 80 classes, detection)", "detect"),
    "pretrained:yolov8m": ("yolov8m.pt", "YOLOv8m (pretrained, COCO 80 classes, detection, more accurate)", "detect"),
    "pretrained:yolov8n-seg": (
        "yolov8n-seg.pt",
        "YOLOv8n-seg (pretrained, COCO 80 classes, segmentation)",
        "segment",
    ),
    "pretrained:yolov8s-seg": (
        "yolov8s-seg.pt",
        "YOLOv8s-seg (pretrained, COCO 80 classes, segmentation)",
        "segment",
    ),
    "pretrained:sam_b": (
        "sam_b.pt",
        "SAM (Segment Anything, base) — class-agnostic, segments every object",
        "sam",
    ),
}

_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


class ModelError(ValueError):
    pass


def _sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")
    return slug or uuid.uuid4().hex[:8]


def _scan_weights_dir(directory: Path, source: str) -> list[ModelInfo]:
    models = []
    for weights_file in sorted(directory.glob("*.pt")):
        run_name = weights_file.stem
        meta_file = directory / f"{run_name}.meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError:
            # a truncated/empty .meta.json (interrupted write, manual edit, etc.)
            # shouldn't take every model in the list down with it — skip just this one.
            continue
        models.append(
            ModelInfo(
                id=f"{source}:{run_name}",
                label=meta.get("label", run_name),
                source=source,
                task=meta.get("task", "detect"),
                classes=meta.get("classes", []),
            )
        )
    return models


def list_models() -> list[ModelInfo]:
    models = [
        ModelInfo(id=model_id, label=label, source="pretrained", task=task, classes=[])
        for model_id, (_, label, task) in PRETRAINED_CATALOG.items()
    ]
    models += _scan_weights_dir(TRAINED_WEIGHTS_DIR, "trained")
    models += _scan_weights_dir(IMPORTED_WEIGHTS_DIR, "imported")
    return models


def resolve(model_id: str) -> tuple[str, ModelTask]:
    """Returns (checkpoint path or name, task) for a model id."""
    if model_id in PRETRAINED_CATALOG:
        checkpoint, _, task = PRETRAINED_CATALOG[model_id]
        return checkpoint, task

    for source, directory in (("trained", TRAINED_WEIGHTS_DIR), ("imported", IMPORTED_WEIGHTS_DIR)):
        prefix = f"{source}:"
        if model_id.startswith(prefix):
            run_name = model_id[len(prefix):]
            weights_file = directory / f"{run_name}.pt"
            meta_file = directory / f"{run_name}.meta.json"
            if not weights_file.exists() or not meta_file.exists():
                raise ModelError(f"Model '{model_id}' not found")
            meta = json.loads(meta_file.read_text())
            return str(weights_file), meta.get("task", "detect")

    raise ModelError(f"Unknown model '{model_id}'")


def load_model(model_id: str):
    checkpoint, task = resolve(model_id)
    with _cache_lock:
        if model_id not in _model_cache:
            if task == "sam":
                from ultralytics import SAM

                _model_cache[model_id] = SAM(checkpoint)
            else:
                from ultralytics import YOLO

                _model_cache[model_id] = YOLO(checkpoint)
        return _model_cache[model_id]


def import_model(filename: str, content: bytes, label: str, task: ModelTask) -> ModelInfo:
    if task not in ("detect", "segment", "sam"):
        raise ModelError(f"Unsupported task '{task}'")

    stem = _sanitize_name(Path(filename).stem)
    name = stem
    suffix = 1
    while (IMPORTED_WEIGHTS_DIR / f"{name}.pt").exists():
        suffix += 1
        name = f"{stem}-{suffix}"

    weights_file = IMPORTED_WEIGHTS_DIR / f"{name}.pt"
    weights_file.write_bytes(content)

    classes: list[str] = []
    try:
        if task == "sam":
            from ultralytics import SAM

            SAM(str(weights_file))
        else:
            from ultralytics import YOLO

            loaded = YOLO(str(weights_file))
            names = getattr(loaded, "names", None) or {}
            classes = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    except Exception as e:
        weights_file.unlink(missing_ok=True)
        raise ModelError(f"Could not load '{filename}' as a {task} model: {e}")

    meta = {"label": label or name, "task": task, "classes": classes}
    (IMPORTED_WEIGHTS_DIR / f"{name}.meta.json").write_text(json.dumps(meta))

    return ModelInfo(id=f"imported:{name}", label=meta["label"], source="imported", task=task, classes=classes)
