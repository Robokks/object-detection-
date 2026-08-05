import json
import random
import shutil
import uuid
from pathlib import Path

from PIL import Image

from app.config import DATASETS_DIR
from app.schemas import CreateDatasetRequest, DatasetInfo, Shape


class DatasetError(ValueError):
    pass


def _dataset_dir(name: str) -> Path:
    return DATASETS_DIR / name


def _meta_path(name: str) -> Path:
    return _dataset_dir(name) / "meta.json"


def _load_meta(name: str) -> dict:
    meta_path = _meta_path(name)
    if not meta_path.exists():
        raise DatasetError(f"Dataset '{name}' does not exist")
    return json.loads(meta_path.read_text())


def _save_meta(name: str, meta: dict) -> None:
    _meta_path(name).write_text(json.dumps(meta, indent=2))


def _polygon_points(shape: Shape) -> list[list[float]]:
    """Absolute pixel polygon outline for a shape, for YOLO-seg label export."""
    if shape.shape != "box" and len(shape.points) >= 3:
        return shape.points
    x, y, w, h = shape.x, shape.y, shape.width, shape.height
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _bbox_from_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def create_dataset(req: CreateDatasetRequest) -> DatasetInfo:
    if not req.name.strip():
        raise DatasetError("Dataset name is required")
    if not req.classes:
        raise DatasetError("At least one class is required")

    ds_dir = _dataset_dir(req.name)
    if ds_dir.exists():
        raise DatasetError(f"Dataset '{req.name}' already exists")

    (ds_dir / "images").mkdir(parents=True)
    (ds_dir / "labels").mkdir(parents=True)

    meta = {"classes": req.classes, "images": {}}
    _save_meta(req.name, meta)
    return DatasetInfo(name=req.name, classes=req.classes, image_count=0, annotated_count=0)


def list_datasets() -> list[DatasetInfo]:
    if not DATASETS_DIR.exists():
        return []
    results = []
    for ds_dir in sorted(DATASETS_DIR.iterdir()):
        meta_path = ds_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        images = meta.get("images", {})
        annotated = sum(1 for img in images.values() if img.get("shapes"))
        results.append(
            DatasetInfo(
                name=ds_dir.name,
                classes=meta.get("classes", []),
                image_count=len(images),
                annotated_count=annotated,
            )
        )
    return results


def get_dataset(name: str) -> dict:
    return _load_meta(name)


def add_image(name: str, filename: str, content: bytes) -> dict:
    meta = _load_meta(name)
    ds_dir = _dataset_dir(name)

    image_id = uuid.uuid4().hex
    ext = Path(filename).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        ext = ".jpg"
    dest = ds_dir / "images" / f"{image_id}{ext}"
    dest.write_bytes(content)

    with Image.open(dest) as img:
        width, height = img.size

    meta["images"][image_id] = {
        "filename": f"{image_id}{ext}",
        "width": width,
        "height": height,
        "shapes": [],
    }
    _save_meta(name, meta)
    return {"image_id": image_id, "width": width, "height": height, "filename": f"{image_id}{ext}"}


def image_path(name: str, image_id: str) -> Path:
    meta = _load_meta(name)
    image_entry = meta["images"].get(image_id)
    if not image_entry:
        raise DatasetError(f"Image '{image_id}' not found in dataset '{name}'")
    return _dataset_dir(name) / "images" / image_entry["filename"]


def save_annotations(
    name: str, image_id: str, image_width: int, image_height: int, shapes: list[Shape]
) -> None:
    meta = _load_meta(name)
    if image_id not in meta["images"]:
        raise DatasetError(f"Image '{image_id}' not found in dataset '{name}'")

    classes: list[str] = meta["classes"]
    for shape in shapes:
        if shape.class_name not in classes:
            classes.append(shape.class_name)
    meta["classes"] = classes

    stored_shapes = []
    label_lines = []
    for shape in shapes:
        polygon = _polygon_points(shape)
        x, y, w, h = _bbox_from_points(polygon) if shape.shape != "box" else (
            shape.x,
            shape.y,
            shape.width,
            shape.height,
        )
        stored = shape.model_copy(
            update={"x": x, "y": y, "width": w, "height": h, "center_x": x + w / 2, "center_y": y + h / 2}
        )
        stored_shapes.append(stored.model_dump())

        class_idx = classes.index(shape.class_name)
        normalized = []
        for px, py in polygon:
            nx = min(max(px / image_width, 0.0), 1.0)
            ny = min(max(py / image_height, 0.0), 1.0)
            normalized.extend([nx, ny])
        coords = " ".join(f"{v:.6f}" for v in normalized)
        label_lines.append(f"{class_idx} {coords}")

    meta["images"][image_id]["shapes"] = stored_shapes
    _save_meta(name, meta)

    ds_dir = _dataset_dir(name)
    label_path = ds_dir / "labels" / f"{image_id}.txt"
    label_path.write_text("\n".join(label_lines))


def delete_image(name: str, image_id: str) -> None:
    meta = _load_meta(name)
    entry = meta["images"].pop(image_id, None)
    if not entry:
        raise DatasetError(f"Image '{image_id}' not found in dataset '{name}'")
    _save_meta(name, meta)

    ds_dir = _dataset_dir(name)
    (ds_dir / "images" / entry["filename"]).unlink(missing_ok=True)
    (ds_dir / "labels" / f"{image_id}.txt").unlink(missing_ok=True)


def export_yolo_dataset(name: str, val_split: float = 0.2, seed: int = 42) -> Path:
    """Builds a YOLO-seg dataset (train/val split + data.yaml) under datasets/<name>/export."""
    meta = _load_meta(name)
    classes: list[str] = meta["classes"]
    annotated_ids = [
        image_id for image_id, entry in meta["images"].items() if entry.get("shapes")
    ]
    if not annotated_ids:
        raise DatasetError(f"Dataset '{name}' has no annotated images to train on")

    ds_dir = _dataset_dir(name)
    export_dir = ds_dir / "export"
    if export_dir.exists():
        shutil.rmtree(export_dir)

    rng = random.Random(seed)
    shuffled = annotated_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    val_ids = set(shuffled[:val_count])

    for split, ids in (("train", [i for i in shuffled if i not in val_ids]), ("val", list(val_ids) or shuffled)):
        (export_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            entry = meta["images"][image_id]
            src_img = ds_dir / "images" / entry["filename"]
            src_label = ds_dir / "labels" / f"{image_id}.txt"
            shutil.copy(src_img, export_dir / "images" / split / entry["filename"])
            if src_label.exists():
                shutil.copy(src_label, export_dir / "labels" / split / f"{image_id}.txt")

    data_yaml = export_dir / "data.yaml"
    class_list = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    data_yaml.write_text(
        f"path: {export_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{class_list}\n"
    )
    return data_yaml
