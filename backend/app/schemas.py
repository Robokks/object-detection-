from typing import Literal, Optional

from pydantic import BaseModel, Field

ShapeKind = Literal["box", "ellipse", "polygon"]
ModelTask = Literal["detect", "segment", "sam"]
ModelSource = Literal["pretrained", "trained", "imported"]


class Shape(BaseModel):
    """A labeled region, in absolute pixel coordinates (top-left origin).

    `x`/`y`/`width`/`height` are always the axis-aligned bounding box of the
    shape. `points` additionally carries the outline for non-box shapes
    (ellipse: a sampled ring; polygon: the freehand path) so the exact
    region can be re-drawn and exported as a YOLO-seg polygon label.
    """

    class_name: str
    confidence: Optional[float] = None
    shape: ShapeKind = "box"
    points: list[list[float]] = Field(default_factory=list)
    x: float = Field(..., description="Bounding box top-left x in pixels")
    y: float = Field(..., description="Bounding box top-left y in pixels")
    width: float = Field(..., description="Bounding box width in pixels")
    height: float = Field(..., description="Bounding box height in pixels")


class SaveAnnotationsRequest(BaseModel):
    dataset_name: str
    image_id: str
    image_width: int
    image_height: int
    shapes: list[Shape]


class DatasetInfo(BaseModel):
    name: str
    classes: list[str]
    image_count: int
    annotated_count: int


class CreateDatasetRequest(BaseModel):
    name: str
    classes: list[str]


class TrainRequest(BaseModel):
    dataset_name: str
    run_name: str
    mode: Literal["finetune", "scratch"] = "finetune"
    epochs: int = 50
    image_size: int = 640
    batch_size: int = 16
    base_model: str = "yolov8n"


class TrainJobStatus(BaseModel):
    run_name: str
    state: Literal["queued", "running", "completed", "failed"]
    mode: str
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    message: str = ""
    metrics: dict = {}
    weights_path: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    label: str
    source: ModelSource
    task: ModelTask
    classes: list[str] = []


class ImportModelResponse(BaseModel):
    model: ModelInfo


class DetectionResult(BaseModel):
    image_width: int
    image_height: int
    boxes: list[Shape]
