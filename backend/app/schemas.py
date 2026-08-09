from typing import Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

ShapeKind = Literal["box", "ellipse", "polygon", "rotated_box"]
ModelTask = Literal["detect", "segment", "sam"]
ModelSource = Literal["pretrained", "trained", "imported"]


def _orientation_angle(points: list[list[float]]) -> float:
    """Long-axis orientation in degrees, 0 = horizontal, increasing clockwise
    (image y-down), normalized to [0, 180) since a line has no direction.
    Needs an outline (ellipse ring / polygon / rotated_box points, or a
    detected mask's contour) — a plain axis-aligned box carries no
    orientation, so it's always 0.
    """
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
    theta = angle if rw >= rh else angle + 90
    return round(float(theta % 180), 1)


class Shape(BaseModel):
    """A labeled region, in absolute pixel coordinates (top-left origin).

    `x`/`y`/`width`/`height` are always the axis-aligned bounding box of the
    shape. `points` additionally carries the outline for non-box shapes
    (ellipse: a sampled ring; polygon/rotated_box: the outline corners) so
    the exact region can be re-drawn and exported as a YOLO-seg polygon
    label. `center_x`/`center_y` is the object's position — the bounding
    box's center, which (unlike its top-left corner) stays meaningful
    regardless of the object's rotation. `angle` is the object's long-axis
    orientation in degrees (0 = horizontal, 90 = vertical), fit from
    `points` via a minimum-area rectangle — this is what answers "which way
    is the pin pointing". Server-computed; any value sent by a client is
    ignored and overwritten.
    """

    class_name: str
    confidence: Optional[float] = None
    shape: ShapeKind = "box"
    points: list[list[float]] = Field(default_factory=list)
    x: float = Field(..., description="Bounding box top-left x in pixels")
    y: float = Field(..., description="Bounding box top-left y in pixels")
    width: float = Field(..., description="Bounding box width in pixels")
    height: float = Field(..., description="Bounding box height in pixels")
    center_x: float = 0.0
    center_y: float = 0.0
    angle: float = 0.0

    @model_validator(mode="after")
    def _compute_derived(self) -> "Shape":
        self.center_x = self.x + self.width / 2
        self.center_y = self.y + self.height / 2
        self.angle = _orientation_angle(self.points)
        return self


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


class ImportAnnotatedImageResult(BaseModel):
    filename: str
    image_id: Optional[str] = None
    shapes_found: int = 0
    error: Optional[str] = None


class DetectionResult(BaseModel):
    image_width: int
    image_height: int
    boxes: list[Shape]
