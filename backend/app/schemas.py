from typing import Literal, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in absolute pixel coordinates, top-left origin."""

    class_name: str
    confidence: Optional[float] = None
    x: float = Field(..., description="Top-left x in pixels")
    y: float = Field(..., description="Top-left y in pixels")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")


class ImageAnnotations(BaseModel):
    image_id: str
    boxes: list[BoundingBox]


class SaveAnnotationsRequest(BaseModel):
    dataset_name: str
    image_id: str
    image_width: int
    image_height: int
    boxes: list[BoundingBox]


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
    source: Literal["pretrained", "trained"]
    classes: list[str] = []


class DetectRequest(BaseModel):
    model_id: str
    confidence: float = 0.25


class DetectionResult(BaseModel):
    image_width: int
    image_height: int
    boxes: list[BoundingBox]
