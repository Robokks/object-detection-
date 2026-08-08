export type ShapeKind = "box" | "ellipse" | "polygon" | "rotated_box";

export interface Shape {
  class_name: string;
  confidence?: number | null;
  shape: ShapeKind;
  points: number[][];
  x: number;
  y: number;
  width: number;
  height: number;
  /** Object position (bounding-box center) — stays meaningful regardless of rotation. Server-computed. */
  center_x?: number;
  center_y?: number;
}

export interface DatasetInfo {
  name: string;
  classes: string[];
  image_count: number;
  annotated_count: number;
}

export interface DatasetImageEntry {
  filename: string;
  width: number;
  height: number;
  shapes: Shape[];
}

export interface DatasetDetail {
  classes: string[];
  images: Record<string, DatasetImageEntry>;
}

export interface UploadedImage {
  image_id: string;
  width: number;
  height: number;
  filename: string;
}

export interface ImportAnnotatedImageResult {
  filename: string;
  image_id?: string | null;
  shapes_found: number;
  error?: string | null;
}

export type TrainMode = "finetune" | "scratch";

export interface TrainJobStatus {
  run_name: string;
  state: "queued" | "running" | "completed" | "failed";
  mode: string;
  current_epoch: number;
  total_epochs: number;
  progress: number;
  message: string;
  metrics: Record<string, number>;
  weights_path?: string | null;
}

export type ModelTask = "detect" | "segment" | "sam";
export type ModelSource = "pretrained" | "trained" | "imported";

export interface ModelInfo {
  id: string;
  label: string;
  source: ModelSource;
  task: ModelTask;
  classes: string[];
}

export interface DetectionResult {
  image_width: number;
  image_height: number;
  boxes: Shape[];
}
