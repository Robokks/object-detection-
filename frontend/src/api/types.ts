export interface BoundingBox {
  class_name: string;
  confidence?: number | null;
  x: number;
  y: number;
  width: number;
  height: number;
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
  boxes: BoundingBox[];
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

export interface ModelInfo {
  id: string;
  label: string;
  source: "pretrained" | "trained";
  classes: string[];
}

export interface DetectionResult {
  image_width: number;
  image_height: number;
  boxes: BoundingBox[];
}
