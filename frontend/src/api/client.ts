import type {
  DatasetDetail,
  DatasetInfo,
  DetectionResult,
  ModelInfo,
  ModelTask,
  Shape,
  TrainJobStatus,
  TrainMode,
  UploadedImage,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      // not JSON, keep raw body
    }
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function imageUrl(datasetName: string, imageId: string): string {
  return `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${encodeURIComponent(imageId)}/file`;
}

export const api = {
  listDatasets: () => request<DatasetInfo[]>("/api/datasets"),

  createDataset: (name: string, classes: string[]) =>
    request<DatasetInfo>("/api/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, classes }),
    }),

  getDataset: (name: string) =>
    request<DatasetDetail>(`/api/datasets/${encodeURIComponent(name)}`),

  uploadImage: async (name: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadedImage>(`/api/datasets/${encodeURIComponent(name)}/images`, {
      method: "POST",
      body: form,
    });
  },

  deleteImage: (name: string, imageId: string) =>
    request<{ ok: boolean }>(
      `/api/datasets/${encodeURIComponent(name)}/images/${encodeURIComponent(imageId)}`,
      { method: "DELETE" }
    ),

  saveAnnotations: (name: string, imageId: string, imageWidth: number, imageHeight: number, shapes: Shape[]) =>
    request<{ ok: boolean }>(`/api/datasets/${encodeURIComponent(name)}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_name: name,
        image_id: imageId,
        image_width: imageWidth,
        image_height: imageHeight,
        shapes,
      }),
    }),

  suggestShapes: (name: string, imageId: string, className: string) =>
    request<Shape[]>(
      `/api/datasets/${encodeURIComponent(name)}/images/${encodeURIComponent(imageId)}/suggest?class_name=${encodeURIComponent(className)}`,
      { method: "POST" }
    ),

  startTraining: (params: {
    dataset_name: string;
    run_name: string;
    mode: TrainMode;
    epochs: number;
    image_size: number;
    batch_size: number;
    base_model: string;
  }) =>
    request<TrainJobStatus>("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  listTrainJobs: () => request<TrainJobStatus[]>("/api/train/jobs"),

  getTrainJob: (runName: string) =>
    request<TrainJobStatus>(`/api/train/jobs/${encodeURIComponent(runName)}`),

  listModels: () => request<ModelInfo[]>("/api/models"),

  importModel: async (file: File, label: string, task: ModelTask) => {
    const form = new FormData();
    form.append("file", file);
    form.append("label", label);
    form.append("task", task);
    return request<ModelInfo>("/api/models/import", { method: "POST", body: form });
  },

  detect: async (file: File, modelId: string, confidence: number) => {
    const form = new FormData();
    form.append("file", file);
    form.append("model_id", modelId);
    form.append("confidence", String(confidence));
    return request<DetectionResult>("/api/detect", { method: "POST", body: form });
  },
};
