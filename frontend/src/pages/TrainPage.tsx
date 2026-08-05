import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { DatasetInfo, TrainJobStatus, TrainMode } from "../api/types";

export default function TrainPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetName, setDatasetName] = useState("");
  const [runName, setRunName] = useState("");
  const [mode, setMode] = useState<TrainMode>("finetune");
  const [baseModel, setBaseModel] = useState("yolov8n");
  const [epochs, setEpochs] = useState(50);
  const [imageSize, setImageSize] = useState(640);
  const [batchSize, setBatchSize] = useState(16);
  const [jobs, setJobs] = useState<TrainJobStatus[]>([]);
  const [error, setError] = useState("");
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch((e) => setError(String(e)));
    refreshJobs();
    pollRef.current = window.setInterval(refreshJobs, 3000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  async function refreshJobs() {
    try {
      const list = await api.listTrainJobs();
      setJobs(list);
    } catch {
      // backend may not be reachable yet; ignore transient poll errors
    }
  }

  async function handleStart() {
    setError("");
    if (!datasetName || !runName.trim()) {
      setError("Choose a dataset and give the run a name");
      return;
    }
    try {
      await api.startTraining({
        dataset_name: datasetName,
        run_name: runName.trim(),
        mode,
        epochs,
        image_size: imageSize,
        batch_size: batchSize,
        base_model: baseModel,
      });
      await refreshJobs();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="page">
      <h2>2. Train a Model</h2>
      <p className="page-desc">
        Fine-tune a pretrained YOLO checkpoint on your labeled dataset (fast, works with small datasets), or
        train a fresh model from scratch using the same architecture with random weights. Since labels can include
        ellipses and freehand outlines, training always targets the instance-segmentation variant of the chosen
        architecture (e.g. yolov8n-seg) so that precision isn't lost — plain boxes still work fine, they're just
        treated as 4-point outlines.
      </p>

      <section className="card">
        <h3>Configure run</h3>
        <div className="form-grid">
          <label>
            Dataset
            <select value={datasetName} onChange={(e) => setDatasetName(e.target.value)}>
              <option value="">-- choose dataset --</option>
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.annotated_count} annotated)
                </option>
              ))}
            </select>
          </label>

          <label>
            Run name
            <input value={runName} onChange={(e) => setRunName(e.target.value)} placeholder="e.g. my-model-v1" />
          </label>

          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value as TrainMode)}>
              <option value="finetune">Fine-tune pretrained (recommended)</option>
              <option value="scratch">Train from scratch</option>
            </select>
          </label>

          <label>
            Base architecture
            <select value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
              <option value="yolov8n">yolov8n (fastest)</option>
              <option value="yolov8s">yolov8s (more accurate, slower)</option>
            </select>
          </label>

          <label>
            Epochs
            <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
          </label>

          <label>
            Image size
            <input
              type="number"
              min={64}
              step={32}
              value={imageSize}
              onChange={(e) => setImageSize(Number(e.target.value))}
            />
          </label>

          <label>
            Batch size
            <input type="number" min={1} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
          </label>
        </div>
        <button onClick={handleStart}>Start training</button>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <h3>Runs</h3>
        {jobs.length === 0 && <p className="page-desc">No training runs yet.</p>}
        {jobs.map((job) => (
          <div key={job.run_name} className="job-row">
            <div className="job-header">
              <strong>{job.run_name}</strong>
              <span className={"badge badge-" + job.state}>{job.state}</span>
              <span className="job-mode">{job.mode}</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${Math.round(job.progress * 100)}%` }} />
            </div>
            <div className="job-meta">
              <span>
                {job.current_epoch}/{job.total_epochs} epochs
              </span>
              <span>{job.message}</span>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
