import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DetectionResult, ModelInfo, ModelTask } from "../api/types";
import BoundingBoxOverlay from "../components/BoundingBoxOverlay";

const SOURCE_LABELS: Record<ModelInfo["source"], string> = {
  pretrained: "Pretrained",
  trained: "Trained by you",
  imported: "Imported",
};

export default function DetectPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [previewSrc, setPreviewSrc] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [importFile, setImportFile] = useState<File | null>(null);
  const [importLabel, setImportLabel] = useState("");
  const [importTask, setImportTask] = useState<ModelTask>("detect");
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState("");

  function refreshModels(selectId?: string) {
    return api.listModels().then((list) => {
      setModels(list);
      if (selectId) setModelId(selectId);
      else if (!modelId && list.length > 0) setModelId(list[0].id);
    });
  }

  useEffect(() => {
    refreshModels().catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleFileChange(f: File | null) {
    setFile(f);
    setResult(null);
    setError("");
    if (f) setPreviewSrc(URL.createObjectURL(f));
    else setPreviewSrc("");
  }

  async function handleDetect() {
    if (!file || !modelId) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.detect(file, modelId, confidence);
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleImport() {
    if (!importFile) return;
    setImporting(true);
    setImportError("");
    try {
      const imported = await api.importModel(importFile, importLabel.trim(), importTask);
      setImportFile(null);
      setImportLabel("");
      await refreshModels(imported.id);
    } catch (e) {
      setImportError(String(e));
    } finally {
      setImporting(false);
    }
  }

  const grouped: Record<string, ModelInfo[]> = {};
  for (const m of models) {
    (grouped[m.source] ??= []).push(m);
  }

  return (
    <div className="page">
      <h2>3. Detect Objects</h2>
      <p className="page-desc">
        Pick a model, upload an image, and run detection to see each object's class, confidence, and position.
      </p>

      <section className="card">
        <h3>Import a model</h3>
        <p className="page-desc">
          Bring your own pretrained checkpoint (a YOLO variant, SAM, or anything Ultralytics can load) by uploading
          its <code>.pt</code> weights file. Class names are read automatically from the checkpoint when available.
        </p>
        <div className="form-grid">
          <label>
            Weights file (.pt)
            <input type="file" accept=".pt" onChange={(e) => setImportFile(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            Display name
            <input
              placeholder="e.g. my-yolov9-checkpoint"
              value={importLabel}
              onChange={(e) => setImportLabel(e.target.value)}
            />
          </label>
          <label>
            Task
            <select value={importTask} onChange={(e) => setImportTask(e.target.value as ModelTask)}>
              <option value="detect">Object detection (boxes)</option>
              <option value="segment">Instance segmentation (masks)</option>
              <option value="sam">SAM-style (class-agnostic, segments everything)</option>
            </select>
          </label>
        </div>
        <button onClick={handleImport} disabled={!importFile || importing}>
          {importing ? "Importing…" : "Import model"}
        </button>
        {importError && <p className="error">{importError}</p>}
      </section>

      <section className="card">
        <div className="form-grid">
          <label>
            Model
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {Object.entries(grouped).map(([source, list]) => (
                <optgroup key={source} label={SOURCE_LABELS[source as ModelInfo["source"]] ?? source}>
                  {list.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label>
            Confidence threshold: {confidence.toFixed(2)}
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
            />
          </label>
        </div>
        <input type="file" accept="image/*" onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)} />
        <button onClick={handleDetect} disabled={!file || !modelId || loading}>
          {loading ? "Detecting…" : "Run detection"}
        </button>
        {error && <p className="error">{error}</p>}
      </section>

      {previewSrc && (
        <section className="card">
          <h3>Result</h3>
          {result ? (
            <BoundingBoxOverlay
              imageSrc={previewSrc}
              naturalWidth={result.image_width}
              naturalHeight={result.image_height}
              boxes={result.boxes}
            />
          ) : (
            <img src={previewSrc} alt="preview" style={{ maxWidth: 760 }} />
          )}

          {result && (
            <table className="results-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Shape</th>
                  <th>Confidence</th>
                  <th>Position (X, Y)</th>
                  <th>Box X</th>
                  <th>Box Y</th>
                  <th>Width</th>
                  <th>Height</th>
                </tr>
              </thead>
              <tbody>
                {result.boxes.map((b, i) => {
                  const cx = b.center_x ?? b.x + b.width / 2;
                  const cy = b.center_y ?? b.y + b.height / 2;
                  return (
                    <tr key={i}>
                      <td>{b.class_name}</td>
                      <td>{b.points.length >= 3 ? "mask" : "box"}</td>
                      <td>{b.confidence != null ? `${(b.confidence * 100).toFixed(1)}%` : "-"}</td>
                      <td className="position-cell">
                        ({cx.toFixed(0)}, {cy.toFixed(0)})
                      </td>
                      <td>{b.x.toFixed(0)}</td>
                      <td>{b.y.toFixed(0)}</td>
                      <td>{b.width.toFixed(0)}</td>
                      <td>{b.height.toFixed(0)}</td>
                    </tr>
                  );
                })}
                {result.boxes.length === 0 && (
                  <tr>
                    <td colSpan={8}>No objects detected above the confidence threshold.</td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </section>
      )}
    </div>
  );
}
