import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DetectionResult, ModelInfo } from "../api/types";
import BoundingBoxOverlay from "../components/BoundingBoxOverlay";

export default function DetectPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [previewSrc, setPreviewSrc] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .listModels()
      .then((list) => {
        setModels(list);
        if (list.length > 0) setModelId(list[0].id);
      })
      .catch((e) => setError(String(e)));
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

  return (
    <div className="page">
      <h2>3. Detect Objects</h2>
      <p className="page-desc">
        Pick a model (pretrained or one you trained), upload an image, and run detection to see each object's
        class, confidence, and position.
      </p>

      <section className="card">
        <div className="form-grid">
          <label>
            Model
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
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
                  <th>Confidence</th>
                  <th>X</th>
                  <th>Y</th>
                  <th>Width</th>
                  <th>Height</th>
                </tr>
              </thead>
              <tbody>
                {result.boxes.map((b, i) => (
                  <tr key={i}>
                    <td>{b.class_name}</td>
                    <td>{b.confidence != null ? `${(b.confidence * 100).toFixed(1)}%` : "-"}</td>
                    <td>{b.x.toFixed(0)}</td>
                    <td>{b.y.toFixed(0)}</td>
                    <td>{b.width.toFixed(0)}</td>
                    <td>{b.height.toFixed(0)}</td>
                  </tr>
                ))}
                {result.boxes.length === 0 && (
                  <tr>
                    <td colSpan={6}>No objects detected above the confidence threshold.</td>
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
