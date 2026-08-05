import { useEffect, useState } from "react";
import { api, imageUrl } from "../api/client";
import type { DatasetDetail, DatasetInfo, Shape } from "../api/types";
import BoundingBoxEditor from "../components/BoundingBoxEditor";

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [activeImageId, setActiveImageId] = useState<string>("");
  const [activeClass, setActiveClass] = useState<string>("");
  const [newDatasetName, setNewDatasetName] = useState("");
  const [newClasses, setNewClasses] = useState("");
  const [newClassName, setNewClassName] = useState("");
  const [error, setError] = useState<string>("");
  const [saving, setSaving] = useState(false);

  async function refreshDatasets(selectName?: string) {
    const list = await api.listDatasets();
    setDatasets(list);
    if (selectName) setSelected(selectName);
  }

  useEffect(() => {
    refreshDatasets().catch((e) => setError(String(e)));
  }, []);

  async function loadDetail(name: string) {
    if (!name) {
      setDetail(null);
      return;
    }
    const d = await api.getDataset(name);
    setDetail(d);
    const ids = Object.keys(d.images);
    setActiveImageId(ids[0] ?? "");
    setActiveClass(d.classes[0] ?? "");
  }

  useEffect(() => {
    loadDetail(selected).catch((e) => setError(String(e)));
  }, [selected]);

  async function handleCreateDataset() {
    setError("");
    const classes = newClasses
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);
    try {
      await api.createDataset(newDatasetName.trim(), classes);
      setNewDatasetName("");
      setNewClasses("");
      await refreshDatasets(newDatasetName.trim());
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!files || !selected) return;
    setError("");
    try {
      let lastId = "";
      for (const file of Array.from(files)) {
        const uploaded = await api.uploadImage(selected, file);
        lastId = uploaded.image_id;
      }
      await loadDetail(selected);
      if (lastId) setActiveImageId(lastId);
      await refreshDatasets();
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleAddClass() {
    if (!detail || !newClassName.trim() || !selected) return;
    const classes = [...detail.classes, newClassName.trim()];
    setDetail({ ...detail, classes });
    setActiveClass(newClassName.trim());
    setNewClassName("");
  }

  async function handleShapesChange(shapes: Shape[]) {
    if (!detail || !activeImageId || !selected) return;
    const image = detail.images[activeImageId];
    const updatedDetail: DatasetDetail = {
      ...detail,
      images: { ...detail.images, [activeImageId]: { ...image, shapes } },
    };
    setDetail(updatedDetail);
    setSaving(true);
    try {
      await api.saveAnnotations(selected, activeImageId, image.width, image.height, shapes);
      await refreshDatasets();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteImage(imageId: string) {
    if (!selected) return;
    await api.deleteImage(selected, imageId);
    await loadDetail(selected);
    await refreshDatasets();
  }

  const imageIds = detail ? Object.keys(detail.images) : [];
  const activeImage = detail && activeImageId ? detail.images[activeImageId] : null;

  return (
    <div className="page">
      <h2>1. Dataset &amp; Labeling</h2>
      <p className="page-desc">
        Create a dataset, upload images, then label objects with the box, ellipse, or pen (freehand) tool.
      </p>

      <section className="card">
        <h3>Create dataset</h3>
        <div className="form-row">
          <input
            placeholder="dataset name"
            value={newDatasetName}
            onChange={(e) => setNewDatasetName(e.target.value)}
          />
          <input
            placeholder="classes, comma separated (e.g. cat, dog)"
            value={newClasses}
            onChange={(e) => setNewClasses(e.target.value)}
          />
          <button onClick={handleCreateDataset} disabled={!newDatasetName.trim()}>
            Create
          </button>
        </div>
      </section>

      <section className="card">
        <h3>Select dataset</h3>
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="">-- choose dataset --</option>
          {datasets.map((d) => (
            <option key={d.name} value={d.name}>
              {d.name} ({d.annotated_count}/{d.image_count} annotated)
            </option>
          ))}
        </select>
      </section>

      {selected && detail && (
        <>
          <section className="card">
            <h3>Upload images</h3>
            <input type="file" accept="image/*" multiple onChange={(e) => handleUpload(e.target.files)} />
          </section>

          <section className="card">
            <h3>Classes</h3>
            <div className="chip-row">
              {detail.classes.map((c) => (
                <button
                  key={c}
                  className={"chip" + (c === activeClass ? " chip-active" : "")}
                  onClick={() => setActiveClass(c)}
                >
                  {c}
                </button>
              ))}
            </div>
            <div className="form-row">
              <input
                placeholder="new class name"
                value={newClassName}
                onChange={(e) => setNewClassName(e.target.value)}
              />
              <button onClick={handleAddClass} disabled={!newClassName.trim()}>
                Add class
              </button>
            </div>
          </section>

          {imageIds.length > 0 && (
            <section className="card">
              <h3>Images ({imageIds.length})</h3>
              <div className="thumb-row">
                {imageIds.map((id) => (
                  <div key={id} className={"thumb" + (id === activeImageId ? " thumb-active" : "")}>
                    <img src={imageUrl(selected, id)} onClick={() => setActiveImageId(id)} alt="" />
                    <div className="thumb-meta">
                      <span>{detail.images[id].shapes.length} shape(s)</span>
                      <button onClick={() => handleDeleteImage(id)}>delete</button>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {activeImage && (
            <section className="card">
              <h3>Label image {saving && <span className="saving-badge">saving…</span>}</h3>
              <BoundingBoxEditor
                imageSrc={imageUrl(selected, activeImageId)}
                naturalWidth={activeImage.width}
                naturalHeight={activeImage.height}
                classes={detail.classes}
                activeClass={activeClass}
                shapes={activeImage.shapes}
                onShapesChange={handleShapesChange}
              />
            </section>
          )}
        </>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}
