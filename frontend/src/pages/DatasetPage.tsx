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
  const [suggesting, setSuggesting] = useState(false);
  const [suggestMessage, setSuggestMessage] = useState("");
  const [importingAnnotated, setImportingAnnotated] = useState(false);
  const [importMessage, setImportMessage] = useState("");

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

  async function handleImportAnnotated(files: FileList | null) {
    if (!files || files.length === 0 || !selected || !activeClass) return;
    setImportingAnnotated(true);
    setImportMessage("");
    setError("");
    try {
      const results = await api.importAnnotatedImages(selected, Array.from(files), activeClass);
      const totalShapes = results.reduce((sum, r) => sum + r.shapes_found, 0);
      const failed = results.filter((r) => r.error);
      await loadDetail(selected);
      await refreshDatasets();
      setImportMessage(
        `Imported ${results.length - failed.length}/${results.length} image(s), extracted ${totalShapes} shape(s) labeled "${activeClass}".` +
          (failed.length > 0 ? ` ${failed.length} failed: ${failed.map((f) => f.filename).join(", ")}` : "")
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setImportingAnnotated(false);
    }
  }

  async function handleSuggestShapes() {
    if (!detail || !activeImageId || !selected || !activeClass) return;
    setSuggesting(true);
    setSuggestMessage("");
    setError("");
    try {
      const suggested = await api.suggestShapes(selected, activeImageId, activeClass);
      const image = detail.images[activeImageId];
      await handleShapesChange([...image.shapes, ...suggested]);
      setSuggestMessage(
        suggested.length > 0
          ? `Found ${suggested.length} candidate(s), labeled "${activeClass}" — review below and delete any that aren't real.`
          : "No candidates found on this image."
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setSuggesting(false);
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

          <section className="card">
            <h3>Import pre-annotated images</h3>
            <p className="page-desc">
              Already marked objects by hand-drawing solid red outlines directly on your images? Upload them here
              instead of plain photos. Each red outline is extracted as a shape labeled{" "}
              <strong>{activeClass || "(select a class above)"}</strong>, and the red lines are removed from the
              stored image so they don't end up as a training artifact.
            </p>
            <input
              type="file"
              accept="image/*"
              multiple
              disabled={!activeClass || importingAnnotated}
              onChange={(e) => handleImportAnnotated(e.target.files)}
            />
            {importingAnnotated && <p className="suggest-message">Extracting outlines…</p>}
            {importMessage && <p className="suggest-message">{importMessage}</p>}
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
              <div className="suggest-row">
                <button onClick={handleSuggestShapes} disabled={!activeClass || suggesting}>
                  {suggesting ? "Scanning…" : "Suggest cylinders"}
                </button>
                <p className="page-desc suggest-hint">
                  Looks for the dark → glare → dark pattern a flash makes on a cylindrical surface, and proposes
                  rotated boxes labeled <strong>{activeClass || "(select a class)"}</strong>. Works best when
                  objects don't overlap; review and delete any bad suggestions below.
                </p>
              </div>
              {suggestMessage && <p className="suggest-message">{suggestMessage}</p>}
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
