# Vision Object Detection Studio

An app for training a custom object-detection/segmentation model and then
using it to detect objects (class + position) in new images. Two interfaces
share the same underlying Python logic and the same data on disk:

- **Desktop app** (`desktop/`): a PySide6 (Qt) app. One process, no server,
  no browser — launch it and use it. See [Desktop app](#desktop-app) below.
- **Web app** (`backend/` + `frontend/`): FastAPI backend + React frontend,
  used from a browser. See [Web app](#web-app) below.

Both read and write the same `backend/data/` folder, so a dataset labeled in
one shows up in the other.

- **Backend** (`backend/`): FastAPI + [Ultralytics](https://docs.ultralytics.com/)
  (YOLOv8 detect/segment + SAM). Handles dataset storage, labeling, training,
  model management, and inference. The desktop app calls this same service
  layer (`backend/app/services/`, `backend/app/schemas.py`) directly,
  in-process — it's the shared core, not something only the web app uses.
- **Frontend** (`frontend/`): React + Vite (TypeScript). Three screens:
  1. **Dataset** — create a dataset, upload images, and label objects with a
     **box**, **rotated box**, **ellipse**, or **pen** (freehand) tool. The
     rotated box is the one to reach for on elongated objects that can appear
     at any angle (rods, pins, tools): drag to size, then drag its handle to
     match the object's orientation — much tighter than an axis-aligned box,
     and much faster than tracing freehand. A **"Suggest cylinders"** button
     runs a classical image-processing pass that proposes rotated boxes for
     you to review (see below).
  2. **Train** — fine-tune a pretrained YOLOv8 checkpoint on your labeled
     dataset, or train a fresh model from scratch. Training runs in the
     background with live progress.
  3. **Detect** — pick a model, upload an image, and see detected objects
     drawn as boxes/masks plus a table with class, confidence, and pixel
     position for each detection. You can also **import your own pretrained
     `.pt` checkpoint** (another YOLO variant, SAM, etc.) here.

## How it works

### Labeling

Datasets are stored on disk as images + per-image shapes drawn in the
browser (box, rotated box, ellipse, or freehand polygon). Every shape's
outline — including plain and rotated boxes, both just 4-point outlines —
is exported as a YOLO-seg polygon label at training time, split into
train/val sets automatically. This is what makes the rotated box tool
worthwhile: an axis-aligned box around a diagonal object is mostly
background, which is a weak training signal; the rotated outline is tight
regardless of angle.

### Auto-suggested labels

Photographed under a flash, a cylindrical object (rod, pin, roller) tends to
show a bright specular highlight running down its length, flanked by
darker surface on both sides — a dark → glare → dark cross-section. The
**"Suggest cylinders"** button (Dataset page) scans the active image for
that pattern with classical image processing (OpenCV: brightness
thresholding, connected components, oriented bounding rects — no ML
involved, no training data needed) and adds each candidate as an editable
rotated-box shape labeled with the currently selected class.

This is a heuristic, not a classifier — review its output before training
on it. It works best when objects are separated; where they touch or cross,
it tends to merge them into one coarse box. Delete anything wrong the same
way you'd delete a hand-drawn shape; accepted candidates are saved exactly
like any other annotation.

### Importing already hand-marked images

If you've already marked objects by drawing solid outlines directly on your
images (common when labeling without a dedicated tool), use **"Import
pre-annotated images"** on the Dataset page instead of the plain image
upload. It looks for outlines drawn in solid red (`RGB 237,28,36` and
similar), extracts one shape per outline, and removes the red pixels from
the stored image (via inpainting) so the red lines themselves don't become
a training artifact the model could latch onto instead of the real object.

Touching or overlapping outlines are still separated correctly: rather than
treating each red blob as one shape (which merges outlines that touch),
this looks at each outline's *interior* — two outlines that touch at an
edge still have separate interiors, so they come out as separate shapes.
Hand-drawn/wobbly outlines are fit with an oriented rectangle same as the
rotated-box tool. If your outlines are a different, non-red color, this
won't find them yet — say so and it can be adjusted.

### Training

Training always targets the instance-segmentation variant of the chosen
architecture (e.g. `yolov8n-seg`), since labels are polygon outlines:

- **Fine-tune mode** starts from pretrained COCO weights — recommended for
  small datasets, converges quickly.
- **Scratch mode** starts from the model architecture only (random weights)
  — useful if you don't want any COCO-derived weights, but needs a larger
  dataset and more epochs to converge.

Trained weights are saved under `backend/data/weights/trained/<run_name>.pt`
and immediately become selectable on the Detect page.

### Models

The Detect page's model list combines three sources:

- **Pretrained** — a small built-in catalog: YOLOv8n/s/m (detection),
  YOLOv8n/s-seg (segmentation), and SAM base (`sam_b.pt`, class-agnostic —
  it segments every object it finds but doesn't name them). Ultralytics
  downloads these automatically on first use (needs network access).
- **Trained by you** — runs completed on the Train page.
- **Imported** — any `.pt` checkpoint you upload on the Detect page. It's
  validated by loading it with Ultralytics; class names are read from the
  checkpoint automatically when present. Choose the task that matches the
  checkpoint (detection, segmentation, or SAM-style).

Detection results carry a polygon outline (`points`) whenever the model
produces a mask (segmentation/SAM); plain detection models return just a
bounding box. Every result also carries `center_x`/`center_y` — the
bounding box's center, computed server-side and always populated. That's
the field to use as an object's "position": unlike the box's top-left
corner, the center stays meaningful regardless of the object's rotation
(rotating a box around its own center doesn't move the center).

## Requirements

- Python 3.10+
- Node.js 18+ (web app only)
- A GPU is optional but speeds up training significantly. CPU training works
  for small datasets/models.

## Desktop app

A single PySide6 (Qt) window with the same three tabs as the web app —
Dataset, Train, Detect — calling `backend/app/services/` directly in the
same process. No server, no browser, nothing else to run.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r ../desktop/requirements.txt
python ../desktop/main.py
```

(`desktop/requirements.txt` pulls in `backend/requirements.txt` too, so if
you're setting up fresh, installing just that one file is enough:
`pip install -r desktop/requirements.txt` from the repo root, using any venv
you like — it doesn't have to be `backend/.venv` specifically, that's just
what's already used elsewhere in this README.)

**PySide6, not PyQt6** — LGPL-licensed, free to use and distribute
(including closed-source) with no royalties or subscription.

Labeling works the same as the web app: pick a tool (Box / Rotated Box /
Ellipse / Pen) and drag on the image. For **Rotated Box**, drag to size, then
drag the small purple handle that appears to rotate it around its center.
Click a shape to select it, then **Delete selected shape** (or press
Delete/Backspace) to remove it. **Suggest cylinders** and **Import
pre-annotated images** work identically to their web-app counterparts,
just without a network round-trip.

Since it's Python calling Python directly, dataset edits, training runs, and
detections all touch `backend/data/` immediately — nothing to sync.

## Web app

The repo ships with shared PyCharm Run Configurations (`.idea/runConfigurations/`)
that appear automatically in the run-configuration dropdown (top toolbar)
once you open the project folder in PyCharm — no command line needed:

- **Backend (FastAPI)** — runs `backend/run.py`. First run, PyCharm needs an
  interpreter for it: create the venv once (`Settings/Preferences → Project →
  Python Interpreter → Add Interpreter → Virtualenv → New`, pointed at
  `backend/.venv`) and install `backend/requirements.txt` through the same
  Settings page's package-install UI — no terminal either way. If PyCharm
  can't find the interpreter the config expects, it'll prompt you to pick
  one; select that same `backend/.venv`.
- **Frontend (Vite dev server)** — runs `npm run dev` in `frontend/`.
  Requires PyCharm's bundled JavaScript/Node.js support (present in
  PyCharm Professional; on Community, add it via `Settings → Plugins →
  Marketplace → "Node.js"`, also just clicks, no terminal). Point it at a
  Node interpreter the same way if prompted.
- **Run Everything (Backend + Frontend)** — a compound configuration that
  starts both with a single click.

Pick a configuration from the dropdown and click the green ▶. Fallback if a
configuration doesn't show up cleanly (PyCharm versions vary): open
`backend/run.py` directly and click the ▶ that appears next to
`if __name__ == "__main__":` — that always works with zero configuration,
same idea as running any Python script from an IDE. The frontend has an
equivalent: open `frontend/package.json`, and PyCharm shows a ▶ run icon
next to the `"dev"` entry under `"scripts"`.

## API overview

| Endpoint | Description |
|---|---|
| `POST /api/datasets` | Create a dataset with a set of class names |
| `GET /api/datasets` | List datasets with image/annotation counts |
| `POST /api/datasets/{name}/images` | Upload an image to a dataset |
| `POST /api/datasets/{name}/annotations` | Save labeled shapes for an image |
| `POST /api/datasets/{name}/images/{image_id}/suggest` | Auto-suggest rotated-box candidates for an image |
| `POST /api/datasets/{name}/import-annotated` | Import images with hand-drawn red outlines; extracts shapes and cleans the images |
| `POST /api/train` | Start a training run (`finetune` or `scratch`) |
| `GET /api/train/jobs` | List training runs and their live progress |
| `GET /api/models` | List available models (pretrained + trained + imported) |
| `POST /api/models/import` | Upload and register a custom `.pt` checkpoint |
| `POST /api/detect` | Run detection on an uploaded image with a chosen model |

## Notes

- This is a local, single-user tool — there is no auth and the backend is
  meant to be run on your own machine or a trusted network. Anyone who can
  reach the backend can upload arbitrary `.pt` files, which Python can
  deserialize with side effects — don't expose it to an untrusted network.
- Training runs in a background thread; only run one training job at a time.
- SAM's "everything" mode (no prompts) can return many masks for a busy
  image; the API caps this at 50, keeping the largest ones.
