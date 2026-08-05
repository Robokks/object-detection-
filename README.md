# Vision Object Detection Studio

A web app for training a custom object-detection/segmentation model and then
using it to detect objects (class + position) in new images.

- **Backend** (`backend/`): FastAPI + [Ultralytics](https://docs.ultralytics.com/)
  (YOLOv8 detect/segment + SAM). Handles dataset storage, labeling, training,
  model management, and inference.
- **Frontend** (`frontend/`): React + Vite (TypeScript). Three screens:
  1. **Dataset** — create a dataset, upload images, and label objects with a
     **box**, **rotated box**, **ellipse**, or **pen** (freehand) tool. The
     rotated box is the one to reach for on elongated objects that can appear
     at any angle (rods, pins, tools): drag to size, then drag its handle to
     match the object's orientation — much tighter than an axis-aligned box,
     and much faster than tracing freehand.
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
- Node.js 18+
- A GPU is optional but speeds up training significantly. CPU training works
  for small datasets/models.

## Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The first time you train or run detection with a pretrained model,
Ultralytics will download its checkpoint automatically — this requires
network access. SAM's checkpoint in particular is a large download.

All data (datasets, uploaded images, training runs, trained/imported
weights) is stored under `backend/data/`, which is gitignored.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

This starts the app at `http://localhost:5173`, configured (via
`.env.development`) to talk to the backend at `http://localhost:8000`. Run
both the backend and frontend at the same time.

## API overview

| Endpoint | Description |
|---|---|
| `POST /api/datasets` | Create a dataset with a set of class names |
| `GET /api/datasets` | List datasets with image/annotation counts |
| `POST /api/datasets/{name}/images` | Upload an image to a dataset |
| `POST /api/datasets/{name}/annotations` | Save labeled shapes for an image |
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
