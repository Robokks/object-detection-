# Vision Object Detection Studio

A web app for training a custom object-detection model and then using it to
detect objects (class + position) in new images.

- **Backend** (`backend/`): FastAPI + [Ultralytics YOLOv8](https://docs.ultralytics.com/).
  Handles dataset storage, labeling, training, and inference.
- **Frontend** (`frontend/`): React + Vite (TypeScript). Three screens:
  1. **Dataset** — create a dataset, upload images, draw bounding boxes to
     label objects.
  2. **Train** — fine-tune a pretrained YOLOv8 checkpoint on your labeled
     dataset, or train a fresh model from scratch. Training runs in the
     background with live progress.
  3. **Detect** — pick a model (pretrained COCO weights or one you trained),
     upload an image, and see detected objects drawn as bounding boxes plus
     a table with class, confidence, and pixel position for each detection.

## How it works

Datasets are stored on disk as images + per-image bounding boxes (drawn in
the browser). At training time, annotations are converted to YOLO-format
labels and split into train/val sets automatically. Training uses
Ultralytics' `YOLO` class:

- **Fine-tune mode** starts from pretrained COCO weights (`yolov8n.pt` /
  `yolov8s.pt`) — recommended for small datasets, converges quickly.
- **Scratch mode** starts from the model architecture only (random weights)
  — useful if you don't want any COCO-derived weights, but needs a larger
  dataset and more epochs to converge.

Trained weights are saved under `backend/data/weights/<run_name>.pt` and
immediately become selectable on the Detect page alongside the stock
pretrained models.

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

The first time you train or run detection, Ultralytics will download the
pretrained checkpoint (e.g. `yolov8n.pt`) automatically — this requires
network access.

All data (datasets, uploaded images, training runs, trained weights) is
stored under `backend/data/`, which is gitignored.

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
| `POST /api/datasets/{name}/annotations` | Save bounding boxes for an image |
| `POST /api/train` | Start a training run (`finetune` or `scratch`) |
| `GET /api/train/jobs` | List training runs and their live progress |
| `GET /api/models` | List available models (pretrained + trained) |
| `POST /api/detect` | Run detection on an uploaded image with a chosen model |

## Notes

- This is a local, single-user tool — there is no auth and the backend is
  meant to be run on your own machine or a trusted network.
- Training runs synchronously per-request in a background thread; only run
  one training job at a time.
