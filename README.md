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

### Training faster on a free GPU (Colab)

Both the desktop app and this backend train on whatever CPU/GPU is on the
machine running them — on a CPU-only machine, that's slow. `colab/` has a
faster path that doesn't need any local GPU, and it accepts **either** of
two zip formats — pick whichever is less hassle:

- **A plain zip of your own photos with hand-drawn red outlines on them** —
  zip them up yourself, no export step needed. The notebook extracts the
  outlines on the Colab machine itself, using the exact same code as
  **"Import pre-annotated images"**.
- **A pre-built dataset export**, if you'd rather do that step locally:
  ```bash
  cd backend
  python scripts/export_for_colab.py <dataset_name>   # writes <dataset_name>-dataset.zip
  ```

Open the notebook — either upload `colab/train_on_colab.ipynb` to
[Google Colab](https://colab.research.google.com) yourself, or use the
**"Open in Google Colab"** PyCharm run configuration (`backend/scripts/open_colab.py`),
which opens it straight from GitHub in your browser, always the current
version, no manual upload needed. Then:

1. `Runtime` menu → `Change runtime type` → **T4 GPU** → Save.
2. `Runtime` → `Run all`.
3. First upload prompt: your dataset zip (either of the two formats above).
4. Second upload prompt is **optional** — upload a previous `best.pt` to
   continue fine-tuning from it (same idea as the desktop app's Train tab
   "Continue from checkpoint"), or click **Cancel** to start from stock
   COCO weights instead.

It fine-tunes `yolov8n-seg` — the same architecture the app trains locally
— then (5) optionally lets you upload a test image (or falls back to a few
images from the validation split if you skip that) and runs the
just-trained model on it right there, downloading `colab-detections.json`
with the same full detail as the desktop app's Detect tab and
`scripts/detect_to_json.py` — class, confidence, center position,
orientation angle, left/right/top/bottom edge midpoints, and the bounding
box for every detection. Useful for a quick sanity check on the new
weights before bothering to download and import them anywhere. Finally it
downloads **`training-run.zip`** — not just `best.pt`, but the whole
Ultralytics run folder: `weights/` (`best.pt` + `last.pt`), the confusion
matrix, precision/recall/F1 curves, `results.csv`/`results.png`,
`args.yaml`, and the training/validation batch preview images — the same
files you'd see in `backend/data/runs/<run_name>/` from a local run, just
zipped up since Colab can't write directly to your disk. Pull `best.pt`
out of `weights/` for **Detect → Import a model** (desktop or web, task
"Instance segmentation") — and it's now also a valid "previous best.pt" to
feed back into step 4 next time, so each round can build on the last
instead of starting over. The rest of the zip is for judging how the run
actually went (is precision/recall trending the right way, is the
confusion matrix clean, etc.) — the same diagnostics you'd check after a
local training run.

There's no supported way to run Colab's free GPU from a terminal/CI
pipeline — it's a browser-based product, and Google doesn't expose that
tier's compute through an API or CLI. `open_colab.py` just saves the
"find and re-upload the notebook" step each time; you still do the T4 GPU
selection, `Runtime → Run all`, and the upload by hand on the page it
opens. (Colab does support pointing it at a *local* Jupyter kernel instead
of its own — "Connect to a local runtime" — but that trades away the free
GPU for your own CPU, the opposite of what this is for.)

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

Every shape also carries `angle` — the long-axis orientation in degrees (0
= horizontal, 90 = vertical), fit from `points` with a minimum-area
rectangle. It needs an outline to fit, so it's always `0` for a plain
axis-aligned box; a rotated-box label or a segmentation/SAM mask both
produce a real angle. The desktop app's Detect tab shows this as an
**Angle** column next to Position.

The desktop Detect tab's results table also breaks the bounding box down
into **Left / Right / Top / Bottom (X, Y)** columns — the midpoints of each
box edge (e.g. Left is `(Box X, Position Y)`, Top is `(Position X, Box Y)`)
— alongside the raw **Box X / Box Y / Width / Height**. Like those, they're
always axis-aligned (not rotated with the object); for an angled pin's
actual tip-to-tip endpoints, use Position + Angle together instead.

### Standalone detection script (no app, just a JSON file)

`backend/scripts/detect_to_json.py` runs a model on an image with no server
and no GUI — for scripting, batch processing, or feeding another program.
Open it, edit the settings block near the top (`WEIGHTS_PATH`, `IMAGE_PATH`,
`TASK`, `CONFIDENCE`, `OUTPUT_PATH`), and run it — PyCharm's ▶ button next
to `if __name__ == "__main__":` works, same as `backend/run.py`. Those same
settings can be overridden with command-line flags instead
(`--weights`, `--image`, `--task`, `--confidence`, `--output` —
`--help` for details), so it also works as a normal CLI tool if you'd
rather not edit the file.

```bash
cd backend
python scripts/detect_to_json.py --weights best.pt --image photo.jpg --task segment
```

`IMAGE_PATH`/`--image` can also point at a folder — every image in it gets
detected and the JSON groups results per image. The output has exactly the
same fields as the desktop Detect tab's results table (both are built from
`app/services/report_service.py`, so they can't drift apart): class,
shape (`box`/`mask`), confidence, center position, orientation angle,
left/right/top/bottom edge midpoints, and the bounding box. Example for a
single image:

```json
{
  "model": {"weights": "best.pt", "task": "segment"},
  "confidence_threshold": 0.25,
  "image": "photo.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {
      "class_name": "pin",
      "shape": "mask",
      "confidence": 0.91,
      "position": {"x": 512.3, "y": 340.1},
      "angle_degrees": 37.2,
      "left": {"x": 480.0, "y": 340.1},
      "right": {"x": 544.6, "y": 340.1},
      "top": {"x": 512.3, "y": 310.0},
      "bottom": {"x": 512.3, "y": 370.2},
      "box": {"x": 480.0, "y": 310.0, "width": 64.6, "height": 60.2}
    }
  ]
}
```

This doesn't touch the app's model registry or `backend/data/` — point it
at any `.pt` checkpoint (e.g. `best.pt` from the Colab notebook) and any
image, independent of whatever's imported into the desktop/web apps.

There's also `backend/scripts/pin_detect_json.py` — the same idea, but a
single fully self-contained file with no dependency on the rest of this
repo (only `pip install ultralytics opencv-python numpy`), so it's the one
to copy onto another machine on its own. Set `MODEL` near the top to your
`.pt` checkpoint, then:

```bash
python pin_detect_json.py photo.jpg     # one image
python pin_detect_json.py photos/       # every image in a folder
```

It writes the same JSON shape as `detect_to_json.py` (`detections.json` by
default) and, as a bonus, also saves an annotated copy of each image
(`<name>_detections.png`) with boxes/masks, class + confidence labels, and
a center marker drawn on it, plus a console summary per image — handy for
eyeballing results without opening a JSON file. Task (`detect`/`segment`)
is read automatically from the checkpoint, so there's nothing to set beyond
`MODEL`, `CONF`, `IOU`, and `IMGSZ`.

Use `detect_to_json.py` when you're working inside this repo and want the
output guaranteed identical to the desktop app's table (it shares the exact
same code). Use `pin_detect_json.py` when you want one portable file with
zero repo dependency — e.g. deploying detection to a machine that doesn't
have this repo at all.

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

On the **Detect** tab, **Set ROI** lets you restrict detections to a region
of the image: click it, then drag a rectangle on the result canvas. Detection
still runs on the whole image (accuracy near the ROI's edge isn't affected by
cropping), but only detections whose center falls inside the rectangle are
kept — the status line next to the button shows how many. **Clear ROI**
removes the restriction. The ROI is per detection run: choosing a new image
resets it.

Clicking a detected shape on the canvas highlights its row in the results
table, and clicking a row selects that shape on the canvas — handy for
matching a specific number in the table (position, angle, ...) back to the
object it belongs to when several are close together.

### Fixing wrong detections and fine-tuning on them

When a detection is wrong (false positive, missed the real box, wrong
class), select it — click it on the canvas or its row in the table — then
click **Enable corrections** in the **"Fix wrong detections &
fine-tuning"** section. This makes the result canvas editable: **Delete
selected shape** removes a false positive; the Box/Rotated Box/Ellipse/Pen
tools let you draw a replacement in the right place with the right class
(pick it from the **Class** dropdown). Deleting everything and saving is
valid too — a "no objects here" example helps just as much as a corrected
box does.

Once the image looks right, pick a dataset from **Save to dataset** and
click **Save corrected image to dataset** — it's added as a new labeled
image (detection confidences are dropped; these are ground-truth labels
now), the same way an uploaded or hand-marked image would be. Repeat across
however many wrongly-detected images you want to fix, then go to the
**Train** tab and start a **fine-tune** run on that dataset — this is how
you turn the model's own mistakes into more training data, without a
separate labeling pass.

### Building a standalone .exe

To hand the desktop app to someone without them installing Python: on
Windows, double-click **`desktop\build_exe.bat`** — no typing required, it
installs [PyInstaller](https://pyinstaller.org/) if needed and builds
`desktop\dist\PinDetector\PinDetector.exe`. Zip that whole `PinDetector`
folder (the `.exe` needs the rest of it alongside it — it's a "folder"
build, not a single portable file, since that's faster to build/rebuild
than a single-file `.exe` for a bundle this size) and copy it to another
Windows machine; no Python, no PyCharm, no venv needed there.

It reuses whatever virtual environment the "Backend (FastAPI)" PyCharm run
configuration uses (`backend\.venv`) — set that up first if you haven't
(**Settings → Project → Python Interpreter → Add Interpreter → Virtualenv →
New**, pointed at `backend\.venv`) before running the `.bat` file. Rerun it
after code changes to rebuild; the first build is slow (bundling PySide6 +
PyTorch + Ultralytics together is a few hundred MB), later ones are
faster.

Prefer PyCharm to a `.bat` file, or want to run it from a terminal
yourself? `desktop\build_exe.py` (also has its own **"Build desktop .exe"**
PyCharm run configuration) and `desktop\pin_detector.spec` do the same
build — the `.bat` file just calls the same spec file.

This build hasn't been run end-to-end on Windows — PyInstaller output is
genuinely platform-specific, so if the first attempt hits a "module not
found" error at runtime, it's almost always fixed by adding that module's
name to `hiddenimports` in `pin_detector.spec` and rebuilding; say what the
error was and it can be fixed properly.

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
- **Open in Google Colab** — opens the training notebook straight from
  GitHub in your browser (see [Training faster on a free
  GPU](#training-faster-on-a-free-gpu-colab) above).

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
