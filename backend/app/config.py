from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
UPLOADS_DIR = DATA_DIR / "uploads"
WEIGHTS_DIR = DATA_DIR / "weights"

for directory in (DATASETS_DIR, RUNS_DIR, UPLOADS_DIR, WEIGHTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Pretrained checkpoint used as the starting point for fine-tuning / inference.
DEFAULT_PRETRAINED_WEIGHTS = "yolov8n.pt"
# Model config used when training a fresh (non-pretrained) model from scratch.
DEFAULT_SCRATCH_CONFIG = "yolov8n.yaml"
