from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
WEIGHTS_DIR = DATA_DIR / "weights"
TRAINED_WEIGHTS_DIR = WEIGHTS_DIR / "trained"
IMPORTED_WEIGHTS_DIR = WEIGHTS_DIR / "imported"

for directory in (DATASETS_DIR, RUNS_DIR, WEIGHTS_DIR, TRAINED_WEIGHTS_DIR, IMPORTED_WEIGHTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
