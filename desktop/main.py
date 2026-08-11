"""Desktop app entry point.

Reuses the backend's Python service layer (dataset/train/detect/autolabel
logic) directly, in-process — no FastAPI, no HTTP, no second process to
run. This only works because that layer was already plain Python with no
web-framework dependency baked in.
"""

import multiprocessing
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Vision Object Detection Studio")
    window = MainWindow()
    window.resize(1280, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Training spins up PyTorch DataLoader worker subprocesses. On Windows,
    # a frozen .exe has no fork() — each worker re-executes this same exe
    # from scratch — so without freeze_support() called first, training can
    # crash or hang (workers re-run the whole app instead of just loading
    # data). No effect running from source; only matters once frozen.
    multiprocessing.freeze_support()
    main()
