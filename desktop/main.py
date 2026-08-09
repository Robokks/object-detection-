"""Desktop app entry point.

Reuses the backend's Python service layer (dataset/train/detect/autolabel
logic) directly, in-process — no FastAPI, no HTTP, no second process to
run. This only works because that layer was already plain Python with no
web-framework dependency baked in.
"""

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
    main()
