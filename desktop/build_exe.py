"""Build a standalone .exe from the desktop app — no Python install needed
to run it afterward, just this one folder.

Run this file directly (PyCharm's ▶ button works, or `python build_exe.py`
from a terminal in `desktop/`). Needs PyInstaller installed first, once:
    pip install pyinstaller

Output lands in desktop/dist/PinDetector/ — zip that whole folder to move
it to another Windows machine; PinDetector.exe inside it is what you run
(it needs the rest of the folder alongside it). First build takes a
while — PySide6 + PyTorch + Ultralytics bundled together is a few hundred
MB — rerun this script to rebuild after code changes.
"""

from pathlib import Path

import PyInstaller.__main__

SPEC_FILE = Path(__file__).resolve().parent / "pin_detector.spec"

if __name__ == "__main__":
    PyInstaller.__main__.run([str(SPEC_FILE), "--noconfirm"])
