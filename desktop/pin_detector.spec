# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the desktop app.

Build with (from the desktop/ folder):
    pyinstaller pin_detector.spec --noconfirm
or just run build_exe.py, which does the same thing and works with
PyCharm's ▶ button — no command line needed.

This is a "onedir" build: output lands in dist/PinDetector/, a folder you
zip up and copy to another Windows machine. PinDetector.exe inside that
folder is what you actually run — it needs the rest of the folder next to
it, it's not a single portable file. onedir also rebuilds much faster than
--onefile while you're iterating, which matters here since PySide6 +
PyTorch + Ultralytics is a large bundle.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO_ROOT = Path(SPECPATH).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

datas = []
binaries = []
# our own backend package (app.services.*, app.schemas, ...) — not
# statically discoverable by PyInstaller since desktop/main.py only reaches
# it via a runtime sys.path.insert(), so it's spelled out via pathex below
# and every submodule is listed explicitly here.
hiddenimports = collect_submodules("app")

# ultralytics and opencv both ship non-Python data files (ultralytics' own
# default config YAMLs, cv2's native libs) and use import patterns
# PyInstaller's static analysis doesn't always catch on its own —
# collect_all() is the belt-and-suspenders way to get all three
# (data files + binaries + hidden imports) for a package like this.
for pkg in ("ultralytics", "cv2"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ["main.py"],
    pathex=[str(BACKEND_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PinDetector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="PinDetector",
)
