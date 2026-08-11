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

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO_ROOT = Path(SPECPATH).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"

# collect_submodules() below runs as plain Python right now, while this
# spec file is being parsed — it needs `app.services` importable in *this*
# process to walk its submodules. `pathex` passed to Analysis() further
# down only affects PyInstaller's own later static-analysis pass, not this
# line, so without this it silently finds nothing and app.* never makes it
# into the build at all (no build-time error — just a runtime
# "ModuleNotFoundError: No module named 'app'" when you launch the .exe).
sys.path.insert(0, str(BACKEND_DIR))

datas = []
binaries = []
# Only the parts of the backend the desktop app actually touches:
# app.schemas, app.config, and app.services.* — not statically discoverable
# by PyInstaller since desktop/main.py only reaches them via a runtime
# sys.path.insert(), so they're spelled out via pathex below and listed
# explicitly here. Deliberately NOT collect_submodules("app") as a whole:
# that would also pull in app.main and app.routers.* — the FastAPI app and
# its HTTP routes — which the desktop build has no business bundling. The
# desktop app is Qt + the plain-Python service layer only, no HTTP server.
hiddenimports = ["app.schemas", "app.config"] + collect_submodules("app.services")

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
    # belt-and-suspenders: nothing the desktop app imports needs these, so
    # this should be a no-op — but excluding them outright guarantees no
    # HTTP server code ends up in a "just Qt" desktop build even if a
    # future change accidentally introduces a transitive import of one.
    excludes=["fastapi", "starlette", "uvicorn", "multipart", "app.main", "app.routers"],
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
