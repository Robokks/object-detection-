"""Background-thread helpers so long-running service calls (inference, a
first-time model download, CV passes, file I/O for many images) don't
freeze the Qt UI thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class FunctionWorker(QThread):
    """Runs any callable on a background thread and reports the result back
    via Qt signals (queued to the GUI thread automatically)."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as e:  # noqa: BLE001 — surfaced to the UI, not swallowed
            self.failed.emit(str(e))
        else:
            self.succeeded.emit(result)


def import_annotated_images(dataset_name: str, image_paths: list[Path], class_name: str) -> list[dict]:
    """Runs red_import_service + dataset_service for each file — the same
    per-image steps the web backend's /import-annotated endpoint performs,
    called directly instead of over HTTP.
    """
    from app.services import dataset_service, red_import_service
    from app.services.dataset_service import DatasetError

    results = []
    for path in image_paths:
        try:
            content = path.read_bytes()
            clean_bytes, shapes = red_import_service.extract_annotated_image(content, class_name)
            added = dataset_service.add_image(dataset_name, path.name, clean_bytes)
            dataset_service.save_annotations(dataset_name, added["image_id"], added["width"], added["height"], shapes)
            results.append({"filename": path.name, "image_id": added["image_id"], "shapes_found": len(shapes)})
        except (ValueError, DatasetError) as e:
            results.append({"filename": path.name, "error": str(e)})
    return results
