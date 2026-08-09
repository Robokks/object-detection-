"""Detect page — desktop equivalent of DetectPage.tsx."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.schemas import DetectionResult, ModelTask, Shape
from app.services import model_service

from .canvas import InteractiveCanvas
from .workers import FunctionWorker

RESULT_COLUMNS = ["Class", "Shape", "Confidence", "Position (X, Y)", "Box X", "Box Y", "Width", "Height"]

SOURCE_LABELS = {"pretrained": "Pretrained", "trained": "Trained by you", "imported": "Imported"}


class DetectPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path: Path | None = None
        self._detect_worker: FunctionWorker | None = None
        self._import_worker: FunctionWorker | None = None

        root = QVBoxLayout(self)

        desc = QLabel("Pick a model, choose an image, and run detection to see each object's class, confidence, and position.")
        desc.setWordWrap(True)
        root.addWidget(desc)

        # --- import model ---
        import_box = QGroupBox("Import a model")
        import_layout = QVBoxLayout(import_box)
        import_hint = QLabel(
            "Bring your own checkpoint (a YOLO variant, SAM, or anything Ultralytics can load). "
            "Class names are read automatically from the checkpoint when available."
        )
        import_hint.setWordWrap(True)
        import_layout.addWidget(import_hint)

        import_row = QHBoxLayout()
        self.import_file_btn = QPushButton("Choose weights file (.pt)…")
        self.import_file_btn.clicked.connect(self._on_choose_weights)
        self._import_file_path: Path | None = None
        self.import_label_edit = QLineEdit(placeholderText="display name")
        self.import_task_combo = QComboBox()
        self.import_task_combo.addItem("Object detection (boxes)", "detect")
        self.import_task_combo.addItem("Instance segmentation (masks)", "segment")
        self.import_task_combo.addItem("SAM-style (class-agnostic)", "sam")
        import_btn = QPushButton("Import model")
        import_btn.clicked.connect(self._on_import_model)
        import_row.addWidget(self.import_file_btn)
        import_row.addWidget(self.import_label_edit)
        import_row.addWidget(self.import_task_combo)
        import_row.addWidget(import_btn)
        import_layout.addLayout(import_row)
        self.import_status = QLabel("")
        import_layout.addWidget(self.import_status)
        root.addWidget(import_box)

        # --- model + confidence + image ---
        run_box = QGroupBox()
        run_layout = QVBoxLayout(run_box)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        model_row.addWidget(self.model_combo, stretch=1)
        model_row.addWidget(QLabel("Confidence"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(5, 95)
        self.confidence_slider.setValue(25)
        self.confidence_label = QLabel("0.25")
        self.confidence_slider.valueChanged.connect(lambda v: self.confidence_label.setText(f"{v / 100:.2f}"))
        model_row.addWidget(self.confidence_slider)
        model_row.addWidget(self.confidence_label)
        run_layout.addLayout(model_row)

        image_row = QHBoxLayout()
        self.choose_image_btn = QPushButton("Choose image…")
        self.choose_image_btn.clicked.connect(self._on_choose_image)
        self.image_path_label = QLabel("No image chosen")
        self.run_btn = QPushButton("Run detection")
        self.run_btn.clicked.connect(self._on_run_detection)
        image_row.addWidget(self.choose_image_btn)
        image_row.addWidget(self.image_path_label, stretch=1)
        image_row.addWidget(self.run_btn)
        run_layout.addLayout(image_row)
        root.addWidget(run_box)

        # --- results ---
        results_box = QGroupBox("Result")
        results_layout = QVBoxLayout(results_box)
        self.canvas = InteractiveCanvas(read_only=True)
        self.canvas.setMinimumHeight(360)
        results_layout.addWidget(self.canvas)

        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_layout.addWidget(self.results_table)
        root.addWidget(results_box, stretch=1)

        self.refresh_models()

    def refresh_models(self, select_id: str = "") -> None:
        current = select_id or self.model_combo.currentData()
        self.model_combo.clear()
        for m in model_service.list_models():
            self.model_combo.addItem(f"[{SOURCE_LABELS.get(m.source, m.source)}] {m.label}", m.id)
        if current:
            idx = self.model_combo.findData(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

    def _on_choose_weights(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Choose weights file", "", "PyTorch weights (*.pt)")
        if f:
            self._import_file_path = Path(f)
            self.import_file_btn.setText(self._import_file_path.name)

    def _on_import_model(self) -> None:
        if not self._import_file_path:
            QMessageBox.information(self, "Choose a file", "Choose a .pt weights file first.")
            return
        label = self.import_label_edit.text().strip() or self._import_file_path.stem
        task: ModelTask = self.import_task_combo.currentData()
        content = self._import_file_path.read_bytes()
        self.import_status.setText("Importing…")
        worker = FunctionWorker(model_service.import_model, self._import_file_path.name, content, label, task)
        worker.succeeded.connect(self._on_import_succeeded)
        worker.failed.connect(lambda msg: self.import_status.setText(f"Import failed: {msg}"))
        self._import_worker = worker
        worker.start()

    def _on_import_succeeded(self, model_info) -> None:
        self.import_status.setText(f'Imported "{model_info.label}".')
        self.refresh_models(select_id=model_info.id)

    def _on_choose_image(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if f:
            self._image_path = Path(f)
            self.image_path_label.setText(self._image_path.name)
            self.canvas.load_image(self._image_path)
            self.canvas.set_shapes([])
            self.results_table.setRowCount(0)

    def _on_run_detection(self) -> None:
        model_id = self.model_combo.currentData()
        if not model_id or not self._image_path:
            QMessageBox.information(self, "Missing info", "Choose a model and an image first.")
            return
        confidence = self.confidence_slider.value() / 100
        content = self._image_path.read_bytes()
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Detecting…")

        from app.services import detect_service

        worker = FunctionWorker(detect_service.run_detection, model_id, content, confidence)
        worker.succeeded.connect(self._on_detect_succeeded)
        worker.failed.connect(self._on_detect_failed)
        self._detect_worker = worker
        worker.start()

    def _on_detect_succeeded(self, result: DetectionResult) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run detection")
        self.canvas.set_classes(list({b.class_name for b in result.boxes}))
        self.canvas.set_shapes(result.boxes)
        self._populate_results_table(result.boxes)

    def _on_detect_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run detection")
        QMessageBox.warning(self, "Detection failed", message)

    def _populate_results_table(self, boxes: list[Shape]) -> None:
        self.results_table.setRowCount(len(boxes))
        for row, b in enumerate(boxes):
            cx = b.center_x if b.center_x is not None else b.x + b.width / 2
            cy = b.center_y if b.center_y is not None else b.y + b.height / 2
            shape_kind = "mask" if len(b.points) >= 3 else "box"
            confidence_text = f"{b.confidence * 100:.1f}%" if b.confidence is not None else "-"
            values = [
                b.class_name,
                shape_kind,
                confidence_text,
                f"({cx:.0f}, {cy:.0f})",
                f"{b.x:.0f}",
                f"{b.y:.0f}",
                f"{b.width:.0f}",
                f"{b.height:.0f}",
            ]
            for col, value in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(value))
