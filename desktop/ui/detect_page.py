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
from app.services import dataset_service, model_service, report_service
from app.services.dataset_service import DatasetError

from .canvas import InteractiveCanvas
from .workers import FunctionWorker

RESULT_COLUMNS = [
    "Class",
    "Shape",
    "Confidence",
    "Position (X, Y)",
    "Angle",
    "Left (X, Y)",
    "Right (X, Y)",
    "Top (X, Y)",
    "Bottom (X, Y)",
    "Box X",
    "Box Y",
    "Width",
    "Height",
]

SOURCE_LABELS = {"pretrained": "Pretrained", "trained": "Trained by you", "imported": "Imported"}


class DetectPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path: Path | None = None
        self._detect_worker: FunctionWorker | None = None
        self._import_worker: FunctionWorker | None = None
        self._last_boxes: list[Shape] = []
        self._syncing_selection = False

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

        roi_row = QHBoxLayout()
        self.roi_btn = QPushButton("Set ROI")
        self.roi_btn.setCheckable(True)
        self.roi_btn.setToolTip("Drag a rectangle on the image below to limit detections to that region.")
        self.roi_btn.toggled.connect(self._on_roi_mode_toggled)
        self.clear_roi_btn = QPushButton("Clear ROI")
        self.clear_roi_btn.clicked.connect(self._on_clear_roi)
        self.roi_status_label = QLabel("ROI: full image")
        roi_row.addWidget(self.roi_btn)
        roi_row.addWidget(self.clear_roi_btn)
        roi_row.addWidget(self.roi_status_label, stretch=1)
        run_layout.addLayout(roi_row)
        root.addWidget(run_box)

        # --- results ---
        results_box = QGroupBox("Result")
        results_layout = QVBoxLayout(results_box)
        self.canvas = InteractiveCanvas(read_only=True)
        self.canvas.setMinimumHeight(360)
        self.canvas.roiChanged.connect(self._on_roi_changed)
        self.canvas.shapesChanged.connect(self._on_canvas_shapes_edited)
        results_layout.addWidget(self.canvas)

        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.canvas.shapeSelected.connect(self._on_canvas_shape_selected)
        results_layout.addWidget(self.results_table)
        root.addWidget(results_box, stretch=1)

        # --- fix wrong detections, save as training data ---
        fix_box = QGroupBox("Fix wrong detections & save for fine-tuning")
        fix_layout = QVBoxLayout(fix_box)
        fix_hint = QLabel(
            "Select a wrong detection above (click it on the image or in the table), then Enable corrections "
            "to delete it or redraw it correctly. When you're happy with this image, save it into a dataset — "
            "then fine-tune from the Train tab once you've corrected a few."
        )
        fix_hint.setWordWrap(True)
        fix_layout.addWidget(fix_hint)

        correction_row = QHBoxLayout()
        self.correction_btn = QPushButton("Enable corrections")
        self.correction_btn.setCheckable(True)
        self.correction_btn.toggled.connect(self._on_correction_mode_toggled)
        self.correction_tool_buttons: dict[str, QPushButton] = {}
        for tool_id, label in (("box", "Box"), ("rotated_box", "Rotated Box"), ("ellipse", "Ellipse"), ("polygon", "Pen")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked, t=tool_id: self._on_correction_tool_selected(t))
            correction_row.addWidget(btn)
            self.correction_tool_buttons[tool_id] = btn
        self.correction_tool_buttons["box"].setChecked(True)
        self.delete_shape_btn = QPushButton("Delete selected shape")
        self.delete_shape_btn.setEnabled(False)
        self.delete_shape_btn.clicked.connect(lambda: self.canvas.delete_selected())
        correction_row.addWidget(self.delete_shape_btn)
        correction_row.addWidget(self.correction_btn)
        fix_layout.addLayout(correction_row)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Class"))
        self.correction_class_combo = QComboBox()
        self.correction_class_combo.setEditable(True)
        self.correction_class_combo.currentTextChanged.connect(self.canvas.set_active_class)
        save_row.addWidget(self.correction_class_combo)
        save_row.addWidget(QLabel("Save to dataset"))
        self.correction_dataset_combo = QComboBox()
        self.correction_dataset_combo.currentTextChanged.connect(self._on_correction_dataset_changed)
        save_row.addWidget(self.correction_dataset_combo, stretch=1)
        self.save_correction_btn = QPushButton("Save corrected image to dataset")
        self.save_correction_btn.clicked.connect(self._on_save_correction)
        save_row.addWidget(self.save_correction_btn)
        fix_layout.addLayout(save_row)

        self.correction_status = QLabel("")
        self.correction_status.setWordWrap(True)
        fix_layout.addWidget(self.correction_status)
        root.addWidget(fix_box)

        self.refresh_models()
        self.refresh_correction_datasets()

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
            self.canvas.load_image(self._image_path)  # also resets any ROI from a previous image
            self.canvas.set_shapes([])
            self.results_table.setRowCount(0)
            self._last_boxes = []
            self.roi_btn.setChecked(False)
            self.roi_status_label.setText("ROI: full image")
            self.correction_btn.setChecked(False)  # also resets canvas.read_only via the toggled handler
            self.correction_status.setText("")

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
        self._last_boxes = result.boxes
        self.canvas.set_classes(list({b.class_name for b in result.boxes}))
        self._apply_roi_filter()

    def _on_detect_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run detection")
        QMessageBox.warning(self, "Detection failed", message)

    # ---- selection sync (canvas <-> results table) --------------------

    def _on_canvas_shape_selected(self, index) -> None:
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            if index is None:
                self.results_table.clearSelection()
            else:
                self.results_table.selectRow(index)
        finally:
            self._syncing_selection = False

    def _on_table_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        rows = {idx.row() for idx in self.results_table.selectedIndexes()}
        self._syncing_selection = True
        try:
            self.canvas.select_shape(next(iter(rows)) if rows else None)
        finally:
            self._syncing_selection = False

    # ---- ROI ---------------------------------------------------------

    def _on_roi_mode_toggled(self, checked: bool) -> None:
        self.canvas.set_roi_mode(checked)
        self.roi_btn.setText("Drawing ROI… drag on image" if checked else "Set ROI")
        self.correction_btn.setEnabled(not checked)

    def _on_clear_roi(self) -> None:
        self.canvas.clear_roi()

    def _on_roi_changed(self, roi) -> None:
        if roi is not None:
            self.roi_btn.setChecked(False)  # one drag defines it; click "Set ROI" again to redraw
        self._apply_roi_filter()

    def _apply_roi_filter(self) -> None:
        roi = self.canvas.roi()
        boxes = self._filter_by_roi(self._last_boxes, roi)
        self.canvas.set_shapes(boxes)
        self._populate_results_table(boxes)
        self.roi_status_label.setText(self._roi_status_text(roi, len(boxes), len(self._last_boxes)))

    @staticmethod
    def _roi_status_text(roi, kept: int, total: int) -> str:
        if roi is None:
            return "ROI: full image"
        x, y, w, h = roi
        text = f"ROI: ({x:.0f}, {y:.0f}) to ({x + w:.0f}, {y + h:.0f})"
        if total:
            text += f" — {kept} of {total} detections inside"
        return text

    @staticmethod
    def _filter_by_roi(boxes: list[Shape], roi) -> list[Shape]:
        if roi is None:
            return boxes
        rx, ry, rw, rh = roi

        def inside(b: Shape) -> bool:
            cx = b.center_x if b.center_x is not None else b.x + b.width / 2
            cy = b.center_y if b.center_y is not None else b.y + b.height / 2
            return rx <= cx <= rx + rw and ry <= cy <= ry + rh

        return [b for b in boxes if inside(b)]

    # ---- fix wrong detections, save for fine-tuning --------------------

    def _on_canvas_shapes_edited(self, shapes: list[Shape]) -> None:
        # fires when a correction (delete/redraw) changes the canvas's shapes;
        # that edited set becomes the new source of truth for this image.
        self._last_boxes = list(shapes)
        self._populate_results_table(shapes)
        self.roi_status_label.setText("ROI: full image")

    def _on_correction_mode_toggled(self, checked: bool) -> None:
        self.canvas.set_read_only(not checked)
        self.roi_btn.setEnabled(not checked)
        self.clear_roi_btn.setEnabled(not checked)
        for btn in self.correction_tool_buttons.values():
            btn.setEnabled(checked)
        self.delete_shape_btn.setEnabled(checked)
        self.correction_btn.setText("Finish corrections" if checked else "Enable corrections")
        if checked:
            self.canvas.clear_roi()
            self.roi_btn.setChecked(False)
            active_tool = next((t for t, b in self.correction_tool_buttons.items() if b.isChecked()), "box")
            self.canvas.set_tool(active_tool)
            self.canvas.set_active_class(self.correction_class_combo.currentText())

    def _on_correction_tool_selected(self, tool_id: str) -> None:
        for t, btn in self.correction_tool_buttons.items():
            btn.setChecked(t == tool_id)
        self.canvas.set_tool(tool_id)

    def refresh_correction_datasets(self, select_name: str = "") -> None:
        current = select_name or self.correction_dataset_combo.currentData()
        self.correction_dataset_combo.blockSignals(True)
        self.correction_dataset_combo.clear()
        self.correction_dataset_combo.addItem("-- choose dataset --", "")
        for d in dataset_service.list_datasets():
            self.correction_dataset_combo.addItem(d.name, d.name)
        self.correction_dataset_combo.blockSignals(False)
        if current:
            idx = self.correction_dataset_combo.findData(current)
            if idx >= 0:
                self.correction_dataset_combo.setCurrentIndex(idx)
        self._refresh_correction_class_options()

    def _on_correction_dataset_changed(self, _text: str) -> None:
        self._refresh_correction_class_options()

    def _refresh_correction_class_options(self) -> None:
        dataset_name = self.correction_dataset_combo.currentData()
        classes: list[str] = []
        if dataset_name:
            try:
                meta = dataset_service.get_dataset(dataset_name)
                classes = list(meta.get("classes", []))
            except DatasetError:
                classes = []
        for b in self._last_boxes:
            if b.class_name not in classes:
                classes.append(b.class_name)

        current = self.correction_class_combo.currentText()
        self.correction_class_combo.blockSignals(True)
        self.correction_class_combo.clear()
        self.correction_class_combo.addItems(classes)
        if current and current in classes:
            self.correction_class_combo.setCurrentText(current)
        elif classes:
            self.correction_class_combo.setCurrentIndex(0)
        self.correction_class_combo.blockSignals(False)
        self.canvas.set_active_class(self.correction_class_combo.currentText())

    def _on_save_correction(self) -> None:
        dataset_name = self.correction_dataset_combo.currentData()
        if not dataset_name:
            QMessageBox.information(self, "Choose a dataset", "Pick a destination dataset first.")
            return
        if not self._image_path:
            QMessageBox.information(self, "Choose an image", "Run detection on an image first.")
            return

        shapes = [s.model_copy(update={"confidence": None}) for s in self.canvas.shapes()]
        try:
            added = dataset_service.add_image(dataset_name, self._image_path.name, self._image_path.read_bytes())
            dataset_service.save_annotations(dataset_name, added["image_id"], added["width"], added["height"], shapes)
        except DatasetError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return

        self.correction_status.setText(
            f'Saved this image with {len(shapes)} corrected label(s) to dataset "{dataset_name}". '
            "Repeat for other wrongly-detected images, then fine-tune from the Train tab."
        )
        self.refresh_correction_datasets(select_name=dataset_name)

    def _populate_results_table(self, boxes: list[Shape]) -> None:
        # Sourced from report_service.shape_to_report — the same helper the
        # standalone scripts/detect_to_json.py uses — so the table and the
        # JSON export always agree on these fields.
        self.results_table.setRowCount(len(boxes))
        for row, b in enumerate(boxes):
            r = report_service.shape_to_report(b)
            confidence_text = f"{r['confidence'] * 100:.1f}%" if r["confidence"] is not None else "-"
            angle_text = f"{r['angle_degrees']:.1f}°" if r["angle_degrees"] is not None else "-"
            values = [
                r["class_name"],
                r["shape"],
                confidence_text,
                f"({r['position']['x']:.0f}, {r['position']['y']:.0f})",
                angle_text,
                f"({r['left']['x']:.0f}, {r['left']['y']:.0f})",
                f"({r['right']['x']:.0f}, {r['right']['y']:.0f})",
                f"({r['top']['x']:.0f}, {r['top']['y']:.0f})",
                f"({r['bottom']['x']:.0f}, {r['bottom']['y']:.0f})",
                f"{r['box']['x']:.0f}",
                f"{r['box']['y']:.0f}",
                f"{r['box']['width']:.0f}",
                f"{r['box']['height']:.0f}",
            ]
            for col, value in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(value))
