"""Dataset & Labeling page — desktop equivalent of DatasetPage.tsx."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.schemas import CreateDatasetRequest, Shape
from app.services import dataset_service
from app.services.dataset_service import DatasetError

from .canvas import InteractiveCanvas
from .workers import FunctionWorker, import_annotated_images

TOOLS = [
    ("box", "Box"),
    ("rotated_box", "Rotated Box"),
    ("ellipse", "Ellipse"),
    ("polygon", "Pen"),
]

THUMB_SIZE = QSize(110, 80)


class DatasetPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset_name: str = ""
        self._meta: dict = {}
        self._active_image_id: str = ""
        self._suggest_worker: FunctionWorker | None = None
        self._import_worker: FunctionWorker | None = None

        root = QVBoxLayout(self)

        desc = QLabel(
            "Create a dataset, upload images, then label objects with the box, rotated box, ellipse, or pen tool."
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        # --- create dataset ---
        create_box = QGroupBox("Create dataset")
        create_layout = QHBoxLayout(create_box)
        self.name_edit = QLineEdit(placeholderText="dataset name")
        self.classes_edit = QLineEdit(placeholderText="classes, comma separated (e.g. cat, dog)")
        create_btn = QPushButton("Create")
        create_btn.clicked.connect(self._on_create_dataset)
        create_layout.addWidget(self.name_edit)
        create_layout.addWidget(self.classes_edit)
        create_layout.addWidget(create_btn)
        root.addWidget(create_box)

        # --- select dataset ---
        select_box = QGroupBox("Select dataset")
        select_layout = QHBoxLayout(select_box)
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_selected)
        select_layout.addWidget(self.dataset_combo)
        root.addWidget(select_box)

        # --- upload / import ---
        upload_box = QGroupBox("Upload images")
        upload_layout = QHBoxLayout(upload_box)
        upload_btn = QPushButton("Choose images…")
        upload_btn.clicked.connect(self._on_upload_images)
        upload_layout.addWidget(upload_btn)
        upload_layout.addStretch()
        root.addWidget(upload_box)

        import_box = QGroupBox("Import pre-annotated images (hand-drawn red outlines)")
        import_layout = QVBoxLayout(import_box)
        import_hint = QLabel(
            "Each red outline is extracted as a shape labeled with the active class below, "
            "and the red lines are removed from the stored image."
        )
        import_hint.setWordWrap(True)
        import_btn = QPushButton("Choose annotated images…")
        import_btn.clicked.connect(self._on_import_annotated)
        self.import_status = QLabel("")
        self.import_status.setWordWrap(True)
        import_layout.addWidget(import_hint)
        import_layout.addWidget(import_btn)
        import_layout.addWidget(self.import_status)
        root.addWidget(import_box)

        # --- classes ---
        classes_box = QGroupBox("Classes")
        classes_layout = QVBoxLayout(classes_box)
        self.classes_list = QListWidget()
        self.classes_list.setMaximumHeight(90)
        self.classes_list.currentTextChanged.connect(self._on_active_class_changed)
        add_class_row = QHBoxLayout()
        self.new_class_edit = QLineEdit(placeholderText="new class name")
        add_class_btn = QPushButton("Add class")
        add_class_btn.clicked.connect(self._on_add_class)
        add_class_row.addWidget(self.new_class_edit)
        add_class_row.addWidget(add_class_btn)
        classes_layout.addWidget(self.classes_list)
        classes_layout.addLayout(add_class_row)
        root.addWidget(classes_box)

        # --- images ---
        images_box = QGroupBox("Images")
        images_layout = QVBoxLayout(images_box)
        self.images_list = QListWidget()
        self.images_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.images_list.setIconSize(THUMB_SIZE)
        self.images_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.images_list.setMaximumHeight(140)
        self.images_list.currentItemChanged.connect(self._on_image_selected)
        delete_image_btn = QPushButton("Delete selected image")
        delete_image_btn.clicked.connect(self._on_delete_image)
        images_layout.addWidget(self.images_list)
        images_layout.addWidget(delete_image_btn)
        root.addWidget(images_box)

        # --- label image ---
        label_box = QGroupBox("Label image")
        label_layout = QVBoxLayout(label_box)

        tool_row = QHBoxLayout()
        self.tool_buttons: dict[str, QPushButton] = {}
        for tool_id, label in TOOLS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, t=tool_id: self._on_tool_selected(t))
            tool_row.addWidget(btn)
            self.tool_buttons[tool_id] = btn
        self.tool_buttons["box"].setChecked(True)
        suggest_btn = QPushButton("Suggest cylinders")
        suggest_btn.clicked.connect(self._on_suggest_cylinders)
        tool_row.addWidget(suggest_btn)
        tool_row.addStretch()
        delete_shape_btn = QPushButton("Delete selected shape")
        delete_shape_btn.clicked.connect(lambda: self.canvas.delete_selected())
        tool_row.addWidget(delete_shape_btn)
        label_layout.addLayout(tool_row)

        self.suggest_status = QLabel("")
        self.suggest_status.setWordWrap(True)
        label_layout.addWidget(self.suggest_status)

        self.canvas = InteractiveCanvas()
        self.canvas.setMinimumHeight(420)
        self.canvas.shapesChanged.connect(self._on_shapes_changed)
        label_layout.addWidget(self.canvas)

        root.addWidget(label_box, stretch=1)

        self.refresh_datasets()

    # ---- dataset list -------------------------------------------------

    def refresh_datasets(self, select_name: str = "") -> None:
        datasets = dataset_service.list_datasets()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem("-- choose dataset --", "")
        for d in datasets:
            self.dataset_combo.addItem(f"{d.name} ({d.annotated_count}/{d.image_count} annotated)", d.name)
        self.dataset_combo.blockSignals(False)
        if select_name:
            idx = self.dataset_combo.findData(select_name)
            if idx >= 0:
                self.dataset_combo.setCurrentIndex(idx)
                self._on_dataset_selected()

    def _on_create_dataset(self) -> None:
        name = self.name_edit.text().strip()
        classes = [c.strip() for c in self.classes_edit.text().split(",") if c.strip()]
        try:
            dataset_service.create_dataset(CreateDatasetRequest(name=name, classes=classes))
        except DatasetError as e:
            QMessageBox.warning(self, "Could not create dataset", str(e))
            return
        self.name_edit.clear()
        self.classes_edit.clear()
        self.refresh_datasets(select_name=name)

    def _on_dataset_selected(self) -> None:
        name = self.dataset_combo.currentData()
        self._dataset_name = name or ""
        self._load_detail()

    def _load_detail(self) -> None:
        self.images_list.clear()
        self.classes_list.clear()
        self.canvas.set_shapes([])
        if not self._dataset_name:
            self._meta = {}
            return
        try:
            self._meta = dataset_service.get_dataset(self._dataset_name)
        except DatasetError as e:
            QMessageBox.warning(self, "Could not load dataset", str(e))
            return

        classes = self._meta.get("classes", [])
        self.classes_list.addItems(classes)
        if classes:
            self.classes_list.setCurrentRow(0)
        self.canvas.set_classes(classes)

        for image_id, entry in self._meta.get("images", {}).items():
            self._add_image_item(image_id, entry)
        if self.images_list.count() > 0:
            self.images_list.setCurrentRow(0)

    def _add_image_item(self, image_id: str, entry: dict) -> None:
        path = Path(dataset_service.image_path(self._dataset_name, image_id))
        pixmap = QPixmap(str(path)).scaled(
            THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        item = QListWidgetItem(QIcon(pixmap), f"{len(entry.get('shapes', []))} shape(s)")
        item.setData(Qt.ItemDataRole.UserRole, image_id)
        self.images_list.addItem(item)

    # ---- classes --------------------------------------------------------

    def _on_active_class_changed(self, class_name: str) -> None:
        self.canvas.set_active_class(class_name)

    def _on_add_class(self) -> None:
        name = self.new_class_edit.text().strip()
        if not name or not self._dataset_name:
            return
        self.classes_list.addItem(name)
        self.classes_list.setCurrentRow(self.classes_list.count() - 1)
        self.canvas.set_classes([self.classes_list.item(i).text() for i in range(self.classes_list.count())])
        self.new_class_edit.clear()

    # ---- images -----------------------------------------------------------

    def _on_upload_images(self) -> None:
        if not self._dataset_name:
            QMessageBox.information(self, "Choose a dataset", "Select or create a dataset first.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Choose images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not files:
            return
        last_id = ""
        for f in files:
            path = Path(f)
            try:
                added = dataset_service.add_image(self._dataset_name, path.name, path.read_bytes())
                last_id = added["image_id"]
            except DatasetError as e:
                QMessageBox.warning(self, "Upload failed", f"{path.name}: {e}")
        self._load_detail()
        self._select_image(last_id)
        self.refresh_datasets(select_name=self._dataset_name)

    def _on_delete_image(self) -> None:
        item = self.images_list.currentItem()
        if not item or not self._dataset_name:
            return
        image_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            dataset_service.delete_image(self._dataset_name, image_id)
        except DatasetError as e:
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        self._load_detail()
        self.refresh_datasets(select_name=self._dataset_name)

    def _on_image_selected(self, current: QListWidgetItem, _previous) -> None:
        if current is None:
            self._active_image_id = ""
            self.canvas.set_shapes([])
            return
        image_id = current.data(Qt.ItemDataRole.UserRole)
        self._select_image(image_id)

    def _select_image(self, image_id: str) -> None:
        if not image_id or image_id not in self._meta.get("images", {}):
            return
        self._active_image_id = image_id
        entry = self._meta["images"][image_id]
        path = dataset_service.image_path(self._dataset_name, image_id)
        self.canvas.load_image(Path(path))
        self.canvas.set_classes(self._meta.get("classes", []))
        shapes = [Shape(**s) for s in entry.get("shapes", [])]
        self.canvas.set_shapes(shapes)
        for i in range(self.images_list.count()):
            if self.images_list.item(i).data(Qt.ItemDataRole.UserRole) == image_id:
                self.images_list.setCurrentRow(i)
                break

    # ---- labeling ---------------------------------------------------------

    def _on_tool_selected(self, tool_id: str) -> None:
        for t, btn in self.tool_buttons.items():
            btn.setChecked(t == tool_id)
        self.canvas.set_tool(tool_id)

    def _on_shapes_changed(self, shapes: list[Shape]) -> None:
        if not self._dataset_name or not self._active_image_id:
            return
        entry = self._meta["images"][self._active_image_id]
        try:
            dataset_service.save_annotations(
                self._dataset_name, self._active_image_id, entry["width"], entry["height"], shapes
            )
        except DatasetError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        entry["shapes"] = [s.model_dump() for s in shapes]
        for i in range(self.images_list.count()):
            item = self.images_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self._active_image_id:
                item.setText(f"{len(shapes)} shape(s)")
                break
        self.refresh_datasets(select_name=self._dataset_name)

    def _on_suggest_cylinders(self) -> None:
        if not self._dataset_name or not self._active_image_id:
            QMessageBox.information(self, "Choose an image", "Select an image to label first.")
            return
        active_class = self.classes_list.currentItem().text() if self.classes_list.currentItem() else ""
        if not active_class:
            QMessageBox.information(self, "Choose a class", "Select a class first.")
            return

        from app.services import autolabel_service

        path = dataset_service.image_path(self._dataset_name, self._active_image_id)
        self.suggest_status.setText("Scanning…")
        worker = FunctionWorker(autolabel_service.suggest_cylinder_shapes, path, active_class)
        worker.succeeded.connect(self._on_suggest_succeeded)
        worker.failed.connect(lambda msg: self.suggest_status.setText(f"Suggest failed: {msg}"))
        self._suggest_worker = worker
        worker.start()

    def _on_suggest_succeeded(self, suggested: list[Shape]) -> None:
        combined = self.canvas.shapes() + suggested
        self.canvas.set_shapes(combined)
        self._on_shapes_changed(combined)
        active_class = self.classes_list.currentItem().text() if self.classes_list.currentItem() else ""
        if suggested:
            self.suggest_status.setText(
                f'Found {len(suggested)} candidate(s), labeled "{active_class}" — review and delete any that aren\'t real.'
            )
        else:
            self.suggest_status.setText("No candidates found on this image.")

    def _on_import_annotated(self) -> None:
        if not self._dataset_name:
            QMessageBox.information(self, "Choose a dataset", "Select or create a dataset first.")
            return
        active_class = self.classes_list.currentItem().text() if self.classes_list.currentItem() else ""
        if not active_class:
            QMessageBox.information(self, "Choose a class", "Select a class first.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose annotated images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not files:
            return
        self.import_status.setText("Extracting outlines…")
        worker = FunctionWorker(import_annotated_images, self._dataset_name, [Path(f) for f in files], active_class)
        worker.succeeded.connect(self._on_import_succeeded)
        worker.failed.connect(lambda msg: self.import_status.setText(f"Import failed: {msg}"))
        self._import_worker = worker
        worker.start()

    def _on_import_succeeded(self, results: list[dict]) -> None:
        total_shapes = sum(r.get("shapes_found", 0) for r in results)
        failed = [r for r in results if r.get("error")]
        ok = len(results) - len(failed)
        msg = f'Imported {ok}/{len(results)} image(s), extracted {total_shapes} shape(s).'
        if failed:
            msg += " Failed: " + ", ".join(f["filename"] for f in failed)
        self.import_status.setText(msg)
        self._load_detail()
        self.refresh_datasets(select_name=self._dataset_name)
