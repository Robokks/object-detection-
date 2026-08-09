"""Train page — desktop equivalent of TrainPage.tsx.

train_service already runs training in its own background thread (it was
built that way for the web backend too), so this page just calls it
directly and polls for progress with a QTimer — no worker thread needed
for the "start" call itself.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.schemas import TrainRequest
from app.services import dataset_service, model_service, train_service
from app.services.train_service import TrainError

POLL_INTERVAL_MS = 3000

JOB_COLUMNS = ["Run name", "State", "Mode", "Progress", "Epoch", "Message"]


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)

        desc = QLabel(
            "Fine-tune a pretrained YOLO checkpoint on a labeled dataset, or train from scratch. "
            "Labels are polygon outlines (a box is a 4-point outline too), so training always targets "
            "the instance-segmentation variant of the chosen architecture (e.g. yolov8n-seg)."
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        config_box = QGroupBox("Configure run")
        form = QFormLayout(config_box)

        self.dataset_combo = QComboBox()
        form.addRow("Dataset", self.dataset_combo)

        self.run_name_edit = QLineEdit(placeholderText="e.g. my-model-v1")
        form.addRow("Run name", self.run_name_edit)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Fine-tune pretrained (recommended)", "finetune")
        self.mode_combo.addItem("Train from scratch", "scratch")
        form.addRow("Mode", self.mode_combo)

        self.base_model_combo = QComboBox()
        self.base_model_combo.addItem("yolov8n (fastest)", "yolov8n")
        self.base_model_combo.addItem("yolov8s (more accurate, slower)", "yolov8s")
        self.base_model_combo.addItem("yolov8m", "yolov8m")
        form.addRow("Base architecture", self.base_model_combo)

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setToolTip(
            "Finetune mode only: continue training from this checkpoint (e.g. a previous run, or "
            "a model you imported) instead of stock COCO weights. Ignored in scratch mode."
        )
        form.addRow("Continue from checkpoint", self.checkpoint_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(50)
        form.addRow("Epochs", self.epochs_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        form.addRow("Image size", self.imgsz_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        form.addRow("Batch size", self.batch_spin)

        start_btn = QPushButton("Start training")
        start_btn.clicked.connect(self._on_start_training)
        form.addRow("", start_btn)

        root.addWidget(config_box)

        jobs_box = QGroupBox("Runs")
        jobs_layout = QVBoxLayout(jobs_box)
        self.jobs_table = QTableWidget(0, len(JOB_COLUMNS))
        self.jobs_table.setHorizontalHeaderLabels(JOB_COLUMNS)
        self.jobs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.jobs_table.verticalHeader().setVisible(False)
        self.jobs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        jobs_layout.addWidget(self.jobs_table)
        root.addWidget(jobs_box, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_jobs)
        self._timer.start(POLL_INTERVAL_MS)

        self.refresh_datasets()
        self.refresh_checkpoints()
        self._refresh_jobs()

    def refresh_datasets(self) -> None:
        current = self.dataset_combo.currentData()
        self.dataset_combo.clear()
        for d in dataset_service.list_datasets():
            self.dataset_combo.addItem(f"{d.name} ({d.annotated_count} annotated)", d.name)
        if current:
            idx = self.dataset_combo.findData(current)
            if idx >= 0:
                self.dataset_combo.setCurrentIndex(idx)

    def refresh_checkpoints(self) -> None:
        current = self.checkpoint_combo.currentData()
        self.checkpoint_combo.blockSignals(True)
        self.checkpoint_combo.clear()
        self.checkpoint_combo.addItem("-- none, start from stock COCO weights --", "")
        for m in model_service.list_models():
            if m.source in ("trained", "imported"):
                self.checkpoint_combo.addItem(f"[{m.source}] {m.label}", m.id)
        self.checkpoint_combo.blockSignals(False)
        if current:
            idx = self.checkpoint_combo.findData(current)
            if idx >= 0:
                self.checkpoint_combo.setCurrentIndex(idx)

    def _on_start_training(self) -> None:
        dataset_name = self.dataset_combo.currentData()
        run_name = self.run_name_edit.text().strip()
        if not dataset_name or not run_name:
            QMessageBox.information(self, "Missing info", "Choose a dataset and give the run a name.")
            return
        checkpoint_id = self.checkpoint_combo.currentData()
        base_checkpoint_path = None
        if checkpoint_id:
            base_checkpoint_path, _ = model_service.resolve(checkpoint_id)
        req = TrainRequest(
            dataset_name=dataset_name,
            run_name=run_name,
            mode=self.mode_combo.currentData(),
            epochs=self.epochs_spin.value(),
            image_size=self.imgsz_spin.value(),
            batch_size=self.batch_spin.value(),
            base_model=self.base_model_combo.currentData(),
            base_checkpoint_path=base_checkpoint_path,
        )
        try:
            train_service.start_training(req)
        except TrainError as e:
            QMessageBox.warning(self, "Could not start training", str(e))
            return
        self._refresh_jobs()

    def _refresh_jobs(self) -> None:
        self.refresh_checkpoints()
        jobs = train_service.list_jobs()
        self.jobs_table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            self.jobs_table.setItem(row, 0, QTableWidgetItem(job.run_name))
            self.jobs_table.setItem(row, 1, QTableWidgetItem(job.state))
            self.jobs_table.setItem(row, 2, QTableWidgetItem(job.mode))

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(round(job.progress * 100))
            self.jobs_table.setCellWidget(row, 3, bar)

            epoch_text = f"{job.current_epoch}/{job.total_epochs}" if job.total_epochs else "-"
            self.jobs_table.setItem(row, 4, QTableWidgetItem(epoch_text))
            self.jobs_table.setItem(row, 5, QTableWidgetItem(job.message))
