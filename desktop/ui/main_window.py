from PySide6.QtWidgets import QMainWindow, QTabWidget

from .dataset_page import DatasetPage
from .detect_page import DetectPage
from .train_page import TrainPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Object Detection Studio")

        self.dataset_page = DatasetPage()
        self.train_page = TrainPage()
        self.detect_page = DetectPage()

        tabs = QTabWidget()
        tabs.addTab(self.dataset_page, "1. Dataset")
        tabs.addTab(self.train_page, "2. Train")
        tabs.addTab(self.detect_page, "3. Detect")
        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)

    def _on_tab_changed(self, index: int) -> None:
        widget = self.centralWidget().widget(index)
        if widget is self.train_page:
            self.train_page.refresh_datasets()
        elif widget is self.detect_page:
            self.detect_page.refresh_models()
