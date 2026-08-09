"""Interactive image canvas: draws an image and lets the user label objects
with a box, rotated box, ellipse, or freehand pen tool — the desktop
equivalent of the web app's BoundingBoxEditor/BoundingBoxOverlay.

Scene coordinates are natural image pixel coordinates throughout; only the
QGraphicsView's own transform handles zoom/fit-to-window.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.schemas import Shape

from .shape_items import (
    LabelItem,
    RotateHandleItem,
    ShapePolygonItem,
    bbox_from_points,
    color_for_class,
    ellipse_ring,
    rect_corners,
    rotate_handle_position,
)

MIN_DRAW_SIZE = 4.0
MIN_PATH_POINT_DISTANCE = 2.0


def _shape_render_points(shape: Shape) -> list[tuple[float, float]]:
    if len(shape.points) >= 3:
        return [(p[0], p[1]) for p in shape.points]
    return rect_corners(shape.x, shape.y, shape.width, shape.height)


RoiRect = tuple[float, float, float, float]  # x, y, width, height, in scene (image pixel) coords


class InteractiveCanvas(QGraphicsView):
    shapesChanged = Signal(list)  # emits list[Shape]
    roiChanged = Signal(object)  # emits RoiRect | None
    shapeSelected = Signal(object)  # emits int index into shapes(), or None

    def __init__(self, parent=None, read_only: bool = False):
        super().__init__(parent)
        self.read_only = read_only
        self.setScene(QGraphicsScene(self))
        self.scene().selectionChanged.connect(self._on_scene_selection_changed)
        self._suspend_selection_signal = False
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(QColor("#1a1a1a")))

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._shapes: list[Shape] = []
        self._classes: list[str] = []
        self._active_class: str = ""
        self._tool: str = "box"

        self._items: list[ShapePolygonItem] = []
        self._labels: list[LabelItem] = []
        self._handles: list[RotateHandleItem | None] = []

        self._drawing = False
        self._draw_start = QPointF()
        self._draw_current = QPointF()
        self._draw_points: list[tuple[float, float]] = []
        self._draft_item: QGraphicsPolygonItem | QGraphicsEllipseItem | None = None

        self._roi_mode = False
        self._roi_drawing = False
        self._roi_start = QPointF()
        self._roi: RoiRect | None = None
        self._roi_item: QGraphicsRectItem | None = None

    # ---- configuration -------------------------------------------------

    def set_classes(self, classes: list[str]) -> None:
        self._classes = classes
        self._redraw()

    def set_active_class(self, class_name: str) -> None:
        self._active_class = class_name

    def set_tool(self, tool: str) -> None:
        self._tool = tool

    def set_read_only(self, read_only: bool) -> None:
        self.read_only = read_only

    def load_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        self.scene().clear()
        self._items, self._labels, self._handles = [], [], []
        self._roi, self._roi_item, self._roi_drawing = None, None, False
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_shapes(self, shapes: list[Shape]) -> None:
        # _redraw() below rebuilds every graphics item from scratch, which would
        # otherwise silently drop the current selection (e.g. an ROI filter
        # change re-rendering the same shapes shouldn't deselect whatever the
        # user had picked). Re-find the same Shape object (by identity, since
        # filtering/reordering may still reuse the same instances) afterward.
        selected_shape = next((self._shapes[i] for i, item in enumerate(self._items) if item.isSelected()), None)
        self._shapes = list(shapes)
        self._redraw()
        if selected_shape is not None:
            new_index = next((i for i, s in enumerate(self._shapes) if s is selected_shape), None)
            if new_index is not None:
                self._items[new_index].setSelected(True)

    def shapes(self) -> list[Shape]:
        return list(self._shapes)

    # ---- selection ---------------------------------------------------------

    def _on_scene_selection_changed(self) -> None:
        if self._suspend_selection_signal:
            return
        selected = [i for i, item in enumerate(self._items) if item.isSelected()]
        self.shapeSelected.emit(selected[0] if selected else None)

    def select_shape(self, index: int | None) -> None:
        """Programmatically select the shape at `index` (or clear selection
        for None) without re-emitting `shapeSelected` — for syncing from an
        external widget (e.g. a results table row click) back onto the canvas.
        """
        self._suspend_selection_signal = True
        try:
            for i, item in enumerate(self._items):
                item.setSelected(i == index)
        finally:
            self._suspend_selection_signal = False

    # ---- region of interest ----------------------------------------------

    def set_roi_mode(self, enabled: bool) -> None:
        """When enabled, the next click-drag on the canvas defines the ROI."""
        self._roi_mode = enabled

    def roi(self) -> RoiRect | None:
        return self._roi

    def set_roi(self, rect: RoiRect | None) -> None:
        self._roi = rect
        self._draw_roi_item()
        self.roiChanged.emit(self._roi)

    def clear_roi(self) -> None:
        self.set_roi(None)

    def _draw_roi_item(self) -> None:
        if self._roi_item is not None:
            self.scene().removeItem(self._roi_item)
            self._roi_item = None
        if self._roi is None:
            return
        x, y, w, h = self._roi
        pen = QPen(QColor("#ffd400"), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._roi_item = QGraphicsRectItem(x, y, w, h)
        self._roi_item.setPen(pen)
        self._roi_item.setZValue(1000)
        self.scene().addItem(self._roi_item)

    # ---- rendering -------------------------------------------------------

    def _redraw(self) -> None:
        for item in self._items + self._labels:
            self.scene().removeItem(item)
        for handle in self._handles:
            if handle is not None:
                self.scene().removeItem(handle)
        self._items, self._labels, self._handles = [], [], []

        for shape in self._shapes:
            self._add_shape_visual(shape)

    def _add_shape_visual(self, shape: Shape) -> ShapePolygonItem:
        points = _shape_render_points(shape)
        color = color_for_class(shape.class_name, self._classes or [shape.class_name])
        item = ShapePolygonItem(points, color)
        self.scene().addItem(item)
        self._items.append(item)

        label_text = shape.class_name
        if shape.confidence is not None:
            label_text += f" {shape.confidence * 100:.0f}%"
        label = LabelItem(label_text, color)
        label.setPos(shape.x, shape.y - 20)
        self.scene().addItem(label)
        self._labels.append(label)

        handle = None
        if not self.read_only and shape.shape == "rotated_box":
            handle = RotateHandleItem(item, lambda pts, idx=len(self._items) - 1: self._on_rotate_commit(idx, pts))
            hx, hy = rotate_handle_position(points)
            handle.setPos(hx, hy)
            self.scene().addItem(handle)
        self._handles.append(handle)
        return item

    def _on_rotate_commit(self, index: int, points: list[tuple[float, float]]) -> None:
        shape = self._shapes[index]
        x, y, w, h = bbox_from_points(points)
        updated = shape.model_copy(update={"points": [[px, py] for px, py in points], "x": x, "y": y, "width": w, "height": h})
        self._shapes[index] = updated
        label = self._labels[index]
        label.setPos(x, y - 20)
        self.shapesChanged.emit(self.shapes())

    # ---- mouse interaction (drawing) -------------------------------------

    def mousePressEvent(self, event):  # noqa: N802
        if self._roi_mode:
            super().mousePressEvent(event)
            self._roi_drawing = True
            self._roi_start = self.mapToScene(event.pos())
            self._update_roi_draft(self._roi_start)
            return
        if self.read_only:
            super().mousePressEvent(event)
            return
        item = self.itemAt(event.pos())
        if isinstance(item, (ShapePolygonItem, RotateHandleItem)):
            self._drawing = False
            super().mousePressEvent(event)
            return
        super().mousePressEvent(event)
        scene_pos = self.mapToScene(event.pos())
        self._start_draw(scene_pos)

    def mouseMoveEvent(self, event):  # noqa: N802
        super().mouseMoveEvent(event)
        if self._roi_drawing:
            self._update_roi_draft(self.mapToScene(event.pos()))
            return
        if self._drawing:
            self._update_draft(self.mapToScene(event.pos()))

    def mouseReleaseEvent(self, event):  # noqa: N802
        super().mouseReleaseEvent(event)
        if self._roi_drawing:
            self._roi_drawing = False
            self._update_roi_draft(self.mapToScene(event.pos()))
            self._commit_roi()
            return
        if self._drawing:
            self._drawing = False
            # a release doesn't always follow a move at the same point (e.g. a
            # single press-drag-release with no intermediate move callback in
            # tests, or a very fast drag) — capture the final position here too.
            self._draw_current = self.mapToScene(event.pos())
            self._commit_draft()

    def _update_roi_draft(self, scene_pos: QPointF) -> None:
        x0, y0 = self._roi_start.x(), self._roi_start.y()
        x1, y1 = scene_pos.x(), scene_pos.y()
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self._roi = (x, y, w, h)
        self._draw_roi_item()

    def _commit_roi(self) -> None:
        if self._roi is not None and (self._roi[2] < MIN_DRAW_SIZE or self._roi[3] < MIN_DRAW_SIZE):
            self._roi = None
            self._draw_roi_item()
        self.roiChanged.emit(self._roi)

    def keyPressEvent(self, event):  # noqa: N802
        if not self.read_only and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            return
        super().keyPressEvent(event)

    def delete_selected(self) -> None:
        for i, item in enumerate(self._items):
            if item.isSelected():
                del self._shapes[i]
                self.set_shapes(self._shapes)
                self.shapesChanged.emit(self.shapes())
                return

    # ---- draw state machine ----------------------------------------------

    def _start_draw(self, scene_pos: QPointF) -> None:
        if not self._active_class:
            return
        self._drawing = True
        self._draw_start = scene_pos
        self._draw_current = scene_pos
        self._draw_points = [(scene_pos.x(), scene_pos.y())]
        pen = QPen(QColor("white"), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        if self._tool == "ellipse":
            self._draft_item = QGraphicsEllipseItem(scene_pos.x(), scene_pos.y(), 0, 0)
            self._draft_item.setPen(pen)
        elif self._tool == "polygon":
            self._draft_item = QGraphicsPolygonItem(QPolygonF([scene_pos]))
            self._draft_item.setPen(pen)
        else:  # box, rotated_box
            self._draft_item = QGraphicsPolygonItem(QPolygonF([scene_pos] * 4))
            self._draft_item.setPen(pen)
        self.scene().addItem(self._draft_item)

    def _update_draft(self, scene_pos: QPointF) -> None:
        self._draw_current = scene_pos
        if self._draft_item is None:
            return
        if self._tool == "polygon":
            last = self._draw_points[-1]
            if (scene_pos.x() - last[0]) ** 2 + (scene_pos.y() - last[1]) ** 2 >= MIN_PATH_POINT_DISTANCE**2:
                self._draw_points.append((scene_pos.x(), scene_pos.y()))
                self._draft_item.setPolygon(QPolygonF([QPointF(x, y) for x, y in self._draw_points]))
            return

        x0, y0 = self._draw_start.x(), self._draw_start.y()
        x1, y1 = scene_pos.x(), scene_pos.y()
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if self._tool == "ellipse":
            self._draft_item.setRect(x, y, w, h)
        else:
            corners = rect_corners(x, y, w, h)
            self._draft_item.setPolygon(QPolygonF([QPointF(px, py) for px, py in corners]))

    def _commit_draft(self) -> None:
        if self._draft_item is not None:
            self.scene().removeItem(self._draft_item)
            self._draft_item = None

        shape: Shape | None = None
        if self._tool == "polygon":
            if len(self._draw_points) >= 3:
                x, y, w, h = bbox_from_points(self._draw_points)
                shape = Shape(
                    class_name=self._active_class,
                    shape="polygon",
                    points=[[px, py] for px, py in self._draw_points],
                    x=x, y=y, width=w, height=h,
                )
        else:
            x0, y0 = self._draw_start.x(), self._draw_start.y()
            x1, y1 = self._draw_current.x(), self._draw_current.y()
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w >= MIN_DRAW_SIZE and h >= MIN_DRAW_SIZE:
                if self._tool == "ellipse":
                    points = ellipse_ring(x, y, w, h)
                elif self._tool == "rotated_box":
                    points = rect_corners(x, y, w, h)
                else:  # box
                    points = []
                shape = Shape(class_name=self._active_class, shape=self._tool, points=points, x=x, y=y, width=w, height=h)

        if shape is None:
            return
        self._shapes.append(shape)
        self._add_shape_visual(shape)
        self.shapesChanged.emit(self.shapes())
