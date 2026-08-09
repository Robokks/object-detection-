"""Shape geometry helpers (ported from the web app's BoundingBoxEditor.tsx)
and the QGraphicsItem classes used to draw/edit them on the canvas.

Scene coordinates are always natural image pixel coordinates — the
QGraphicsView applies zoom as a view-level transform, so none of this code
needs to know about display scale.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

BOX_COLORS = ["#e6483d", "#3d8ce6", "#3de68c", "#e6c73d", "#a13de6", "#e63dae"]
ELLIPSE_RING_POINTS = 24
ROTATE_HANDLE_OFFSET = 18.0
ROTATE_HANDLE_RADIUS = 6.0


def color_for_class(class_name: str, classes: list[str]) -> QColor:
    idx = classes.index(class_name) if class_name in classes else 0
    return QColor(BOX_COLORS[idx % len(BOX_COLORS)])


def ellipse_ring(x: float, y: float, width: float, height: float) -> list[tuple[float, float]]:
    cx, cy = x + width / 2, y + height / 2
    rx, ry = width / 2, height / 2
    points = []
    for i in range(ELLIPSE_RING_POINTS):
        t = (i / ELLIPSE_RING_POINTS) * 2 * math.pi
        points.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
    return points


def rect_corners(x: float, y: float, width: float, height: float) -> list[tuple[float, float]]:
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, y_min = min(xs), min(ys)
    return x_min, y_min, max(xs) - x_min, max(ys) - y_min


def center_of_points(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    return sum(p[0] for p in points) / n, sum(p[1] for p in points) / n


def rotate_points(
    points: list[tuple[float, float]], center: tuple[float, float], angle_rad: float
) -> list[tuple[float, float]]:
    cx, cy = center
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    result = []
    for px, py in points:
        dx, dy = px - cx, py - cy
        result.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return result


def angle_at(center: tuple[float, float], x: float, y: float) -> float:
    return math.atan2(y - center[1], x - center[0])


def rotate_handle_position(points: list[tuple[float, float]]) -> tuple[float, float]:
    center = center_of_points(points)
    top_mid = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
    dx, dy = top_mid[0] - center[0], top_mid[1] - center[1]
    length = math.hypot(dx, dy) or 1.0
    return (
        top_mid[0] + (dx / length) * ROTATE_HANDLE_OFFSET,
        top_mid[1] + (dy / length) * ROTATE_HANDLE_OFFSET,
    )


class ShapePolygonItem(QGraphicsPolygonItem):
    """Renders any shape kind (box/rotated_box/ellipse/polygon) as a polygon —
    a box is 4 points, an ellipse is a 24-point ring, etc. Uniform rendering
    keeps selection/hit-testing/deletion logic identical across shape kinds.
    """

    def __init__(self, points: list[tuple[float, float]], color: QColor):
        super().__init__(QPolygonF([QPointF(x, y) for x, y in points]))
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(255, 255, 255, 8)))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.shape_index: int = -1

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self.setPolygon(QPolygonF([QPointF(x, y) for x, y in points]))

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            painter.save()
            pen = QPen(QColor("#ffffff"), 3, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPolygon(self.polygon())
            painter.restore()
        super().paint(painter, option, widget)


class LabelItem(QGraphicsItem):
    """A small colored pill with the class name, drawn above a shape."""

    def __init__(self, text: str, color: QColor):
        super().__init__()
        self._bg = QGraphicsRectItem(self)
        self._bg.setBrush(QBrush(color))
        self._bg.setPen(QPen(Qt.PenStyle.NoPen))
        self._text = QGraphicsSimpleTextItem(text, self)
        self._text.setBrush(QBrush(QColor("white")))
        self._text.setPos(4, 1)
        text_rect = self._text.boundingRect()
        self._bg.setRect(0, 0, text_rect.width() + 8, text_rect.height() + 2)

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return self._bg.boundingRect()

    def paint(self, painter, option, widget=None):  # noqa: N802 (Qt override)
        pass


class RotateHandleItem(QGraphicsEllipseItem):
    """Drag this to rotate its associated ShapePolygonItem live around its
    current center. Reports the final rotation back via `on_commit`.
    """

    def __init__(self, shape_item: ShapePolygonItem, on_commit):
        r = ROTATE_HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.shape_item = shape_item
        self._on_commit = on_commit
        self.setBrush(QBrush(QColor("#aa3bff")))
        self.setPen(QPen(QColor("white"), 2))
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self._dragging = False
        self._center = (0.0, 0.0)
        self._start_angle = 0.0
        self._original_points: list[tuple[float, float]] = []

    def mousePressEvent(self, event):  # noqa: N802
        pts = [(p.x(), p.y()) for p in self.shape_item.polygon()]
        self._original_points = pts
        self._center = center_of_points(pts)
        self._start_angle = angle_at(self._center, event.scenePos().x(), event.scenePos().y())
        self._dragging = True
        event.accept()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self._dragging:
            return
        current_angle = angle_at(self._center, event.scenePos().x(), event.scenePos().y())
        delta = current_angle - self._start_angle
        rotated = rotate_points(self._original_points, self._center, delta)
        self.shape_item.set_points(rotated)
        handle_x, handle_y = rotate_handle_position(rotated)
        self.setPos(handle_x, handle_y)
        event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._dragging = False
        final_points = [(p.x(), p.y()) for p in self.shape_item.polygon()]
        self._on_commit(final_points)
        event.accept()
