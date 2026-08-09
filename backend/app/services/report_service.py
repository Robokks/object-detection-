"""Turns a Shape into the same set of fields the Detect tab's results table
shows (and the standalone `scripts/detect_to_json.py` writes to JSON), so
both stay in sync by construction instead of duplicating this math.
"""

from app.schemas import Shape


def shape_to_report(shape: Shape) -> dict:
    cx, cy = shape.center_x, shape.center_y
    has_outline = len(shape.points) >= 3
    return {
        "class_name": shape.class_name,
        "shape": "mask" if has_outline else "box",
        "confidence": shape.confidence,
        "position": {"x": round(cx, 1), "y": round(cy, 1)},
        "angle_degrees": round(shape.angle, 1) if has_outline else None,
        "left": {"x": round(shape.x, 1), "y": round(cy, 1)},
        "right": {"x": round(shape.x + shape.width, 1), "y": round(cy, 1)},
        "top": {"x": round(cx, 1), "y": round(shape.y, 1)},
        "bottom": {"x": round(cx, 1), "y": round(shape.y + shape.height, 1)},
        "box": {
            "x": round(shape.x, 1),
            "y": round(shape.y, 1),
            "width": round(shape.width, 1),
            "height": round(shape.height, 1),
        },
    }
