"""Import images that already have annotations hand-drawn on them as solid
red outlines (a common way to mark objects without a separate labeling tool).

Strategy: threshold for the (very consistent) red used to draw outlines,
then take each outline's *interior hole* rather than its outer blob — two
outlines that touch or overlap at an edge still have separate interiors,
so this correctly splits touching boxes that a naive connected-component
pass would merge. Each hole is fit with an oriented rectangle (works for
both clean and hand-drawn/wobbly outlines). The red pixels are then
inpainted out so the stored training image doesn't carry a red-line
artifact the model could latch onto instead of the real object.
"""

import io

import numpy as np
from PIL import Image

from app.schemas import Shape

RED_MIN_CHANNEL = 120
RED_DOMINANCE = 40
MIN_AREA_FRAC = 0.0003  # relative to image area, so this scales with resolution
STROKE_MARGIN = 3  # px added back since the hole is inset from the true outline


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    mask = (r > RED_MIN_CHANNEL) & (r - g > RED_DOMINANCE) & (r - b > RED_DOMINANCE)
    return (mask.astype(np.uint8)) * 255


def extract_annotated_image(image_bytes: bytes, class_name: str) -> tuple[bytes, list[Shape]]:
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    height, width = bgr.shape[:2]

    red_mask = _red_mask(bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    min_area = MIN_AREA_FRAC * width * height

    shapes: list[Shape] = []
    for i, c in enumerate(contours):
        is_hole = len(hierarchy) > 0 and hierarchy[i][3] != -1
        if not is_hole:
            continue
        if cv2.contourArea(c) < min_area:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
        rw, rh = rw + STROKE_MARGIN * 2, rh + STROKE_MARGIN * 2
        points = [[float(px), float(py)] for px, py in cv2.boxPoints(((cx, cy), (rw, rh), angle))]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        shapes.append(
            Shape(
                class_name=class_name,
                shape="rotated_box",
                points=points,
                x=min(xs),
                y=min(ys),
                width=max(xs) - min(xs),
                height=max(ys) - min(ys),
            )
        )

    # Inpaint the red outlines out so the stored training image doesn't carry
    # a red-line artifact correlated with object locations.
    inpaint_mask = cv2.dilate(red_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    clean_bgr = cv2.inpaint(bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    clean_rgb = clean_bgr[:, :, ::-1]
    out = io.BytesIO()
    Image.fromarray(clean_rgb).save(out, format="PNG")
    return out.getvalue(), shapes
