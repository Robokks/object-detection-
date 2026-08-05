"""Classical-CV candidate finder for elongated objects under flash lighting.

An elongated cylindrical object (rod, pin, roller) photographed with a
camera flash typically shows a bright specular highlight running along its
length, flanked by darker cylinder surface on both sides — a
dark -> glare -> dark cross-section. This scans an image for that pattern
and proposes rotated-box candidates. It's a heuristic, not a classifier: it
finds *candidates* for a human to review, accept, or discard — not every
suggestion will be a true positive, and closely touching/overlapping
objects tend to merge into a single coarse candidate.
"""

import numpy as np
from PIL import Image

from app.schemas import Shape

BRIGHT_PERCENTILE = 92
BRIGHT_FLOOR = 150.0
MIN_AREA = 12
MIN_LENGTH = 30.0
MIN_ELONGATION = 2.8
DARK_MARGIN = 10.0
DARK_DROP_RATIO = 0.75
CLOSE_KERNEL = 5
CLOSE_ITERATIONS = 2
BRIDGE_KERNEL = 17
BRIDGE_SHRINK_LENGTH = 17.0
BRIDGE_SHRINK_THICKNESS = 10.0


def _sample_line(gray: np.ndarray, p0: tuple, p1: tuple, n: int = 15) -> np.ndarray:
    xs = np.clip(np.linspace(p0[0], p1[0], n), 0, gray.shape[1] - 1)
    ys = np.clip(np.linspace(p0[1], p1[1], n), 0, gray.shape[0] - 1)
    return gray[ys.astype(int), xs.astype(int)]


def suggest_cylinder_shapes(image_path, class_name: str) -> list[Shape]:
    import cv2  # heavy import, deferred like the ultralytics imports elsewhere

    gray = np.array(Image.open(image_path).convert("L")).astype(np.float32)

    thresh_val = max(float(np.percentile(gray, BRIGHT_PERCENTILE)), BRIGHT_FLOOR)
    bright = (gray >= thresh_val).astype(np.uint8) * 255

    # Pass 1: light closing only, so we validate individual streak fragments
    # without bridging into neighboring (unrelated) bright mesh/background texture.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel_small, iterations=CLOSE_ITERATIONS)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    validated_mask = np.zeros_like(bright)
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
        length = max(rw, rh)
        thickness = max(min(rw, rh), 1.0)
        if length < MIN_LENGTH or length / thickness < MIN_ELONGATION:
            continue

        theta = np.deg2rad(angle if rw >= rh else angle + 90)
        ux, uy = np.cos(theta), np.sin(theta)
        px, py = -uy, ux
        streak_b = float(
            _sample_line(
                gray, (cx - ux * length / 3, cy - uy * length / 3), (cx + ux * length / 3, cy + uy * length / 3)
            ).mean()
        )
        flank_offset = thickness / 2 + DARK_MARGIN
        flank_a = _sample_line(
            gray,
            (cx + px * flank_offset - ux * length / 3, cy + py * flank_offset - uy * length / 3),
            (cx + px * flank_offset + ux * length / 3, cy + py * flank_offset + uy * length / 3),
        )
        flank_b = _sample_line(
            gray,
            (cx - px * flank_offset - ux * length / 3, cy - py * flank_offset - uy * length / 3),
            (cx - px * flank_offset + ux * length / 3, cy - py * flank_offset + uy * length / 3),
        )
        flank_brightness = float(np.mean(np.concatenate([flank_a, flank_b])))
        if flank_brightness >= streak_b * DARK_DROP_RATIO:
            continue  # not flanked by darker surface on both sides — likely mesh/background texture

        cv2.drawContours(validated_mask, [c], -1, 255, -1)

    # Pass 2: bridge validated fragments of the same object (a highlight is
    # often broken by a thin dark centerline where curvature kills the
    # reflection) now that unrelated texture noise has already been dropped.
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BRIDGE_KERNEL, BRIDGE_KERNEL))
    merged_mask = cv2.dilate(validated_mask, kernel_big, iterations=1)
    merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes: list[Shape] = []
    for c in merged_contours:
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
        # undo the dilation's fattening so the box hugs the actual streak again
        length = max(max(rw, rh) - BRIDGE_SHRINK_LENGTH, MIN_LENGTH)
        thickness = max(min(rw, rh) - BRIDGE_SHRINK_THICKNESS, 4.0)
        width, height = (length, thickness) if rw >= rh else (thickness, length)

        points = [[float(px), float(py)] for px, py in cv2.boxPoints(((cx, cy), (width, height), angle))]
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
    return shapes
