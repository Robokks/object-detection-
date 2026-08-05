import { useRef, useState } from "react";
import type { Shape, ShapeKind } from "../api/types";
import "./BoundingBoxEditor.css";

const DISPLAY_MAX_WIDTH = 760;
const ELLIPSE_RING_POINTS = 24;
const MIN_PATH_POINT_DISTANCE = 2;
export const BOX_COLORS = ["#e6483d", "#3d8ce6", "#3de68c", "#e6c73d", "#a13de6", "#e63dae"];

export function colorForClass(className: string, classes: string[]): string {
  const idx = classes.indexOf(className);
  return BOX_COLORS[(idx < 0 ? 0 : idx) % BOX_COLORS.length];
}

function ellipseRing(x: number, y: number, width: number, height: number): number[][] {
  const cx = x + width / 2;
  const cy = y + height / 2;
  const rx = width / 2;
  const ry = height / 2;
  const points: number[][] = [];
  for (let i = 0; i < ELLIPSE_RING_POINTS; i++) {
    const t = (i / ELLIPSE_RING_POINTS) * Math.PI * 2;
    points.push([cx + rx * Math.cos(t), cy + ry * Math.sin(t)]);
  }
  return points;
}

function bboxFromPoints(points: number[][]) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const xMin = Math.min(...xs);
  const yMin = Math.min(...ys);
  return { x: xMin, y: yMin, width: Math.max(...xs) - xMin, height: Math.max(...ys) - yMin };
}

function toSvgPoints(points: number[][], scale: number): string {
  return points.map(([px, py]) => `${px * scale},${py * scale}`).join(" ");
}

type Draft =
  | { kind: "rect"; startX: number; startY: number; x: number; y: number }
  | { kind: "path"; points: number[][] };

const TOOLS: { id: ShapeKind; label: string; hint: string }[] = [
  { id: "box", label: "Box", hint: "Drag to draw a rectangle." },
  { id: "ellipse", label: "Ellipse", hint: "Drag to draw an ellipse." },
  { id: "polygon", label: "Pen", hint: "Drag to freehand-draw an outline." },
];

interface Props {
  imageSrc: string;
  naturalWidth: number;
  naturalHeight: number;
  classes: string[];
  activeClass: string;
  shapes: Shape[];
  onShapesChange: (shapes: Shape[]) => void;
}

export default function BoundingBoxEditor({
  imageSrc,
  naturalWidth,
  naturalHeight,
  classes,
  activeClass,
  shapes,
  onShapesChange,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tool, setTool] = useState<ShapeKind>("box");
  const [draft, setDraft] = useState<Draft | null>(null);

  const scale = Math.min(1, DISPLAY_MAX_WIDTH / naturalWidth);
  const displayWidth = naturalWidth * scale;
  const displayHeight = naturalHeight * scale;

  function toNatural(clientX: number, clientY: number): [number, number] {
    const rect = containerRef.current!.getBoundingClientRect();
    const x = Math.min(Math.max(clientX - rect.left, 0), rect.width) / scale;
    const y = Math.min(Math.max(clientY - rect.top, 0), rect.height) / scale;
    return [x, y];
  }

  function handleMouseDown(e: React.MouseEvent) {
    const [x, y] = toNatural(e.clientX, e.clientY);
    if (tool === "polygon") {
      setDraft({ kind: "path", points: [[x, y]] });
    } else {
      setDraft({ kind: "rect", startX: x, startY: y, x, y });
    }
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!draft) return;
    const [x, y] = toNatural(e.clientX, e.clientY);
    if (draft.kind === "rect") {
      setDraft({ ...draft, x, y });
    } else {
      const last = draft.points[draft.points.length - 1];
      if (Math.hypot(x - last[0], y - last[1]) >= MIN_PATH_POINT_DISTANCE) {
        setDraft({ kind: "path", points: [...draft.points, [x, y]] });
      }
    }
  }

  function handleMouseUp() {
    if (!draft) return;
    if (draft.kind === "rect") {
      const x = Math.min(draft.startX, draft.x);
      const y = Math.min(draft.startY, draft.y);
      const width = Math.abs(draft.x - draft.startX);
      const height = Math.abs(draft.y - draft.startY);
      setDraft(null);
      if (width < 4 || height < 4) return;
      const points = tool === "ellipse" ? ellipseRing(x, y, width, height) : [];
      onShapesChange([...shapes, { class_name: activeClass, shape: tool, points, x, y, width, height }]);
    } else {
      const points = draft.points;
      setDraft(null);
      if (points.length < 3) return;
      const bbox = bboxFromPoints(points);
      onShapesChange([...shapes, { class_name: activeClass, shape: "polygon", points, ...bbox }]);
    }
  }

  function removeShape(idx: number) {
    onShapesChange(shapes.filter((_, i) => i !== idx));
  }

  const activeTool = TOOLS.find((t) => t.id === tool)!;

  return (
    <div className="bbox-editor">
      <div className="tool-row">
        {TOOLS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={"tool-btn" + (t.id === tool ? " tool-btn-active" : "")}
            onClick={() => setTool(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div
        ref={containerRef}
        className="bbox-canvas"
        style={{ width: displayWidth, height: displayHeight, backgroundImage: `url(${imageSrc})` }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => setDraft(null)}
      >
        <svg className="bbox-svg" width={displayWidth} height={displayHeight}>
          {shapes.map((shape, idx) => {
            const color = colorForClass(shape.class_name, classes);
            if (shape.shape === "ellipse") {
              return (
                <ellipse
                  key={idx}
                  cx={(shape.x + shape.width / 2) * scale}
                  cy={(shape.y + shape.height / 2) * scale}
                  rx={(shape.width / 2) * scale}
                  ry={(shape.height / 2) * scale}
                  className="bbox-shape"
                  stroke={color}
                />
              );
            }
            if (shape.shape === "polygon") {
              return <polygon key={idx} points={toSvgPoints(shape.points, scale)} className="bbox-shape" stroke={color} />;
            }
            return (
              <rect
                key={idx}
                x={shape.x * scale}
                y={shape.y * scale}
                width={shape.width * scale}
                height={shape.height * scale}
                className="bbox-shape"
                stroke={color}
              />
            );
          })}

          {draft?.kind === "rect" &&
            (() => {
              const x = Math.min(draft.startX, draft.x);
              const y = Math.min(draft.startY, draft.y);
              const width = Math.abs(draft.x - draft.startX);
              const height = Math.abs(draft.y - draft.startY);
              return tool === "ellipse" ? (
                <ellipse
                  cx={(x + width / 2) * scale}
                  cy={(y + height / 2) * scale}
                  rx={(width / 2) * scale}
                  ry={(height / 2) * scale}
                  className="bbox-shape bbox-draft"
                />
              ) : (
                <rect x={x * scale} y={y * scale} width={width * scale} height={height * scale} className="bbox-shape bbox-draft" />
              );
            })()}
          {draft?.kind === "path" && <polyline points={toSvgPoints(draft.points, scale)} className="bbox-shape bbox-draft" fill="none" />}
        </svg>

        {shapes.map((shape, idx) => (
          <div key={idx} className="bbox-controls" style={{ left: shape.x * scale, top: shape.y * scale - 20 }}>
            <span className="bbox-label" style={{ background: colorForClass(shape.class_name, classes) }}>
              {shape.class_name}
            </span>
            <button className="bbox-remove" onClick={() => removeShape(idx)} title="Remove shape">
              ×
            </button>
          </div>
        ))}
      </div>
      <p className="bbox-hint">
        {activeTool.hint} New shapes are labeled <strong>{activeClass || "(select a class)"}</strong>.
      </p>
    </div>
  );
}
