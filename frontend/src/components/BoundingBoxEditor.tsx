import { useRef, useState } from "react";
import type { BoundingBox } from "../api/types";
import "./BoundingBoxEditor.css";

const DISPLAY_MAX_WIDTH = 760;
const BOX_COLORS = ["#e6483d", "#3d8ce6", "#3de68c", "#e6c73d", "#a13de6", "#e63dae"];

function colorForClass(className: string, classes: string[]): string {
  const idx = classes.indexOf(className);
  return BOX_COLORS[(idx < 0 ? 0 : idx) % BOX_COLORS.length];
}

interface Props {
  imageSrc: string;
  naturalWidth: number;
  naturalHeight: number;
  classes: string[];
  activeClass: string;
  boxes: BoundingBox[];
  onBoxesChange: (boxes: BoundingBox[]) => void;
}

export default function BoundingBoxEditor({
  imageSrc,
  naturalWidth,
  naturalHeight,
  classes,
  activeClass,
  boxes,
  onBoxesChange,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<{ startX: number; startY: number; x: number; y: number } | null>(
    null
  );

  const scale = Math.min(1, DISPLAY_MAX_WIDTH / naturalWidth);
  const displayWidth = naturalWidth * scale;
  const displayHeight = naturalHeight * scale;

  function toNatural(clientX: number, clientY: number) {
    const rect = containerRef.current!.getBoundingClientRect();
    const x = Math.min(Math.max(clientX - rect.left, 0), rect.width) / scale;
    const y = Math.min(Math.max(clientY - rect.top, 0), rect.height) / scale;
    return { x, y };
  }

  function handleMouseDown(e: React.MouseEvent) {
    const { x, y } = toNatural(e.clientX, e.clientY);
    setDraft({ startX: x, startY: y, x, y });
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!draft) return;
    const { x, y } = toNatural(e.clientX, e.clientY);
    setDraft({ ...draft, x, y });
  }

  function handleMouseUp() {
    if (!draft) return;
    const x = Math.min(draft.startX, draft.x);
    const y = Math.min(draft.startY, draft.y);
    const width = Math.abs(draft.x - draft.startX);
    const height = Math.abs(draft.y - draft.startY);
    setDraft(null);
    if (width < 4 || height < 4) return;
    onBoxesChange([...boxes, { class_name: activeClass, x, y, width, height }]);
  }

  function removeBox(idx: number) {
    onBoxesChange(boxes.filter((_, i) => i !== idx));
  }

  const draftRect = draft
    ? {
        left: Math.min(draft.startX, draft.x) * scale,
        top: Math.min(draft.startY, draft.y) * scale,
        width: Math.abs(draft.x - draft.startX) * scale,
        height: Math.abs(draft.y - draft.startY) * scale,
      }
    : null;

  return (
    <div className="bbox-editor">
      <div
        ref={containerRef}
        className="bbox-canvas"
        style={{ width: displayWidth, height: displayHeight, backgroundImage: `url(${imageSrc})` }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => setDraft(null)}
      >
        {boxes.map((box, idx) => (
          <div
            key={idx}
            className="bbox-rect"
            style={{
              left: box.x * scale,
              top: box.y * scale,
              width: box.width * scale,
              height: box.height * scale,
              borderColor: colorForClass(box.class_name, classes),
            }}
          >
            <span className="bbox-label" style={{ background: colorForClass(box.class_name, classes) }}>
              {box.class_name}
            </span>
            <button className="bbox-remove" onClick={() => removeBox(idx)} title="Remove box">
              ×
            </button>
          </div>
        ))}
        {draftRect && <div className="bbox-rect bbox-draft" style={draftRect} />}
      </div>
      <p className="bbox-hint">
        Drag on the image to draw a box for class <strong>{activeClass || "(select a class)"}</strong>.
      </p>
    </div>
  );
}
