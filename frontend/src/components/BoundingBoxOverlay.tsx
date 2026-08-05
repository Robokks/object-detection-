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
  boxes: BoundingBox[];
}

export default function BoundingBoxOverlay({ imageSrc, naturalWidth, naturalHeight, boxes }: Props) {
  const scale = Math.min(1, DISPLAY_MAX_WIDTH / naturalWidth);
  const displayWidth = naturalWidth * scale;
  const displayHeight = naturalHeight * scale;
  const classes = Array.from(new Set(boxes.map((b) => b.class_name)));

  return (
    <div
      className="bbox-canvas"
      style={{ width: displayWidth, height: displayHeight, backgroundImage: `url(${imageSrc})`, cursor: "default" }}
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
            {box.confidence != null ? ` ${(box.confidence * 100).toFixed(0)}%` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}
