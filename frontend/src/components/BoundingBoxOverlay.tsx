import type { Shape } from "../api/types";
import { colorForClass } from "./BoundingBoxEditor";
import "./BoundingBoxEditor.css";

const DISPLAY_MAX_WIDTH = 760;

function toSvgPoints(points: number[][], scale: number): string {
  return points.map(([px, py]) => `${px * scale},${py * scale}`).join(" ");
}

interface Props {
  imageSrc: string;
  naturalWidth: number;
  naturalHeight: number;
  boxes: Shape[];
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
      <svg className="bbox-svg" width={displayWidth} height={displayHeight}>
        {boxes.map((shape, idx) => {
          const color = colorForClass(shape.class_name, classes);
          if (shape.points.length >= 3) {
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
        {boxes.map((shape, idx) => {
          const cx = shape.center_x ?? shape.x + shape.width / 2;
          const cy = shape.center_y ?? shape.y + shape.height / 2;
          const color = colorForClass(shape.class_name, classes);
          return (
            <circle key={idx} cx={cx * scale} cy={cy * scale} r={4} className="bbox-center-dot" fill={color} />
          );
        })}
      </svg>

      {boxes.map((shape, idx) => (
        <div key={idx} className="bbox-controls" style={{ left: shape.x * scale, top: shape.y * scale - 20 }}>
          <span className="bbox-label" style={{ background: colorForClass(shape.class_name, classes) }}>
            {shape.class_name}
            {shape.confidence != null ? ` ${(shape.confidence * 100).toFixed(0)}%` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}
