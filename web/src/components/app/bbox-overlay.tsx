"use client";
import { useState } from "react";
import type { Detection } from "@/lib/types";
import { CLASS_COLORS } from "./common";

/** Interactive detection boxes drawn over the original image (normalised coords from the API). */
export function BBoxOverlay({ src, detections, selected, onSelect }: {
  src: string; detections: Detection[]; selected?: string | null; onSelect?: (id: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div className="relative w-full overflow-hidden rounded-lg border bg-muted">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="analysed" className="block w-full" />
      {detections.map((d) => {
        const c = CLASS_COLORS[d.class_name] || "#0ea5e9";
        const on = hover === d.id || selected === d.id;
        const big = ["corridor", "walkway"].includes(d.class_name);
        return (
          <button
            key={d.id}
            onMouseEnter={() => setHover(d.id)}
            onMouseLeave={() => setHover(null)}
            onClick={() => onSelect?.(d.id)}
            className="absolute rounded-sm transition-all"
            style={{
              left: `${d.bbox.x1n * 100}%`, top: `${d.bbox.y1n * 100}%`,
              width: `${(d.bbox.x2n - d.bbox.x1n) * 100}%`, height: `${(d.bbox.y2n - d.bbox.y1n) * 100}%`,
              border: `${on ? 3 : big ? 1 : 2}px ${big ? "dashed" : "solid"} ${c}`,
              background: on ? `${c}22` : "transparent", zIndex: on ? 20 : big ? 1 : 10,
            }}
            title={`${d.class_name} ${(d.confidence * 100).toFixed(0)}%`}
          >
            {(on || !big) && (
              <span className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 text-[10px] font-semibold text-white" style={{ background: c }}>
                {d.class_name} {(d.confidence * 100).toFixed(0)}%
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
