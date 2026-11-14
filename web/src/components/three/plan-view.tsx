"use client";
// Top-down floor plan of a reconstruction (local frame, +Y up => plan is the x/z plane).
// Base layer: plan generated from the sparse points + splat centres. Optional overlay: the user's own
// architectural plan, placed with a similarity transform (drag to move, sliders for scale/rotation/opacity).
// Vector layers: walked path (as a corridor), facilities, tagged places, route, live camera position.
import { useMemo, useRef, useState } from "react";
import { apiUrl } from "@/lib/api";
import type { PlanInfo, PlanTransform } from "@/lib/places";
import { PLACE_COLORS, type CameraPose, type PlacePin } from "@/components/three/scene-viewer";
import { CLASS_COLORS } from "@/components/app/common";

export type PlanIssue = { id: string; severity: "high" | "medium" | "low"; title: string; position: number[] };
const SEV_COLOR = { high: "#e11d48", medium: "#f59e0b", low: "#64748b" };

export default function PlanView({ plan, path, places = [], facilities = [], route, camera, selectedPlace, onPlaceClick, onPick,
  placing, align, onTransform, showUploaded = true, showGenerated = true, className = "", corridor, issues = [], selectedIssue, onIssueClick }: {
  plan: PlanInfo;
  path: number[][];
  places?: PlacePin[];
  facilities?: { id: string; class_name: string; pos: number[] }[];
  route?: number[][] | null;
  camera?: CameraPose | null;
  selectedPlace?: string | null;
  onPlaceClick?: (id: string) => void;
  onPick?: (pos: number[]) => void;          // click on the plan (placing mode) -> local [x, y, z]
  placing?: boolean;
  align?: boolean;                           // drag moves the uploaded plan
  onTransform?: (t: PlanTransform) => void;
  showUploaded?: boolean;
  showGenerated?: boolean;
  className?: string;
  corridor?: { left: number[][]; right: number[][] } | null;   // measured walls, [x, z] in the local frame
  issues?: PlanIssue[];
  selectedIssue?: string | null;
  onIssueClick?: (id: string) => void;
}) {
  const g = plan.generated!;
  const W = g.width, H = g.height, ppu = g.px_per_unit, [ox, oz] = g.origin;
  const P = (p: number[]) => [(p[0] - ox) * ppu, (p[2] - oz) * ppu] as const;
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; t: PlanTransform } | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const pathD = useMemo(() => path.map((p, i) => `${i ? "L" : "M"}${P(p).join(",")}`).join(" "),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [path, ox, oz, ppu]);
  const Q = (p: number[]) => [(p[0] - ox) * ppu, (p[1] - oz) * ppu] as const;       // [x, z] pairs
  const walls = useMemo(() => {
    if (!corridor || corridor.left.length < 2) return null;
    const line = (a: number[][]) => a.map((p, i) => `${i ? "L" : "M"}${Q(p).join(",")}`).join(" ");
    const fill = `${line(corridor.left)} ${[...corridor.right].reverse().map((p) => `L${Q(p).join(",")}`).join(" ")} Z`;
    return { fill, left: line(corridor.left), right: line(corridor.right) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [corridor, ox, oz, ppu]);
  const routeD = route && route.length > 1 ? route.map((p, i) => `${i ? "L" : "M"}${P(p).join(",")}`).join(" ") : null;
  const u = plan.uploaded;
  const t = u?.transform;
  const unit = Math.max(W, H) / 100;                    // UI size unit in plan pixels
  const corridorW = g.units === "m" ? Math.max(unit * 1.8, 1.6 * ppu) : unit * 2.4;   // ~1.6 m wide when metric

  function toPlan(e: React.PointerEvent | React.MouseEvent) {
    const s = svg.current!;
    const pt = s.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const m = s.getScreenCTM();
    if (!m) return null;
    const q = pt.matrixTransform(m.inverse());
    return [q.x, q.y] as [number, number];
  }
  function nearestY(x: number, z: number) {
    let best = Infinity, y = path[0]?.[1] ?? 0;
    for (const p of path) {
      const d = (p[0] - x) ** 2 + (p[2] - z) ** 2;
      if (d < best) { best = d; y = p[1]; }
    }
    return y;
  }
  return (
    <svg ref={svg} viewBox={`0 0 ${W} ${H}`} className={`h-full w-full select-none ${placing ? "cursor-crosshair" : align ? "cursor-move" : ""} ${className}`}
      preserveAspectRatio="xMidYMid meet"
      onPointerDown={(e) => {
        if (align && t) { const q = toPlan(e); if (q) { drag.current = { x: q[0], y: q[1], t: { ...t } }; (e.target as Element).setPointerCapture?.(e.pointerId); } }
      }}
      onPointerMove={(e) => {
        const q = toPlan(e);
        if (placing) setHover(q);
        if (drag.current && q && onTransform) onTransform({ ...drag.current.t, x: drag.current.t.x + q[0] - drag.current.x, y: drag.current.t.y + q[1] - drag.current.y });
      }}
      onPointerUp={() => { drag.current = null; }}
      onPointerLeave={() => setHover(null)}
      onClick={(e) => {
        if (!placing || !onPick) return;
        const q = toPlan(e);
        if (!q) return;
        const x = q[0] / ppu + ox, z = q[1] / ppu + oz;
        onPick([x, nearestY(x, z), z]);
      }}>
      <rect width={W} height={H} fill="#f8fafc" />
      {showGenerated && <image href={apiUrl(g.url)} width={W} height={H} opacity={u && showUploaded ? 0.55 : 1} />}
      {u && t && showUploaded && (
        <g transform={`translate(${t.x} ${t.y}) rotate(${t.rotation} ${(u.width * t.scale) / 2} ${(u.height * t.scale) / 2}) scale(${t.scale})`} opacity={t.opacity}>
          <image href={apiUrl(u.url)} width={u.width} height={u.height} />
          {align && <rect width={u.width} height={u.height} fill="none" stroke="#7c3aed" strokeWidth={3 / t.scale} strokeDasharray={`${12 / t.scale} ${8 / t.scale}`} />}
        </g>
      )}
      {/* measured corridor: floor + walls (falls back to a band along the walked path) */}
      {walls ? (
        <g>
          <path d={walls.fill} fill="#ffffff" fillOpacity={0.92} stroke="none" />
          <path d={walls.fill} fill="#ccfbf1" fillOpacity={0.35} stroke="none" />
          <path d={walls.left} fill="none" stroke="#0f172a" strokeWidth={unit * 0.45} strokeLinejoin="round" strokeLinecap="round" />
          <path d={walls.right} fill="none" stroke="#0f172a" strokeWidth={unit * 0.45} strokeLinejoin="round" strokeLinecap="round" />
        </g>
      ) : (
        <path d={pathD} fill="none" stroke="#99f6e4" strokeOpacity={0.55} strokeWidth={corridorW} strokeLinecap="round" strokeLinejoin="round" />
      )}
      <path d={pathD} fill="none" stroke="#0f766e" strokeWidth={unit * 0.25} strokeDasharray={`${unit * 0.8} ${unit * 0.6}`} strokeLinecap="round" />
      {facilities.map((f) => {
        const [x, y] = P(f.pos);
        return <circle key={f.id} cx={x} cy={y} r={unit * 0.55} fill={CLASS_COLORS[f.class_name] || "#64748b"} stroke="#fff" strokeWidth={unit * 0.15}><title>{f.class_name}</title></circle>;
      })}
      {routeD && (
        <>
          <path d={routeD} fill="none" stroke="#fff" strokeWidth={unit * 1.5} strokeLinecap="round" strokeLinejoin="round" />
          <path d={routeD} fill="none" stroke="#f43f5e" strokeWidth={unit * 0.9} strokeLinecap="round" strokeLinejoin="round" />
          <path d={routeD} fill="none" stroke="#fff" strokeWidth={unit * 0.3} strokeDasharray={`${unit} ${unit * 1.2}`} strokeLinecap="round">
            <animate attributeName="stroke-dashoffset" from={unit * 4.4} to="0" dur="1s" repeatCount="indefinite" />
          </path>
        </>
      )}
      {path[0] && (() => { const [x, y] = P(path[0]); return <g><circle cx={x} cy={y} r={unit * 0.9} fill="#16a34a" stroke="#fff" strokeWidth={unit * 0.25} /><text x={x + unit * 1.4} y={y + unit * 0.4} fontSize={unit * 1.3} fill="#166534" fontWeight={600}>Start</text></g>; })()}
      {places.map((p) => {
        const [x, y] = P(p.pos);
        const col = p.color || PLACE_COLORS[p.kind] || "#475569";
        const sel = selectedPlace === p.id;
        const fs = unit * (sel ? 1.6 : 1.35);
        const w = p.name.length * fs * 0.58 + unit * 1.6;
        return (
          <g key={p.id} transform={`translate(${x} ${y})`} className="cursor-pointer" onClick={(e) => { if (placing) return; e.stopPropagation(); onPlaceClick?.(p.id); }}>
            {sel && <circle r={unit * 2.4} fill={col} opacity={0.18}><animate attributeName="r" values={`${unit * 1.6};${unit * 2.8};${unit * 1.6}`} dur="1.8s" repeatCount="indefinite" /></circle>}
            <path d={`M0 0 C ${-unit * 1.1} ${-unit * 1.4}, ${-unit * 1.1} ${-unit * 2.6}, 0 ${-unit * 2.6} C ${unit * 1.1} ${-unit * 2.6}, ${unit * 1.1} ${-unit * 1.4}, 0 0 Z`} fill={col} stroke="#fff" strokeWidth={unit * 0.2} />
            <circle cy={-unit * 1.8} r={unit * 0.4} fill="#fff" />
            <g transform={`translate(${unit * 1.2} ${-unit * 3.6})`}>
              <rect width={w} height={fs * 1.5} rx={fs * 0.75} fill={col} />
              <text x={unit * 0.8} y={fs * 1.05} fontSize={fs} fill="#fff" fontWeight={600}>{p.name}</text>
            </g>
          </g>
        );
      })}
      {issues.map((i) => {
        const [x, y] = P(i.position);
        const sel = selectedIssue === i.id, r = unit * (sel ? 1.7 : 1.3), col = SEV_COLOR[i.severity];
        return (
          <g key={i.id} transform={`translate(${x} ${y})`} className="cursor-pointer" onClick={(e) => { if (placing) return; e.stopPropagation(); onIssueClick?.(i.id); }}>
            <circle r={r * 2} fill={col} opacity={0.16}>{sel && <animate attributeName="r" values={`${r * 1.4};${r * 2.6};${r * 1.4}`} dur="1.6s" repeatCount="indefinite" />}</circle>
            <path d={`M0 ${-r} L${r} 0 L0 ${r} L${-r} 0 Z`} fill={col} stroke="#fff" strokeWidth={unit * 0.2} />
            <text y={r * 0.4} textAnchor="middle" fontSize={r * 1.15} fontWeight={800} fill="#fff">!</text>
            <title>{i.title}</title>
          </g>
        );
      })}
      {camera && (() => {
        const [x, y] = P(camera.pos);
        const a = (-camera.yaw * 180) / Math.PI + 180;
        return (
          <g transform={`translate(${x} ${y}) rotate(${a})`}>
            <path d={`M0 0 L ${-unit * 2.6} ${unit * 5} A ${unit * 5.6} ${unit * 5.6} 0 0 0 ${unit * 2.6} ${unit * 5} Z`} fill="#3b82f6" opacity={0.18} />
            <circle r={unit * 1} fill="#2563eb" stroke="#fff" strokeWidth={unit * 0.35} />
          </g>
        );
      })()}
      {placing && hover && <circle cx={hover[0]} cy={hover[1]} r={unit * 1.2} fill="none" stroke="#7c3aed" strokeWidth={unit * 0.3} />}
    </svg>
  );
}
