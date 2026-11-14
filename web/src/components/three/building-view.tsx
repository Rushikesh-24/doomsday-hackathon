"use client";
// Stacked multi-floor building: each floor's measured corridor (floor + walls) at its level, tagged places,
// lift shafts and stairs between floors, and a route across floors. Building frame: x / z on the floor, y up.
import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { Html, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

export type BFloor = {
  id: string; level: number; name: string; label: string; y: number;
  left: number[][]; right: number[][]; centre: number[][];
  segments?: { left: number[][]; right: number[][]; centre: number[][] }[];
  places: { id: string; name: string; kind: string; position: number[] }[];
  transform: { source: string };
};
export type BConnector = { id: string; name: string; kind: "lift" | "stairs" | "flight"; out_of_service: boolean; floors: number[]; points: number[][] };
export type BModel = { id: string; name: string; floor_height: number; floors: BFloor[]; connectors: BConnector[] };

function FloorSlab({ f, y, active, dim }: { f: BFloor; y: number; active: boolean; dim: boolean }) {
  const pieces = f.segments?.length ? f.segments : [{ left: f.left, right: f.right, centre: f.centre }];
  return (
    <group>
      {pieces.map((p, i) => <Piece key={i} f={{ ...f, left: p.left, right: p.right, centre: p.centre }} y={y} active={active} dim={dim} label={i === 0} />)}
    </group>
  );
}

function Piece({ f, y, active, dim, label }: { f: BFloor; y: number; active: boolean; dim: boolean; label: boolean }) {
  const shape = useMemo(() => {
    if (f.left.length < 2) return null;
    const s = new THREE.Shape();
    s.moveTo(f.left[0][0], -f.left[0][1]);
    for (const p of f.left.slice(1)) s.lineTo(p[0], -p[1]);
    for (const p of [...f.right].reverse()) s.lineTo(p[0], -p[1]);
    s.closePath();
    return s;
  }, [f]);
  const op = dim ? 0.12 : active ? 0.95 : 0.55;
  const wallH = 1.1;
  const wall = (pts: number[][], h: number) => pts.map((p) => [p[0], y + h, p[1]] as [number, number, number]);
  return (
    <group>
      {shape && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, y, 0]}>
          <shapeGeometry args={[shape]} />
          <meshStandardMaterial color={active ? "#ede9fe" : "#e2e8f0"} transparent opacity={op} side={THREE.DoubleSide} />
        </mesh>
      )}
      {[f.left, f.right].map((pts, i) => pts.length > 1 && (
        <group key={i}>
          <Line points={wall(pts, 0.02)} color={dim ? "#cbd5e1" : "#0f172a"} lineWidth={active ? 2.5 : 1.5} transparent opacity={dim ? 0.3 : 1} />
          {!dim && <Line points={wall(pts, wallH)} color="#64748b" lineWidth={1} transparent opacity={0.45} />}
        </group>
      ))}
      {/* labels stay mounted (drei Html roots must not be created / destroyed mid-render); dimmed floors hide them */}
      <Html position={[f.centre[0]?.[0] ?? 0, y + 0.2, f.centre[0]?.[1] ?? 0]} center distanceFactor={14} zIndexRange={[10, 0]}>
        <div className={`pointer-events-none -translate-x-[110%] whitespace-nowrap rounded-full bg-slate-900 px-2 py-0.5 text-[11px] font-semibold text-white ${dim || !label ? "invisible" : ""}`}>{f.label}</div>
      </Html>
      {(label ? f.places : []).map((p) => (
        <group key={p.id} position={[p.position[0], y + 0.05, p.position[2]]} visible={!dim}>
          <mesh position={[0, 0.5, 0]}><cylinderGeometry args={[0.04, 0.04, 1, 6]} /><meshBasicMaterial color="#7c3aed" /></mesh>
          <mesh position={[0, 1.05, 0]}><sphereGeometry args={[0.18, 16, 12]} /><meshBasicMaterial color={p.kind === "lift" ? "#2563eb" : p.kind === "stairwell" || p.kind === "stairs" ? "#ea580c" : "#7c3aed"} /></mesh>
          <Html position={[0, 1.45, 0]} center distanceFactor={14} zIndexRange={[10, 0]}>
            <div className={`pointer-events-none whitespace-nowrap rounded-md bg-white/90 px-1.5 py-0.5 text-[10px] font-semibold text-slate-800 shadow ${dim ? "invisible" : ""}`}>{p.name}</div>
          </Html>
        </group>
      ))}
    </group>
  );
}

function Connector({ c, gap, fh }: { c: BConnector; gap: number; fh: number }) {
  if (c.points.length < 2) return null;
  if (c.kind === "flight") {   // a filmed stair climb: its real 3D path (heights spread like the floors)
    const pts = c.points.map((p) => [p[0], p[1] * gap + 0.1, p[2]] as [number, number, number]);
    return <Line points={pts} color="#ea580c" lineWidth={5} />;
  }
  const ys = c.floors.map((lv) => lv * fh * gap);
  const color = c.out_of_service ? "#94a3b8" : c.kind === "lift" ? "#2563eb" : "#ea580c";
  if (c.kind === "lift") {
    const [x, , z] = c.points[0];
    const y0 = Math.min(...ys), y1 = Math.max(...ys) + 2.4;
    return (
      <group>
        <mesh position={[x, (y0 + y1) / 2, z]}>
          <boxGeometry args={[1.4, y1 - y0, 1.4]} />
          <meshStandardMaterial color={color} transparent opacity={0.22} />
        </mesh>
        <lineSegments position={[x, (y0 + y1) / 2, z]}>
          <edgesGeometry args={[new THREE.BoxGeometry(1.4, y1 - y0, 1.4)]} />
          <lineBasicMaterial color={color} />
        </lineSegments>
        <Html position={[x, y1 + 0.4, z]} center distanceFactor={14}>
          <div className="pointer-events-none whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-semibold text-white" style={{ background: color }}>{c.name}{c.out_of_service ? " · out of service" : ""}</div>
        </Html>
      </group>
    );
  }
  // stairs: a zig-zag flight between each pair of served floors
  const pts: [number, number, number][] = [];
  c.points.forEach((p, i) => {
    const y = ys[i];
    if (i > 0) {
      const q = c.points[i - 1], yq = ys[i - 1];
      pts.push([(p[0] + q[0]) / 2 + 1.2, (y + yq) / 2, (p[2] + q[2]) / 2]);
    }
    pts.push([p[0], y + 0.05, p[2]]);
  });
  return (
    <group>
      <Line points={pts} color={color} lineWidth={4} dashed={c.out_of_service} dashSize={0.4} gapSize={0.3} />
      <Html position={[pts[pts.length - 1][0], pts[pts.length - 1][1] + 1.2, pts[pts.length - 1][2]]} center distanceFactor={14}>
        <div className="pointer-events-none whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-semibold text-white" style={{ background: color }}>{c.name}</div>
      </Html>
    </group>
  );
}

export default function BuildingView({ model, route, activeFloor, gap = 1, onlyActive = false }: {
  model: BModel; route?: { path: number[][]; path_floors: number[] } | null; activeFloor?: number | null; gap?: number; onlyActive?: boolean;
}) {
  const fh = model.floor_height;
  // frame the bulk of the building (median centre, 85th-percentile radius) so one stray, unlinked walk parked to
  // the side does not shrink everything else to a dot
  const all = model.floors.flatMap((f) => (f.segments?.length ? f.segments : [f]).flatMap((sg) => [...sg.left, ...sg.right]));
  const med = (v: number[]) => { const a = [...v].sort((x, y) => x - y); return a.length ? a[Math.floor(a.length / 2)] : 0; };
  const cx = med(all.map((p) => p[0])), cz = med(all.map((p) => p[1]));
  const top = (Math.max(0, ...model.floors.map((f) => f.level)) * fh * gap) / 2;
  const radii = all.map((p) => Math.hypot(p[0] - cx, p[1] - cz)).sort((x, y) => x - y);
  const span = Math.max(8, radii.length ? radii[Math.floor(radii.length * 0.85)] : 10, top * 1.4);
  // route: floor-level line, lifted onto the stacked floors (local heights are replaced by the floor level)
  const routePts = route?.path.map((p, i) => [p[0], route.path_floors[i] * fh * gap + 0.25, p[2]] as [number, number, number]);
  return (
    <Canvas camera={{ position: [cx + span * 1.6, top + span * 0.9, cz + span * 1.8], fov: 45 }} gl={{ antialias: true }}>
      <color attach="background" args={["#f8fafc"]} />
      <ambientLight intensity={0.9} />
      <directionalLight position={[10, 30, 10]} intensity={0.6} />
      <OrbitControls target={[cx, top, cz]} makeDefault enableDamping />
      {model.floors.map((f) => {
        const dim = activeFloor != null && f.level !== activeFloor;
        return <group key={f.id} visible={!(onlyActive && dim)}><FloorSlab f={f} y={f.level * fh * gap} active={activeFloor === f.level} dim={dim} /></group>;
      })}
      {model.connectors.map((c) => <Connector key={c.id} c={c} gap={gap} fh={fh} />)}
      {routePts && routePts.length > 1 && (
        <>
          <Line points={routePts} color="#f43f5e" lineWidth={6} />
          <mesh position={routePts[0]}><sphereGeometry args={[0.3, 16, 12]} /><meshBasicMaterial color="#16a34a" /></mesh>
          <mesh position={routePts[routePts.length - 1]}><sphereGeometry args={[0.35, 16, 12]} /><meshBasicMaterial color="#f43f5e" /></mesh>
        </>
      )}
    </Canvas>
  );
}
