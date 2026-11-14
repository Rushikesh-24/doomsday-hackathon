"use client";
/* eslint-disable react-hooks/immutability -- three.js cameras/controls are mutated imperatively inside useFrame
   and event handlers; that is the intended react-three-fiber pattern, not render-time mutation. */
// Reconstruction viewer: Gaussian splat (GaussianSplats3D drop-in), COLMAP sparse point cloud, camera
// trajectory, registered facility boxes and user-tagged places, all in the gravity-aligned local frame (+Y up).
// SfM-frame content sits inside a group carrying the stored T_local_from_sfm matrix.
//
// Modes:  orbit - drag to orbit, scroll to zoom (OrbitControls)
//         walk  - first person at eye height along the walked floor: WASD / arrows move, drag to look,
//                 Shift = faster. Eye height follows the nearest walked keyframe (stairs go up and down).
//         tour  - camera glides along a given path (route or the walk itself), looking ahead.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line, Grid } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { apiUrl } from "@/lib/api";
import { CLASS_COLORS } from "@/components/app/common";

export type SceneFacility = { facility_id: string; class_name: string; bbox_sfm?: number[][] | null; position_sfm?: number[] | null; registration_confidence?: number };
export type SceneData = {
  T: number[][];                       // T_local_from_sfm (4x4 row-major)
  splatUrl?: string | null;
  pointsUrl?: string | null;
  trajectory: { center_sfm: number[]; center_local: number[]; forward_local?: number[] }[];
  facilities: SceneFacility[];
  metric?: boolean;                    // local frame is in metres (scale_status != relative)
};
export type PlacePin = { id: string; name: string; kind: string; pos: number[]; color?: string };
export type ViewMode = "orbit" | "walk";
export type CameraPose = { pos: number[]; yaw: number };
export type Tour = { path: number[][]; nonce: number; speed?: number } | null;

export const PLACE_COLORS: Record<string, string> = {
  lab: "#7c3aed", classroom: "#2563eb", office: "#0891b2", entrance: "#16a34a", toilet: "#0ea5e9", lift: "#4f46e5",
  stairs: "#dc2626", stairwell: "#f97316", ramp: "#0d9488", library: "#b45309", canteen: "#ea580c", help_desk: "#db2777",
  meeting_point: "#65a30d", other: "#475569",
};

function Points({ url }: { url: string }) {
  const [geo, setGeo] = useState<THREE.BufferGeometry | null>(null);
  useEffect(() => {
    let alive = true;
    new PLYLoader().load(apiUrl(url), (g) => alive && setGeo(g));
    return () => { alive = false; };
  }, [url]);
  if (!geo) return null;
  return (
    <points geometry={geo}>
      <pointsMaterial size={0.035} vertexColors sizeAttenuation />
    </points>
  );
}

function Splat({ url, onState }: { url: string; onState: (s: string) => void }) {
  const [obj, setObj] = useState<THREE.Object3D | null>(null);
  useEffect(() => {
    let disposed = false;
    let viewer: { dispose?: () => void } | null = null;
    onState("loading");
    import("@mkkellogg/gaussian-splats-3d").then((GS) => {
      if (disposed) return;
      const v = new GS.DropInViewer({ gpuAcceleratedSort: false, sharedMemoryForWorkers: false });
      viewer = v as unknown as { dispose?: () => void };
      v.addSplatScene(apiUrl(url), { format: GS.SceneFormat.Ply, showLoadingUI: false, splatAlphaRemovalThreshold: 5 })
        .then(() => !disposed && onState("ready"))
        .catch((e: Error) => onState(`error: ${e.message}`));
      setObj(v as unknown as THREE.Object3D);
    });
    return () => { disposed = true; viewer?.dispose?.(); };
  }, [url, onState]);
  return obj ? <primitive object={obj} /> : null;
}

/** Text label as a canvas-texture sprite (no nested React roots, unlike drei <Html>). */
function Label({ text, color = "#ffffff", bg = "#0f172a", height, lift = 1.2 }: { text: string; color?: string; bg?: string; height: number; lift?: number }) {
  const { tex, aspect } = useMemo(() => {
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d")!;
    const font = "600 44px system-ui, sans-serif";
    ctx.font = font;
    const w = Math.ceil(ctx.measureText(text).width) + 40;
    c.width = w; c.height = 68;
    ctx.font = font;
    ctx.fillStyle = bg;
    ctx.beginPath(); ctx.roundRect(0, 0, w, 68, 34); ctx.fill();
    ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, 20, 36);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return { tex: t, aspect: w / 68 };
  }, [text, color, bg]);
  useEffect(() => () => tex.dispose(), [tex]);
  return (
    <sprite scale={[height * aspect, height, 1]} position={[0, height * lift, 0]} renderOrder={10}>
      <spriteMaterial map={tex} depthTest={false} transparent />
    </sprite>
  );
}

function FitCamera({ center, radius }: { center: THREE.Vector3; radius: number }) {
  const { camera } = useThree();
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    cam.position.set(center.x + radius * 0.9, center.y + radius * 0.8, center.z + radius * 0.9);
    cam.lookAt(center);
    Object.assign(cam, { near: radius / 2000, far: radius * 50 });
    cam.updateProjectionMatrix();
  }, [camera, center, radius]);
  return null;
}

/** Nearest walked keyframe (by ground distance) -> eye height there. */
function eyeHeightAt(traj: THREE.Vector3[], x: number, z: number): number {
  let best = Infinity, y = traj[0]?.y ?? 0;
  for (const p of traj) {
    const d = (p.x - x) ** 2 + (p.z - z) ** 2;
    if (d < best) { best = d; y = p.y; }
  }
  return y;
}

type Keys = Record<string, boolean>;
export type WalkInput = { forward: number; turn: number };

function WalkRig({ traj, step, start, onCamera, input }: {
  traj: THREE.Vector3[]; step: number; start: { pos: THREE.Vector3; yaw: number; pitch?: number };
  onCamera?: (p: CameraPose) => void; input: React.MutableRefObject<WalkInput>;
}) {
  const { camera, gl } = useThree();
  const st = useRef({ yaw: start.yaw, pitch: 0, keys: {} as Keys, drag: null as null | { x: number; y: number }, last: 0 });
  useEffect(() => {
    camera.position.copy(start.pos);
    st.current.yaw = start.yaw;
    st.current.pitch = start.pitch ?? 0;
  }, [camera, start]);
  useEffect(() => {
    const el = gl.domElement;
    const kd = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.("input,textarea,select")) return;
      st.current.keys[e.code] = true;
      if (e.code.startsWith("Arrow") || e.code === "Space") e.preventDefault();
    };
    const ku = (e: KeyboardEvent) => { st.current.keys[e.code] = false; };
    const pd = (e: PointerEvent) => { st.current.drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); };
    const pm = (e: PointerEvent) => {
      const d = st.current.drag;
      if (!d) return;
      st.current.yaw -= (e.clientX - d.x) * 0.004;
      st.current.pitch = Math.max(-1.2, Math.min(1.2, st.current.pitch - (e.clientY - d.y) * 0.004));
      st.current.drag = { x: e.clientX, y: e.clientY };
    };
    const pu = () => { st.current.drag = null; };
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);
    el.addEventListener("pointerdown", pd);
    el.addEventListener("pointermove", pm);
    el.addEventListener("pointerup", pu);
    el.addEventListener("pointerleave", pu);
    return () => {
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
      el.removeEventListener("pointerdown", pd);
      el.removeEventListener("pointermove", pm);
      el.removeEventListener("pointerup", pu);
      el.removeEventListener("pointerleave", pu);
    };
  }, [gl]);
  useFrame((state, rawDt) => {
    const s = st.current, k = s.keys;
    const dt = Math.min(rawDt, 0.1);
    const fast = k.ShiftLeft || k.ShiftRight ? 3 : 1;
    const speed = step * 6 * fast;
    let fwd = (k.KeyW || k.ArrowUp ? 1 : 0) - (k.KeyS || k.ArrowDown ? 1 : 0) + input.current.forward;
    const strafe = (k.KeyD ? 1 : 0) - (k.KeyA ? 1 : 0);
    const turn = (k.ArrowLeft || k.KeyQ ? 1 : 0) - (k.ArrowRight || k.KeyE ? 1 : 0) + input.current.turn;
    fwd = Math.max(-1, Math.min(1, fwd));
    s.yaw += turn * dt * 1.6;
    const f = new THREE.Vector3(-Math.sin(s.yaw), 0, -Math.cos(s.yaw));
    const r = new THREE.Vector3(Math.cos(s.yaw), 0, -Math.sin(s.yaw));
    const p = camera.position;
    p.addScaledVector(f, fwd * speed * dt).addScaledVector(r, strafe * speed * dt);
    const ty = eyeHeightAt(traj, p.x, p.z);
    p.y += (ty - p.y) * Math.min(1, dt * 4);
    camera.quaternion.setFromEuler(new THREE.Euler(s.pitch, s.yaw, 0, "YXZ"));
    if (onCamera && state.clock.elapsedTime - s.last > 0.15) {
      s.last = state.clock.elapsedTime;
      onCamera({ pos: [p.x, p.y, p.z], yaw: s.yaw });
    }
  });
  return null;
}

/** Glide the camera along a polyline at a constant speed, looking ahead. */
function TourRig({ path, speed, onDone, onCamera }: { path: THREE.Vector3[]; speed: number; onDone?: () => void; onCamera?: (p: CameraPose) => void }) {
  const { camera } = useThree();
  const s = useRef({ d: 0, last: 0, done: false, look: new THREE.Vector3() });
  const cum = useMemo(() => {
    const c = [0];
    for (let i = 1; i < path.length; i++) c.push(c[i - 1] + path[i].distanceTo(path[i - 1]));
    return c;
  }, [path]);
  const at = (d: number) => {
    const L = cum[cum.length - 1];
    d = Math.max(0, Math.min(L, d));
    let i = 1;
    while (i < cum.length - 1 && cum[i] < d) i++;
    const seg = cum[i] - cum[i - 1] || 1;
    return path[i - 1].clone().lerp(path[i], (d - cum[i - 1]) / seg);
  };
  useEffect(() => {
    s.current = { d: 0, last: 0, done: false, look: at(speed * 1.5) };
    camera.position.copy(path[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path]);
  useFrame((state, dt) => {
    const L = cum[cum.length - 1];
    const t = s.current;
    t.d = Math.min(L, t.d + speed * Math.min(dt, 0.1));
    const p = at(t.d);
    camera.position.lerp(p, 0.25);
    const ahead = at(t.d + speed * 1.5);
    if (ahead.distanceTo(p) > 1e-6) t.look.lerp(ahead, 0.08);
    const lk = t.look.clone();
    lk.y = camera.position.y - (camera.position.y - lk.y) * 0.3;
    camera.lookAt(lk);
    if (onCamera && state.clock.elapsedTime - t.last > 0.15) {
      t.last = state.clock.elapsedTime;
      const dir = lk.clone().sub(camera.position);
      onCamera({ pos: [camera.position.x, camera.position.y, camera.position.z], yaw: Math.atan2(-dir.x, -dir.z) });
    }
    if (t.d >= L && !t.done) { t.done = true; onDone?.(); }
  });
  return null;
}

/** Smoothly move the orbit camera to look at a target. */
function OrbitFocus({ focus, radius }: { focus: { pos: number[]; nonce: number } | null | undefined; radius: number }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.PerspectiveCamera; controls: { target: THREE.Vector3; update: () => void } | null };
  const anim = useRef<{ t: number; fromP: THREE.Vector3; toP: THREE.Vector3; fromT: THREE.Vector3; toT: THREE.Vector3 } | null>(null);
  useEffect(() => {
    if (!focus || !controls) return;
    const tgt = new THREE.Vector3(...(focus.pos as [number, number, number]));
    const dir = camera.position.clone().sub(controls.target).normalize();
    const toP = tgt.clone().add(dir.multiplyScalar(radius * 0.35)).add(new THREE.Vector3(0, radius * 0.12, 0));
    anim.current = { t: 0, fromP: camera.position.clone(), toP, fromT: controls.target.clone(), toT: tgt };
  }, [focus, controls, camera, radius]);
  useFrame((_, dt) => {
    const a = anim.current;
    if (!a || !controls) return;
    a.t = Math.min(1, a.t + dt * 1.2);
    const e = 1 - Math.pow(1 - a.t, 3);
    camera.position.lerpVectors(a.fromP, a.toP, e);
    controls.target.lerpVectors(a.fromT, a.toT, e);
    controls.update();
    if (a.t >= 1) anim.current = null;
  });
  return null;
}

function PlacePinMesh({ p, size, stick, selected, onClick }: { p: PlacePin; size: number; stick: number; selected: boolean; onClick?: () => void }) {
  const col = p.color || PLACE_COLORS[p.kind] || "#475569";
  const ring = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (ring.current) {
      const s = 1 + 0.35 * Math.sin(clock.elapsedTime * 3);
      ring.current.scale.set(s, s, s);
    }
  });
  return (
    <group position={p.pos as [number, number, number]}>
      <mesh position={[0, stick, 0]} onClick={(e) => { e.stopPropagation(); onClick?.(); }}>
        <sphereGeometry args={[size * 0.45, 20, 20]} />
        <meshBasicMaterial color={col} />
      </mesh>
      <mesh position={[0, stick / 2, 0]}>
        <cylinderGeometry args={[size * 0.06, size * 0.06, stick, 8]} />
        <meshBasicMaterial color={col} />
      </mesh>
      <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[size * 0.5, size * (selected ? 0.8 : 0.62), 32]} />
        <meshBasicMaterial color={col} transparent opacity={selected ? 0.9 : 0.5} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <group position={[0, stick + size * 0.3, 0]}><Label text={p.name} bg={col} height={size * (selected ? 1.15 : 0.9)} lift={0.6} /></group>
    </group>
  );
}

/** Animated route: thick line + a pulse travelling from start to goal + chevrons. */
function RouteLine({ route, size }: { route: number[][]; size: number }) {
  const pts = useMemo(() => route.map((p) => new THREE.Vector3(...(p as [number, number, number]))), [route]);
  const pulse = useRef<THREE.Mesh>(null);
  const cum = useMemo(() => {
    const c = [0];
    for (let i = 1; i < pts.length; i++) c.push(c[i - 1] + pts[i].distanceTo(pts[i - 1]));
    return c;
  }, [pts]);
  const chevrons = useMemo(() => {
    const out: { p: THREE.Vector3; q: THREE.Quaternion }[] = [];
    const L = cum[cum.length - 1];
    const n = Math.min(40, Math.max(3, Math.floor(L / (size * 3))));
    for (let k = 1; k < n; k++) {
      const d = (k / n) * L;
      let i = 1;
      while (i < cum.length - 1 && cum[i] < d) i++;
      const a = pts[i - 1], b = pts[i];
      const p = a.clone().lerp(b, (d - cum[i - 1]) / (cum[i] - cum[i - 1] || 1));
      const dir = b.clone().sub(a).normalize();
      out.push({ p, q: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir) });
    }
    return out;
  }, [pts, cum, size]);
  useFrame(({ clock }) => {
    const L = cum[cum.length - 1];
    if (!pulse.current || L <= 0) return;
    const d = ((clock.elapsedTime * size * 4) % L);
    let i = 1;
    while (i < cum.length - 1 && cum[i] < d) i++;
    pulse.current.position.copy(pts[i - 1].clone().lerp(pts[i], (d - cum[i - 1]) / (cum[i] - cum[i - 1] || 1)));
  });
  return (
    <group>
      <Line points={pts} color="#f43f5e" lineWidth={6} transparent opacity={0.9} depthTest={false} renderOrder={5} />
      <Line points={pts} color="#ffffff" lineWidth={1.5} dashed dashSize={size * 0.6} gapSize={size * 0.6} depthTest={false} renderOrder={6} />
      {chevrons.map((c, i) => (
        <mesh key={i} position={c.p} quaternion={c.q} renderOrder={7}>
          <coneGeometry args={[size * 0.28, size * 0.6, 3]} />
          <meshBasicMaterial color="#be123c" depthTest={false} transparent />
        </mesh>
      ))}
      <mesh ref={pulse} renderOrder={8}>
        <sphereGeometry args={[size * 0.35, 16, 16]} />
        <meshBasicMaterial color="#fff1f2" depthTest={false} transparent />
      </mesh>
    </group>
  );
}

/** Invisible picking plane on the walked floor; click places a pin (stored at the nearest keyframe's eye height). */
function PlacePicker({ traj, y, drop, radius, onPlace }: { traj: THREE.Vector3[]; y: number; drop: number; radius: number; onPlace: (p: number[]) => void }) {
  const [hover, setHover] = useState<THREE.Vector3 | null>(null);
  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, y - drop, 0]}
        onPointerMove={(e) => setHover(e.point.clone())} onPointerOut={() => setHover(null)}
        onClick={(e) => { e.stopPropagation(); const p = e.point; onPlace([p.x, eyeHeightAt(traj, p.x, p.z), p.z]); }}>
        <planeGeometry args={[radius * 40, radius * 40]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      {hover && (
        <mesh position={[hover.x, eyeHeightAt(traj, hover.x, hover.z) - drop, hover.z]} rotation={[-Math.PI / 2, 0, 0]} renderOrder={9}>
          <ringGeometry args={[radius * 0.012, radius * 0.022, 32]} />
          <meshBasicMaterial color="#7c3aed" depthTest={false} transparent side={THREE.DoubleSide} />
        </mesh>
      )}
    </>
  );
}

export default function SceneViewer({ data, layers, route, highlight, onPick, markers, places, selectedPlace, onPlaceClick,
  mode = "orbit", placing = false, onPlace, tour, onTourEnd, focus, onCamera, walkInput, walkStart, background = "#eef2f7" }: {
  data: SceneData;
  layers: { splat: boolean; points: boolean; trajectory: boolean; facilities: boolean };
  route?: number[][] | null;
  highlight?: string | null;
  onPick?: (id: string) => void;
  markers?: { pos: number[]; label: string; color?: string }[];
  places?: PlacePin[];
  selectedPlace?: string | null;
  onPlaceClick?: (id: string) => void;
  mode?: ViewMode;
  placing?: boolean;
  onPlace?: (pos: number[]) => void;
  tour?: Tour;
  onTourEnd?: () => void;
  focus?: { pos: number[]; nonce: number } | null;
  onCamera?: (p: CameraPose) => void;
  walkInput?: React.MutableRefObject<WalkInput>;
  walkStart?: { pos: number[]; yaw: number; nonce: number } | null;
  background?: string;
}) {
  const [splatState, setSplatState] = useState("idle");
  const fallbackInput = useRef<WalkInput>({ forward: 0, turn: 0 });
  const M = useMemo(() => {
    const m = new THREE.Matrix4();
    const t = data.T;
    m.set(t[0][0], t[0][1], t[0][2], t[0][3], t[1][0], t[1][1], t[1][2], t[1][3], t[2][0], t[2][1], t[2][2], t[2][3], 0, 0, 0, 1);
    return m;
  }, [data.T]);
  const traj = useMemo(() => data.trajectory.map((c) => new THREE.Vector3(...(c.center_local as [number, number, number]))), [data.trajectory]);
  const { center, radius, step, eyeY } = useMemo(() => {
    const b = new THREE.Box3().setFromPoints(traj.length ? traj : [new THREE.Vector3()]);
    const c = b.getCenter(new THREE.Vector3());
    const steps = traj.slice(1).map((p, i) => p.distanceTo(traj[i])).sort((a, b) => a - b);
    const ys = traj.map((p) => p.y).sort((a, b) => a - b);
    return { center: c, radius: Math.max(b.getSize(new THREE.Vector3()).length(), 2), step: steps[Math.floor(steps.length / 2)] || 0.1, eyeY: ys[Math.floor(ys.length / 2)] ?? 0 };
  }, [traj]);
  const scale = Math.cbrt(Math.abs(M.determinant())) || 1;
  const pin = radius * 0.025;
  const startPose = useMemo(() => {
    if (walkStart) return { pos: new THREE.Vector3(...(walkStart.pos as [number, number, number])), yaw: walkStart.yaw, pitch: 0 };
    const p0 = traj[0] || new THREE.Vector3();
    const f = data.trajectory[0]?.forward_local;          // where the phone actually looked at the start
    if (f) return { pos: p0.clone(), yaw: Math.atan2(-f[0], -f[2]), pitch: Math.max(-0.35, Math.asin(Math.max(-1, Math.min(1, f[1]))) * 0.5) };
    const far = traj.find((p) => p.distanceTo(p0) > 2 * step) || p0.clone().add(new THREE.Vector3(0, 0, -1));
    const d = far.clone().sub(p0);
    return { pos: p0.clone(), yaw: Math.atan2(-d.x, -d.z), pitch: 0 };
  }, [walkStart, traj, data.trajectory, step]);
  // eye height above the floor: ~1.4 m for a hand-held phone; unscaled scans use ~4.5 walking steps
  const drop = data.metric ? 1.4 : step * 4.5;
  const floorTraj = useMemo(() => traj.map((p) => p.clone().setY(p.y - drop)), [traj, drop]);
  const floorRoute = useMemo(() => route?.map((p) => [p[0], p[1] - drop, p[2]]) ?? null, [route, drop]);
  const tourPath = useMemo(() => tour?.path.map((p) => new THREE.Vector3(...(p as [number, number, number]))) ?? null, [tour]);
  const orbit = mode === "orbit" && !tourPath;
  return (
    <div className="relative h-full w-full">
      <Canvas camera={{ fov: 62 }} gl={{ antialias: true }} onCreated={(state) => {
        state.gl.setClearColor(background);
        if (process.env.NODE_ENV !== "production") (window as unknown as { __r3f?: unknown }).__r3f = state;   // dev inspection
      }}>
        <FitCamera center={center} radius={radius} />
        {orbit && <OrbitControls target={center} makeDefault enableDamping />}
        {orbit && <OrbitFocus focus={focus} radius={radius} />}
        {mode === "walk" && !tourPath && <WalkRig traj={traj} step={step} start={startPose} onCamera={onCamera} input={walkInput || fallbackInput} />}
        {tourPath && tourPath.length > 1 && <TourRig key={tour!.nonce} path={tourPath} speed={tour!.speed ?? step * 5} onDone={onTourEnd} onCamera={onCamera} />}
        <ambientLight intensity={0.9} />
        {mode === "orbit" && <Grid position={[center.x, center.y - radius * 0.35, center.z]} args={[radius * 4, radius * 4]} cellColor="#d5dde5" sectionColor="#b8c4cf" fadeDistance={radius * 6} infiniteGrid />}
        <group matrixAutoUpdate={false} matrix={M}>
          {layers.splat && data.splatUrl && <Splat url={data.splatUrl} onState={setSplatState} />}
          {layers.points && data.pointsUrl && <Points url={data.pointsUrl} />}
          {layers.facilities && data.facilities.filter((f) => f.bbox_sfm).map((f) => {
            const lo = new THREE.Vector3(...(f.bbox_sfm![0] as [number, number, number]));
            const hi = new THREE.Vector3(...(f.bbox_sfm![1] as [number, number, number]));
            const size = hi.clone().sub(lo);
            const mid = lo.clone().add(hi).multiplyScalar(0.5);
            const col = CLASS_COLORS[f.class_name] || "#0ea5e9";
            const on = highlight === f.facility_id;
            return (
              <group key={f.facility_id} position={mid}>
                <mesh onClick={(e) => { if (placing) return; e.stopPropagation(); onPick?.(f.facility_id); }}>
                  <boxGeometry args={[size.x, size.y, size.z]} />
                  <meshBasicMaterial color={col} transparent opacity={on ? 0.35 : 0.12} depthWrite={false} />
                </mesh>
                <lineSegments>
                  <edgesGeometry args={[new THREE.BoxGeometry(size.x, size.y, size.z)]} />
                  <lineBasicMaterial color={col} />
                </lineSegments>
                {on && <Label text={f.class_name.replaceAll("_", " ")} bg={col} height={radius * 0.04 / scale} />}
              </group>
            );
          })}
        </group>
        {layers.trajectory && traj.length > 1 && <Line points={floorTraj} color="#0e7490" lineWidth={2} transparent opacity={route ? 0.35 : 0.9} />}
        {layers.trajectory && traj.length > 0 && (
          <mesh position={floorTraj[0]}><sphereGeometry args={[pin * 0.5]} /><meshBasicMaterial color="#16a34a" /></mesh>
        )}
        {floorRoute && floorRoute.length > 1 && <RouteLine route={floorRoute} size={Math.min(pin, data.metric ? 0.35 : pin)} />}
        {places?.map((p) => <PlacePinMesh key={p.id} p={{ ...p, pos: [p.pos[0], p.pos[1] - drop, p.pos[2]] }} size={Math.min(pin, data.metric ? 0.45 : pin)} stick={drop} selected={selectedPlace === p.id} onClick={() => !placing && onPlaceClick?.(p.id)} />)}
        {markers?.map((m, i) => (
          <group key={i} position={m.pos as [number, number, number]}>
            <mesh><sphereGeometry args={[pin * 0.6]} /><meshBasicMaterial color={m.color || "#e11d48"} /></mesh>
            <Label text={m.label} color="#0f172a" bg="#ffffffee" height={pin * 0.9} />
          </group>
        ))}
        {placing && onPlace && <PlacePicker traj={traj} y={eyeY} drop={drop} radius={radius} onPlace={onPlace} />}
      </Canvas>
      {layers.splat && data.splatUrl && splatState !== "ready" && (
        <div className="absolute left-3 top-3 rounded-full bg-white/90 px-3 py-1 text-xs shadow">
          Gaussian splat: {splatState === "loading" ? "loading…" : splatState}
        </div>
      )}
    </div>
  );
}
