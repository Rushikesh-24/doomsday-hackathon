"use client";
// Real reconstruction viewer: Brush/in-house Gaussian splat (GaussianSplats3D drop-in), COLMAP sparse point
// cloud, camera trajectory and registered facility boxes, all in the gravity-aligned local frame
// (SfM-frame content is placed inside a group carrying the stored T_local_from_sfm matrix).
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Line, Grid } from "@react-three/drei";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { apiUrl } from "@/lib/api";
import { CLASS_COLORS } from "@/components/app/common";

export type SceneFacility = { facility_id: string; class_name: string; bbox_sfm?: number[][] | null; position_sfm?: number[] | null; registration_confidence?: number };
export type SceneData = {
  T: number[][];                       // T_local_from_sfm (4x4 row-major)
  splatUrl?: string | null;
  pointsUrl?: string | null;
  trajectory: { center_sfm: number[]; center_local: number[] }[];
  facilities: SceneFacility[];
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
function Label({ text, color = "#ffffff", bg = "#0f172a", height }: { text: string; color?: string; bg?: string; height: number }) {
  const { tex, aspect } = useMemo(() => {
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d")!;
    const font = "600 44px system-ui, sans-serif";
    ctx.font = font;
    const w = Math.ceil(ctx.measureText(text).width) + 32;
    c.width = w; c.height = 64;
    ctx.font = font;
    ctx.fillStyle = bg;
    ctx.beginPath(); ctx.roundRect(0, 0, w, 64, 12); ctx.fill();
    ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, 16, 34);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return { tex: t, aspect: w / 64 };
  }, [text, color, bg]);
  useEffect(() => () => tex.dispose(), [tex]);
  return (
    <sprite scale={[height * aspect, height, 1]} position={[0, height * 1.2, 0]} renderOrder={10}>
      <spriteMaterial map={tex} depthTest={false} transparent />
    </sprite>
  );
}

function FitCamera({ center, radius }: { center: THREE.Vector3; radius: number }) {
  const { camera } = useThree();
  useEffect(() => {
    // mutating the r3f camera imperatively is the intended three.js pattern
    const cam = camera as THREE.PerspectiveCamera;
    cam.position.set(center.x + radius * 0.9, center.y + radius * 0.8, center.z + radius * 0.9);
    cam.lookAt(center);
    Object.assign(cam, { near: radius / 500, far: radius * 50 });
    cam.updateProjectionMatrix();
  }, [camera, center, radius]);
  return null;
}

export default function SceneViewer({ data, layers, route, highlight, onPick, markers }: {
  data: SceneData;
  layers: { splat: boolean; points: boolean; trajectory: boolean; facilities: boolean };
  route?: number[][] | null;
  highlight?: string | null;
  onPick?: (id: string) => void;
  markers?: { pos: number[]; label: string; color?: string }[];
}) {
  const [splatState, setSplatState] = useState("idle");
  const M = useMemo(() => {
    const m = new THREE.Matrix4();
    const t = data.T;
    m.set(t[0][0], t[0][1], t[0][2], t[0][3], t[1][0], t[1][1], t[1][2], t[1][3], t[2][0], t[2][1], t[2][2], t[2][3], 0, 0, 0, 1);
    return m;
  }, [data.T]);
  const traj = useMemo(() => data.trajectory.map((c) => new THREE.Vector3(...(c.center_local as [number, number, number]))), [data.trajectory]);
  const { center, radius } = useMemo(() => {
    const b = new THREE.Box3().setFromPoints(traj.length ? traj : [new THREE.Vector3()]);
    const c = b.getCenter(new THREE.Vector3());
    return { center: c, radius: Math.max(b.getSize(new THREE.Vector3()).length(), 2) };
  }, [traj]);
  const scale = Math.cbrt(Math.abs(M.determinant())) || 1;
  return (
    <div className="relative h-full w-full">
      <Canvas camera={{ fov: 55 }} gl={{ antialias: true }} onCreated={({ gl }) => gl.setClearColor("#f4f7fa")}>
        <FitCamera center={center} radius={radius} />
        <OrbitControls target={center} makeDefault enableDamping />
        <ambientLight intensity={0.9} />
        <Grid position={[center.x, center.y - radius * 0.35, center.z]} args={[radius * 4, radius * 4]} cellColor="#d5dde5" sectionColor="#b8c4cf" fadeDistance={radius * 6} infiniteGrid />
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
                <mesh onClick={(e) => { e.stopPropagation(); onPick?.(f.facility_id); }}>
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
        {layers.trajectory && traj.length > 1 && <Line points={traj} color="#0e7490" lineWidth={2} />}
        {layers.trajectory && traj.length > 0 && (
          <mesh position={traj[0]}><sphereGeometry args={[radius * 0.012]} /><meshBasicMaterial color="#16a34a" /></mesh>
        )}
        {route && route.length > 1 && (
          <Line points={route.map((p) => new THREE.Vector3(...(p as [number, number, number])))} color="#e11d48" lineWidth={5} />
        )}
        {markers?.map((m, i) => (
          <group key={i} position={m.pos as [number, number, number]}>
            <mesh><sphereGeometry args={[radius * 0.02]} /><meshBasicMaterial color={m.color || "#e11d48"} /></mesh>
<Label text={m.label} color="#0f172a" bg="#ffffffee" height={radius * 0.04} />
          </group>
        ))}
      </Canvas>
      {layers.splat && data.splatUrl && splatState !== "ready" && (
        <div className="absolute left-3 top-3 rounded-md bg-white/90 px-2 py-1 text-xs shadow">
          Gaussian splat: {splatState === "loading" ? "loading…" : splatState}
        </div>
      )}
    </div>
  );
}
