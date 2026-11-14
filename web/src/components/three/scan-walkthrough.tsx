"use client";
// The 3D Explorer's viewer as a self-contained panel: one scan's 3D render (Gaussian splat) with Orbit / Walk /
// Replay-walk, the floor plan (3D | Split | Plan), tagged places, a mini-map while walking and an optional route
// drawn in the scan's own frame. Used by the Buildings page to walk through each floor of a building.
import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Box, ExternalLink, Footprints, Map as MapIcon, Play, Rotate3d, X } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { useScene } from "@/hooks/use-scene";
import type { Place, PlanInfo } from "@/lib/places";
import { cn } from "@/lib/utils";
import { ErrorState, Loading } from "@/components/app/common";
import type { CameraPose, PlacePin, Tour, ViewMode, WalkInput } from "@/components/three/scene-viewer";

const SceneViewer = dynamic(() => import("@/components/three/scene-viewer"), { ssr: false, loading: () => <Loading label="Starting 3D viewer…" /> });
const PlanView = dynamic(() => import("@/components/three/plan-view"), { ssr: false });

function Seg<T extends string>({ value, options, onChange }: { value: T; options: { v: T; label: string; icon?: React.ReactNode }[]; onChange: (v: T) => void }) {
  return (
    <div className="glass inline-flex rounded-full p-1">
      {options.map((o) => (
        <button key={o.v} onClick={() => onChange(o.v)}
          className={cn("flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all",
            value === o.v ? "bg-primary text-primary-foreground shadow" : "text-foreground/70 hover:text-foreground")}>
          {o.icon}{o.label}
        </button>
      ))}
    </div>
  );
}

export default function ScanWalkthrough({ rid, title, routeIds, topLeft }: {
  rid: string;
  title?: string;
  routeIds?: string[] | null;    // the route's nodes inside this scan ("cam12", place ids), in walking order
  topLeft?: React.ReactNode;     // extra controls (e.g. floor / walk pickers) placed in the toolbar
}) {
  const s = useScene(rid);
  const placesApi = useApi<{ places: Place[] }>(s.scene ? `/navigation/${rid}/places` : null);
  const planApi = useApi<PlanInfo>(s.scene ? `/navigation/${rid}/plan` : null);
  const [view, setView] = useState<"3d" | "split" | "plan">("3d");
  const [mode, setMode] = useState<ViewMode>("walk");
  const [tour, setTour] = useState<Tour>(null);
  const [cam, setCam] = useState<CameraPose | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const walkInput = useRef<WalkInput>({ forward: 0, turn: 0 });
  const press = useCallback((k: keyof WalkInput, v: number) => { walkInput.current[k] = v; }, []);
  const onCamera = useCallback((p: CameraPose) => setCam(p), []);
  const pins: PlacePin[] = useMemo(() => (placesApi.data?.places ?? []).map((p) => ({ id: p.id, name: p.name, kind: p.kind, pos: p.position })), [placesApi.data]);
  const path = useMemo(() => (s.scene?.trajectory ?? []).filter((c) => c.center_local).map((c) => c.center_local), [s.scene]);
  // graph node camK = K-th registered camera of the walk
  const route = useMemo(() => {
    if (!routeIds?.length) return null;
    const pts = routeIds.map((id) => (/^cam\d+$/.test(id) ? path[+id.slice(3)] : pins.find((p) => p.id === id)?.pos)).filter(Boolean) as number[][];
    return pts.length > 1 ? pts : null;
  }, [routeIds, path, pins]);
  const plan = planApi.data;
  const liveCam = mode === "walk" || tour ? cam : null;
  const layers = { splat: true, points: false, trajectory: true, facilities: false };

  if (s.error) return <div className="grid h-full place-items-center p-6"><ErrorState message={s.error} /></div>;
  if (!s.scene) return <div className="grid h-full place-items-center"><Loading label="Loading 3D render…" /></div>;

  const viewer = (
    <SceneViewer key={rid} data={s.scene} layers={layers} route={route} places={pins} selectedPlace={sel} onPlaceClick={setSel}
      mode={mode} tour={tour} onTourEnd={() => setTour(null)} onCamera={onCamera} walkInput={walkInput} />
  );
  const planEl = plan?.generated && (
    <PlanView plan={plan} path={path} places={pins} camera={liveCam} selectedPlace={sel} route={route}
      onPlaceClick={(id) => setSel(id)} />
  );
  return (
    <div className="relative h-full">
      <div className={cn("grid h-full", view === "split" ? "grid-rows-2 lg:grid-cols-2 lg:grid-rows-1" : "grid-cols-1")}>
        {view !== "plan" && <div className="relative min-h-0">{viewer}</div>}
        {view !== "3d" && (
          <div className={cn("relative min-h-0 bg-slate-50", view === "split" && "border-t lg:border-l lg:border-t-0")}>
            {planEl || <Loading label="Generating floor plan…" />}
          </div>
        )}
      </div>
      <div className="pointer-events-none absolute inset-x-3 top-3 flex flex-wrap items-start gap-2">
        {topLeft && <div className="pointer-events-auto">{topLeft}</div>}
        <div className="pointer-events-auto"><Seg value={view} onChange={setView} options={[{ v: "3d", label: "3D", icon: <Box className="size-3.5" /> }, { v: "split", label: "Split" }, { v: "plan", label: "Plan", icon: <MapIcon className="size-3.5" /> }]} /></div>
        {view !== "plan" && (
          <div className="pointer-events-auto"><Seg value={tour ? ("tour" as ViewMode) : mode} onChange={(v) => { setTour(null); setMode(v as ViewMode); }}
            options={[{ v: "orbit", label: "Orbit", icon: <Rotate3d className="size-3.5" /> }, { v: "walk", label: "Walk", icon: <Footprints className="size-3.5" /> }]} /></div>
        )}
        {view !== "plan" && (
          <button className="glass pointer-events-auto flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-medium" onClick={() => setTour(tour ? null : { path: route && route.length > 1 ? route : path, nonce: Date.now() })}>
            {tour ? <><X className="size-3.5" /> Stop</> : <><Play className="size-3.5" /> {route && route.length > 1 ? "Fly the route" : "Replay walk"}</>}
          </button>
        )}
        <Link href={`/explorer?id=${rid}`} className="glass pointer-events-auto ml-auto flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-medium" title="Tag places, upload a floor plan…">
          <ExternalLink className="size-3.5" /> {title ? `${title} in ` : ""}3D Explorer
        </Link>
      </div>
      {mode === "walk" && view !== "plan" && !tour && (
        <div className="absolute bottom-3 left-3 flex items-end gap-3">
          <div className="glass grid grid-cols-3 gap-1 rounded-2xl p-2">
            <span /><button className="hud-btn" onPointerDown={() => press("forward", 1)} onPointerUp={() => press("forward", 0)} onPointerLeave={() => press("forward", 0)} aria-label="Walk forward"><ArrowUp className="size-4" /></button><span />
            <button className="hud-btn" onPointerDown={() => press("turn", 1)} onPointerUp={() => press("turn", 0)} onPointerLeave={() => press("turn", 0)} aria-label="Turn left"><ArrowLeft className="size-4" /></button>
            <button className="hud-btn" onPointerDown={() => press("forward", -1)} onPointerUp={() => press("forward", 0)} onPointerLeave={() => press("forward", 0)} aria-label="Walk back"><ArrowDown className="size-4" /></button>
            <button className="hud-btn" onPointerDown={() => press("turn", -1)} onPointerUp={() => press("turn", 0)} onPointerLeave={() => press("turn", 0)} aria-label="Turn right"><ArrowRight className="size-4" /></button>
          </div>
          <div className="glass hidden rounded-2xl px-3 py-2 text-[11px] leading-relaxed text-foreground/80 sm:block">
            <b>W A S D</b> move · <b>← →</b> / <b>Q E</b> turn · drag to look · <b>Shift</b> run
          </div>
        </div>
      )}
      {view === "3d" && plan?.generated && (mode === "walk" || tour) && (
        <div className="glass absolute bottom-3 right-3 h-44 w-72 overflow-hidden rounded-2xl p-1">
          <PlanView plan={plan} path={path} places={pins} camera={liveCam} selectedPlace={sel} route={route} />
        </div>
      )}
    </div>
  );
}
