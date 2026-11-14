"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  Accessibility, ArrowUp, Box, Camera, CornerUpLeft, CornerUpRight, Eye, Flag, Footprints, Loader2, Map as MapIcon,
  MapPin, Mic, Play, Search, Square, Undo2, Volume2, X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/hooks/use-api";
import { useScene } from "@/hooks/use-scene";
import { LANGUAGES, useRecorder } from "@/hooks/use-recorder";
import { apiSend, apiUpload } from "@/lib/api";
import { speak, type DirectionStep, type Graph, type PlanInfo, type Route } from "@/lib/places";
import { cn } from "@/lib/utils";
import { Empty, ErrorState, Loading, StatusBadge } from "@/components/app/common";
import { PLACE_COLORS, type CameraPose, type PlacePin, type Tour } from "@/components/three/scene-viewer";

const SceneViewer = dynamic(() => import("@/components/three/scene-viewer"), { ssr: false, loading: () => <Loading label="Starting 3D viewer…" /> });
const PlanView = dynamic(() => import("@/components/three/plan-view"), { ssr: false });

type Transcript = { text: string; language: string | null; intent: { destination_class: string; keyword: string } | null; ms: number };
type Loc = { best_node: string; estimated_position_local: number[]; confidence: string; matches: { frame: string; similarity: number }[]; method: string };
type Ask = { interpreted: { start: string; goal: string; start_name: string | null; goal_name: string }; route: Route };

const PROFILES = [
  { k: "standard", label: "Walking", icon: Footprints, hint: "Shortest route" },
  { k: "wheelchair", label: "Wheelchair", icon: Accessibility, hint: "Step-free only" },
  { k: "low_vision", label: "Low vision", icon: Eye, hint: "Handrails, fewer obstacles" },
];
const WALK_SPEED: Record<string, number> = { standard: 1.3, wheelchair: 0.9, low_vision: 0.8 };   // m/s

function StepIcon({ s }: { s: DirectionStep }) {
  const cls = "size-4";
  if (s.kind === "start") return <Flag className={cls} />;
  if (s.kind === "arrive") return <MapPin className={cls} />;
  if (s.kind === "turn") {
    if (s.text.startsWith("Turn around")) return <Undo2 className={cls} />;
    return (s.angle ?? 0) > 0 ? <CornerUpRight className={cls} /> : <CornerUpLeft className={cls} />;
  }
  return <ArrowUp className={cls} />;
}

function Wayfinding() {
  const sp = useSearchParams();
  const recons = useApi<{ reconstructions: { id: string; status: string }[] }>("/spatial/reconstructions");
  const [chosen, setRid] = useState<string | null>(sp.get("id"));
  const rid = chosen ?? recons.data?.reconstructions.find((x) => x.status.startsWith("completed"))?.id ?? null;
  const graph = useApi<Graph>(rid ? `/navigation/${rid}/graph` : null);
  const planApi = useApi<PlanInfo>(rid && graph.data ? `/navigation/${rid}/plan` : null);
  const scene = useScene(rid);
  const [profile, setProfile] = useState("wheelchair");
  const [start, setStart] = useState<string>("");
  const [query, setQuery] = useState(sp.get("to") ? `Take me to ${sp.get("to")}` : "");
  const [route, setRoute] = useState<Route | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [lang, setLang] = useState("");
  const [heard, setHeard] = useState<Transcript | null>(null);
  const [loc, setLoc] = useState<Loc | null>(null);
  const [view, setView] = useState<"3d" | "plan" | "split">("split");
  const [tour, setTour] = useState<Tour>(null);
  const [cam, setCam] = useState<CameraPose | null>(null);
  const rec = useRecorder();
  const fileRef = useRef<HTMLInputElement>(null);
  const autoAsked = useRef(false);

  const g = graph.data;
  const places = useMemo(() => g?.places ?? [], [g]);
  const defaultStart = places.find((p) => p.is_start)?.id ?? "cam0";
  const from = start || defaultStart;
  const camNodes = useMemo(() => (g?.nodes || []).filter((n) => n.kind === "trajectory"), [g]);
  const classes = useMemo(() => [...new Set((g?.nodes || []).filter((n) => n.kind === "facility").map((n) => n.class_name!))].sort(), [g]);
  const nodeById = useMemo(() => new Map((g?.nodes || []).map((n) => [n.id, n])), [g]);
  const pins: PlacePin[] = useMemo(() => places.map((p) => ({ id: p.id, name: p.name, kind: p.kind, pos: p.position })), [places]);
  const path = useMemo(() => scene.scene?.trajectory.map((c) => c.center_local) ?? [], [scene.scene]);
  const onCamera = useCallback((p: CameraPose) => setCam(p), []);

  const ask = useCallback(async (text: string, p = profile, st = from) => {
    if (!rid || !text.trim()) return;
    setBusy("route"); setErr(null); setTour(null);
    try {
      const r = await apiSend<Ask>(`/navigation/${rid}/ask`, { text, profile: p, start: st });
      setRoute(r.route);
      if (r.interpreted.start !== st) setStart(r.interpreted.start);
    } catch (e) { setErr((e as Error).message); setRoute(null); } finally { setBusy(null); }
  }, [rid, profile, from]);

  const routeTo = useCallback(async (goal: string, p = profile, st = from) => {
    if (!rid) return;
    setBusy("route"); setErr(null); setTour(null);
    try { setRoute(await apiSend<Route>(`/navigation/${rid}/route`, { start: st, goal, profile: p })); }
    catch (e) { setErr((e as Error).message); setRoute(null); } finally { setBusy(null); }
  }, [rid, profile, from]);

  useEffect(() => {   // deep link from the Explorer: /navigation?id=..&to=CG%20Lab
    if (!autoAsked.current && g && sp.get("to")) { autoAsked.current = true; ask(`Take me to ${sp.get("to")}`); }
  }, [g, sp, ask]);

  function changeProfile(k: string) {
    setProfile(k);
    if (route?.goal) routeTo(route.goal, k);
  }

  function voice() {
    if (rec.recording) return rec.stop();
    rec.start(async (blob) => {
      setBusy("voice"); setErr(null);
      const fd = new FormData();
      fd.append("file", blob, "speech.webm");
      if (lang) fd.append("language", lang);
      try {
        const t = await apiUpload<Transcript>("/ai/transcribe", fd);
        setHeard(t);
        setQuery(t.text);
        await ask(t.text);
      } catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
    });
  }

  async function locate(f: File) {
    if (!rid) return;
    setBusy("locate"); setErr(null);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const l = await apiUpload<Loc>(`/ai/localize/${rid}`, fd);
      setLoc(l); setStart(l.best_node); setRoute(null);
    } catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }

  const startNode = nodeById.get(from);
  const markers = startNode && !places.some((p) => p.id === from) ? [{ pos: startNode.pos, label: "You are here", color: "#16a34a" }] : [];
  const metric = route?.units === "m";
  const minutes = metric && route?.length ? Math.max(1, Math.round(route.length / WALK_SPEED[route.profile] / 60)) : null;
  const scale = scene.recon?.alignment?.scale_status;
  const walkPath = route?.feasible ? route.path : null;

  const viewer = scene.scene && (
    <SceneViewer data={scene.scene} layers={{ splat: true, points: false, trajectory: true, facilities: true }} route={walkPath}
      highlight={route?.goal} markers={markers} places={pins} selectedPlace={route?.goal}
      onPlaceClick={(id) => { const p = places.find((x) => x.id === id); if (p) { setQuery(`Take me to ${p.name}`); routeTo(p.id); } }}
      mode="orbit" tour={tour} onTourEnd={() => setTour(null)} onCamera={onCamera} />
  );
  const planEl = scene.scene && planApi.data?.generated && (
    <PlanView plan={planApi.data} path={path} places={pins} route={walkPath} camera={tour ? cam : null} selectedPlace={route?.goal}
      onPlaceClick={(id) => { const p = places.find((x) => x.id === id); if (p) { setQuery(`Take me to ${p.name}`); routeTo(p.id); } }} />
  );

  return (
    <div className="space-y-5">
      {/* hero search */}
      <div className="hero-gradient relative overflow-hidden rounded-[2rem] px-6 py-7 text-white shadow-xl md:px-10">
        <div className="pointer-events-none absolute -right-10 -top-10 size-64 rounded-full border border-white/10" />
        <div className="pointer-events-none absolute -right-24 -top-24 size-96 rounded-full border border-white/5" />
        <div className="relative max-w-3xl">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-violet-200">Accessible wayfinding</div>
          <h1 className="font-display mt-1 text-3xl font-bold tracking-tight md:text-4xl">Where do you want to go?</h1>
          <p className="mt-1 text-sm text-white/70">Ask in your own words. Routes follow the walked space in the 3D scan and adapt to your needs.</p>
          <form className="mt-5 flex items-center gap-2 rounded-2xl bg-white p-1.5 text-foreground shadow-lg" onSubmit={(e) => { e.preventDefault(); ask(query); }}>
            <Search className="ml-2 size-5 shrink-0 text-muted-foreground" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. Take me to the CG Lab  ·  From CC Lab to CG Lab  ·  Nearest lift"
              className="min-w-0 flex-1 bg-transparent px-1 py-2 text-[15px] outline-none" aria-label="Where do you want to go" />
            <select value={lang} onChange={(e) => setLang(e.target.value)} className="hidden h-9 rounded-lg border-0 bg-muted px-2 text-xs sm:block" title="Spoken language hint">
              {LANGUAGES.map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <button type="button" onClick={voice} disabled={busy === "voice"} title="Say your destination"
              className={cn("grid size-10 place-items-center rounded-xl", rec.recording ? "animate-pulse bg-rose-500 text-white" : "bg-muted hover:bg-accent")}>
              {rec.recording ? <Square className="size-4" /> : busy === "voice" ? <Loader2 className="size-4 animate-spin" /> : <Mic className="size-4" />}
            </button>
            <Button type="submit" className="h-10 rounded-xl px-5" disabled={!query.trim() || busy === "route"}>{busy === "route" ? <Loader2 className="animate-spin" /> : "Go"}</Button>
          </form>
          {places.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {places.filter((p) => p.id !== from).slice(0, 8).map((p) => (
                <button key={p.id} onClick={() => { setQuery(`Take me to ${p.name}`); routeTo(p.id); }}
                  className="flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1 text-xs font-medium ring-1 ring-white/20 backdrop-blur hover:bg-white/20">
                  <span className="size-2 rounded-full" style={{ background: PLACE_COLORS[p.kind] }} />{p.name}
                </button>
              ))}
              {classes.filter((c) => ["elevator", "ramp", "toilet_sign", "entrance_exit"].includes(c)).map((c) => (
                <button key={c} onClick={() => { setQuery(`Nearest ${c.replaceAll("_", " ")}`); routeTo(c); }} className="rounded-full bg-white/5 px-3 py-1 text-xs ring-1 ring-white/15 hover:bg-white/15">Nearest {c.replaceAll("_", " ")}</button>
              ))}
            </div>
          )}
          {rec.error && <div className="mt-2 text-xs text-rose-200">{rec.error}</div>}
          {heard && <div className="mt-2 text-xs text-white/70">Heard ({heard.language || "?"}): &ldquo;{heard.text}&rdquo;</div>}
        </div>
      </div>

      {recons.error && <ErrorState message={recons.error} onRetry={recons.reload} />}
      {recons.data && !recons.data.reconstructions.length && <Empty title="No reconstructions">Build one from a walkthrough video in AI Studio first.</Empty>}
      {graph.error && <ErrorState message={graph.error} />}
      {graph.loading && <Loading label="Building navigation graph…" />}
      {g && (
        <div className="grid gap-5 xl:grid-cols-[400px_1fr]">
          <div className="space-y-4">
            <div className="surface space-y-4 p-5">
              <div>
                <div className="eyebrow mb-2">I am travelling as</div>
                <div className="grid grid-cols-3 gap-2">
                  {PROFILES.filter((p) => g.profiles[p.k]).map((p) => (
                    <button key={p.k} onClick={() => changeProfile(p.k)} aria-pressed={profile === p.k}
                      className={cn("flex flex-col items-center gap-1 rounded-2xl border p-3 text-center transition", profile === p.k ? "border-primary bg-accent text-accent-foreground ring-2 ring-primary/20" : "hover:bg-muted")}>
                      <p.icon className="size-5" /><span className="text-xs font-semibold">{p.label}</span><span className="text-[10px] leading-tight text-muted-foreground">{p.hint}</span>
                    </button>
                  ))}
                </div>
              </div>
              <label className="block">
                <span className="eyebrow">Starting from</span>
                <select value={from} onChange={(e) => { setStart(e.target.value); setLoc(null); setRoute(null); }} className="field mt-2">
                  {places.length > 0 && <optgroup label="Tagged places">{places.map((p) => <option key={p.id} value={p.id}>{p.name}{p.is_start ? " (default start)" : ""}</option>)}</optgroup>}
                  <optgroup label="Points along the scan">{camNodes.map((n, i) => (i % 5 === 0 || n.id === from) && <option key={n.id} value={n.id}>{i === 0 ? "Start of the walk" : `Along the walk · ${Math.round((100 * i) / Math.max(1, camNodes.length - 1))}%`}</option>)}</optgroup>
                </select>
              </label>
              <Button variant="outline" size="sm" className="w-full rounded-full" disabled={!!busy} onClick={() => fileRef.current?.click()}>
                {busy === "locate" ? <Loader2 className="animate-spin" /> : <Camera />} Locate me from a photo
              </Button>
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && locate(e.target.files[0])} />
              {loc && (
                <div className="rounded-xl bg-muted p-2 text-xs">
                  Matched keyframe {loc.matches[0].frame} (similarity {loc.matches[0].similarity}) · <StatusBadge status={loc.confidence} />
                  {loc.confidence === "low" && <div className="mt-1 text-amber-700">Low confidence: check the start point.</div>}
                </div>
              )}
              {!places.length && (
                <div className="rounded-2xl border border-dashed border-violet-300 bg-violet-50/50 p-3 text-xs">
                  No rooms tagged in this scan yet. <Link href={`/explorer?id=${rid}`} className="font-semibold text-primary underline">Tag CC Lab, CG Lab…</Link> in the 3D Explorer to ask for them by name.
                </div>
              )}
              <select value={rid || ""} onChange={(e) => { setRid(e.target.value); setRoute(null); setStart(""); }} className="field text-xs text-muted-foreground" aria-label="Reconstruction">
                {recons.data?.reconstructions.map((r) => <option key={r.id} value={r.id}>Scan {r.id} · {r.status}</option>)}
              </select>
            </div>
            {err && <ErrorState message={err} />}
            {route && (
              <div className="surface overflow-hidden p-0">
                <div className={cn("p-5", route.feasible ? "bg-gradient-to-br from-violet-50 to-teal-50" : "bg-rose-50")}>
                  {route.feasible ? (
                    <>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground"><span className="truncate">{route.start_name}</span><span>→</span><span className="truncate font-semibold text-foreground">{route.goal_name}</span></div>
                      <div className="mt-1 flex items-end gap-3">
                        <div className="font-display text-3xl font-bold">{minutes ? `${minutes} min` : `${route.length?.toFixed(1)} ${route.units}`}</div>
                        <div className="pb-1 text-sm text-muted-foreground">{minutes ? `${Math.round(route.length!)} m · ` : ""}{route.profile_label}</div>
                      </div>
                      {route.warnings && route.warnings.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">{route.warnings.map((w) => <span key={w} className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-800">{w.replaceAll("_", " ")}</span>)}</div>
                      )}
                      <div className="mt-3 flex gap-2">
                        <Button size="sm" className="rounded-full" onClick={() => { setView((v) => (v === "plan" ? "split" : v)); setTour({ path: route.path, nonce: Date.now() }); }}><Play /> Walk me there</Button>
                        <Button size="sm" variant="outline" className="rounded-full bg-white" onClick={() => speak(route.spoken || route.steps.map((s) => s.instruction).join(". "))}><Volume2 /> Read aloud</Button>
                        {tour && <Button size="sm" variant="ghost" className="rounded-full" onClick={() => setTour(null)}><X /> Stop</Button>}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="font-semibold text-rose-800">No {route.profile === "wheelchair" ? "step-free " : ""}route found</div>
                      <div className="mt-1 text-sm text-rose-700">{route.reason}</div>
                      {route.profile === "wheelchair" && <div className="mt-2 text-xs text-rose-700">Tip: ask staff about a lift, or tag the lift / ramp in the 3D Explorer if the scan missed it.</div>}
                    </>
                  )}
                </div>
                {route.directions && (
                  <ol className="relative space-y-0 p-5">
                    {route.directions.map((s, i) => (
                      <li key={i} className="relative flex gap-3 pb-4 last:pb-0">
                        {i < route.directions!.length - 1 && <span className="absolute left-[15px] top-8 h-[calc(100%-24px)] w-0.5 bg-border" />}
                        <span className={cn("relative z-10 grid size-8 shrink-0 place-items-center rounded-full",
                          s.kind === "start" ? "bg-emerald-500 text-white" : s.kind === "arrive" ? "bg-rose-500 text-white" : s.kind === "turn" ? "bg-violet-600 text-white" : "bg-muted text-foreground")}><StepIcon s={s} /></span>
                        <div className="pt-1 text-sm">
                          <div className={cn(s.kind === "turn" && "font-semibold")}>{s.text}</div>
                          {s.notes?.map((n, j) => <div key={j} className="mt-1 rounded-lg bg-amber-50 px-2 py-1 text-xs text-amber-900">{n}</div>)}
                        </div>
                      </li>
                    ))}
                  </ol>
                )}
                {route.units === "units" && <div className="px-5 pb-4 text-xs text-muted-foreground">This scan has no metric scale, so distances are descriptive rather than in metres.</div>}
              </div>
            )}
          </div>
          <div className="surface relative h-[calc(100vh-150px)] min-h-[560px] overflow-hidden p-0">
            <div className={cn("grid h-full", view === "split" ? "grid-rows-2" : "grid-cols-1")}>
              {view !== "plan" && <div className="relative min-h-0">{viewer || (scene.error ? <ErrorState message={scene.error} /> : <Loading label="Loading 3D scan…" />)}</div>}
              {view !== "3d" && <div className={cn("relative min-h-0 bg-slate-50", view === "split" && "border-t")}>{planEl || <Loading label="Preparing floor plan…" />}</div>}
            </div>
            <div className="glass absolute left-3 top-3 inline-flex rounded-full p-1">
              {([["3d", "3D", Box], ["split", "Split", null], ["plan", "Plan", MapIcon]] as const).map(([k, l, I]) => (
                <button key={k} onClick={() => setView(k)} className={cn("flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium", view === k ? "bg-primary text-primary-foreground shadow" : "text-foreground/70")}>{I && <I className="size-3.5" />}{l}</button>
              ))}
            </div>
            <div className="glass absolute right-3 top-3 rounded-full px-3 py-1.5 text-[11px] text-foreground/70">{scale === "relative" ? "relative scale" : scale ? scale.replaceAll("_", " ") : ""}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function NavigationPage() {
  return <Suspense fallback={<Loading />}><Wayfinding /></Suspense>;
}
