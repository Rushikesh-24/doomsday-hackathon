"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Suspense, useCallback, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Box, Crosshair, Download, Eye, Footprints, Layers, Loader2, Map as MapIcon,
  Gauge, MapPin, Move, Navigation, Pencil, Play, Plus, RefreshCw, Rotate3d, Star, Trash2, Upload, X,
} from "lucide-react";
import { AccessibilityPanel, SEVERITY, type Accessibility, type Issue } from "@/components/app/accessibility-panel";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { useApi } from "@/hooks/use-api";
import { useScene } from "@/hooks/use-scene";
import { apiSend, apiUpload, apiUrl } from "@/lib/api";
import { PLACE_KINDS, type Place, type PlanInfo, type PlanTransform } from "@/lib/places";
import { cn } from "@/lib/utils";
import { CLASS_COLORS, Empty, ErrorState, Loading, StatusBadge, fmt } from "@/components/app/common";
import type { CameraPose, PlacePin, Tour, ViewMode, WalkInput } from "@/components/three/scene-viewer";
import { PLACE_COLORS } from "@/components/three/scene-viewer";

const SceneViewer = dynamic(() => import("@/components/three/scene-viewer"), { ssr: false, loading: () => <Loading label="Starting 3D viewer…" /> });
const PlanView = dynamic(() => import("@/components/three/plan-view"), { ssr: false });

type ReconRow = { id: string; video_id: string; status: string; created: string; scale_status: string | null; gaussians: number | null; registered_frames: number | null };
type Draft = { id?: string; name: string; kind: string; position: number[]; aliases: string; description: string; step_free: boolean | null; notes: string; is_start: boolean };
type Panel = "access" | "places" | "plan" | "facilities" | "details";

const emptyDraft = (pos: number[]): Draft => ({ name: "", kind: "lab", position: pos, aliases: "", description: "", step_free: null, notes: "", is_start: false });

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

function Explorer() {
  const sp = useSearchParams();
  const list = useApi<{ reconstructions: ReconRow[] }>("/spatial/reconstructions");
  const [chosen, setRid] = useState<string | null>(sp.get("id"));
  const rows = list.data?.reconstructions || [];
  const rid = chosen ?? (rows.find((r) => r.status.startsWith("completed")) || rows[0])?.id ?? null;
  const s = useScene(rid);
  const placesApi = useApi<{ places: Place[] }>(rid && s.scene ? `/navigation/${rid}/places` : null);
  const planApi = useApi<PlanInfo>(rid && s.scene ? `/navigation/${rid}/plan` : null);
  const access = useApi<Accessibility>(rid && s.scene ? `/spatial/reconstructions/${rid}/accessibility` : null);
  const [selIssue, setSelIssue] = useState<string | null>(null);
  const [showIssues, setShowIssues] = useState(true);
  const [layers, setLayers] = useState({ splat: true, points: false, trajectory: true, facilities: true });
  const [showLayers, setShowLayers] = useState(false);
  const [view, setView] = useState<"3d" | "split" | "plan">("3d");
  const [mode, setMode] = useState<ViewMode>("orbit");
  const [tour, setTour] = useState<Tour>(null);
  const [panel, setPanel] = useState<Panel>((sp.get("panel") as Panel) || "access");
  const [pick, setPick] = useState<string | null>(null);
  const [selPlace, setSelPlace] = useState<string | null>(null);
  const [placing, setPlacing] = useState(false);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ pos: number[]; nonce: number } | null>(null);
  const [walkStart, setWalkStart] = useState<{ pos: number[]; yaw: number; nonce: number } | null>(null);
  const [cam, setCam] = useState<CameraPose | null>(null);
  const [align, setAlign] = useState(false);
  const [trDraft, setTrDraft] = useState<{ url: string; t: PlanTransform } | null>(null);
  const [planLayers, setPlanLayers] = useState({ generated: true, uploaded: true });
  const walkInput = useRef<WalkInput>({ forward: 0, turn: 0 });
  const fileRef = useRef<HTMLInputElement>(null);

  const places = useMemo(() => placesApi.data?.places ?? [], [placesApi.data]);
  const pins: PlacePin[] = useMemo(() => [
    ...places.map((p) => ({ id: p.id, name: p.name, kind: p.kind, pos: p.position })),
    ...(draft && !draft.id ? [{ id: "__draft", name: draft.name || "New place", kind: draft.kind, pos: draft.position, color: "#a855f7" }] : []),
  ], [places, draft]);
  const issues = useMemo(() => (showIssues ? access.data?.issues ?? [] : []), [access.data, showIssues]);
  const issuePins: PlacePin[] = useMemo(() => issues.map((i) => ({ id: `issue:${i.id}`, name: `⚠ ${i.title}`, kind: "other", pos: i.position, color: SEVERITY[i.severity].color })), [issues]);
  const corridor = access.data?.corridor ?? null;
  const path = useMemo(() => s.scene?.trajectory.map((c) => c.center_local) ?? [], [s.scene]);
  const facPts = useMemo(() => (s.facilities || []).filter((f) => f.position_local).map((f) => ({ id: f.facility_id, class_name: f.class_name, pos: f.position_local as number[] })), [s.facilities]);
  const m = s.recon?.metadata;
  const picked = s.facilities?.find((f) => f.facility_id === (pick ?? sp.get("facility")));
  const plan = planApi.data;
  const tr = trDraft && plan?.uploaded && trDraft.url === plan.uploaded.url ? trDraft.t : null;   // unsaved edits of this plan
  const setTr = (t: PlanTransform | null) => setTrDraft(t && plan?.uploaded ? { url: plan.uploaded.url, t } : null);
  const transform = tr ?? plan?.uploaded?.transform ?? null;
  const liveCam = mode === "walk" || tour ? cam : null;
  const planForView = plan && transform && plan.uploaded ? { ...plan, uploaded: { ...plan.uploaded, transform } } : plan;
  const onCamera = useCallback((p: CameraPose) => setCam(p), []);
  // ?facility=<id> (from the Facilities review page): select it and fly to it once the scene has loaded
  const focusFac = sp.get("facility");
  const facHit = focusFac ? s.facilities?.find((x) => x.facility_id === focusFac) : undefined;
  const shownPick = pick ?? facHit?.facility_id ?? null;
  const facFocus = useMemo(() => (facHit?.position_local ? { pos: facHit.position_local as number[], nonce: 1 } : null), [facHit]);


  function startPlacing() {
    setDraft(null); setPlacing(true); setPanel("places"); setErr(null);
  }
  function placeAt(pos: number[]) {
    setPlacing(false);
    setDraft(emptyDraft(pos.map((v) => +v.toFixed(4))));
  }
  function tagHere() {
    if (!cam) return;
    setDraft(emptyDraft(cam.pos.map((v) => +v.toFixed(4))));
    setPanel("places");
  }
  function edit(p: Place) {
    setDraft({ id: p.id, name: p.name, kind: p.kind, position: p.position, aliases: p.aliases.join(", "), description: p.description || "",
      step_free: p.access?.step_free ?? null, notes: p.access?.notes || "", is_start: p.is_start });
    setSelPlace(p.id);
  }
  async function save() {
    if (!rid || !draft) return;
    setBusy("save"); setErr(null);
    const body = { name: draft.name, kind: draft.kind, position: draft.position, aliases: draft.aliases, description: draft.description || null,
      access: { step_free: draft.step_free, accessible_toilet: null, notes: draft.notes || null }, is_start: draft.is_start };
    try {
      const p = draft.id ? await apiSend<Place>(`/navigation/${rid}/places/${draft.id}`, body, "PATCH") : await apiSend<Place>(`/navigation/${rid}/places`, body);
      setDraft(null); setSelPlace(p.id); placesApi.reload();
    } catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }
  async function remove(p: Place) {
    if (!rid || !confirm(`Delete "${p.name}"? Routes to it will stop working.`)) return;
    try { await apiSend(`/navigation/${rid}/places/${p.id}`, undefined, "DELETE"); placesApi.reload(); if (selPlace === p.id) setSelPlace(null); }
    catch (e) { setErr((e as Error).message); }
  }
  function showIssue(i: Issue) {
    setSelIssue(i.id); setShowIssues(true);
    if (view === "plan") { setPanel("access"); return; }
    goTo(i.position);
  }
  function onPinClick(id: string) {
    if (id.startsWith("issue:")) { const i = issues.find((x) => `issue:${x.id}` === id); setPanel("access"); if (i) showIssue(i); return; }
    setSelPlace(id); setPanel("places");
  }
  function goTo(pos: number[], id?: string) {
    if (id) setSelPlace(id);
    if (mode === "walk") setWalkStart({ pos, yaw: cam?.yaw ?? 0, nonce: Date.now() });
    else setFocus({ pos, nonce: Date.now() });
  }
  async function uploadPlan(f: File) {
    if (!rid) return;
    setBusy("upload"); setErr(null);
    const fd = new FormData(); fd.append("file", f);
    try { planApi.setData(await apiUpload<PlanInfo>(`/navigation/${rid}/plan/upload`, fd)); setAlign(true); setView((v) => (v === "3d" ? "split" : v)); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }
  async function planCall(path: string, body?: unknown, method = "POST") {
    if (!rid) return;
    setBusy(path); setErr(null);
    try { planApi.setData(await apiSend<PlanInfo>(`/navigation/${rid}/plan${path}`, body, method)); setTr(null); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }
  const press = useCallback((k: keyof WalkInput, v: number) => { walkInput.current[k] = v; }, []);

  const viewer = s.scene && (
    <SceneViewer data={s.scene} layers={layers} highlight={shownPick} onPick={setPick} places={[...pins, ...issuePins]} selectedPlace={selIssue ? `issue:${selIssue}` : selPlace}
      onPlaceClick={onPinClick} mode={mode} placing={placing} onPlace={placeAt}
      tour={tour} onTourEnd={() => setTour(null)} focus={focus ?? facFocus} onCamera={onCamera} walkInput={walkInput} walkStart={walkStart} />
  );
  const planEl = s.scene && plan?.generated && (
    <PlanView plan={planForView!} path={path} places={pins} facilities={layers.facilities ? facPts : []} camera={liveCam} selectedPlace={selPlace}
      onPlaceClick={(id) => { setSelPlace(id); setPanel("places"); const p = places.find((x) => x.id === id); if (p) goTo(p.position); }}
      placing={placing} onPick={placeAt} align={align && !placing} onTransform={setTr}
      showGenerated={planLayers.generated} showUploaded={planLayers.uploaded}
      corridor={corridor} issues={issues} selectedIssue={selIssue} onIssueClick={(id) => { setSelIssue(id); setPanel("access"); }} />
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="eyebrow">Digital twin</div>
          <h1 className="font-display text-3xl font-semibold tracking-tight">3D Explorer</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">Walk through the reconstruction, tag rooms like <b>CC Lab</b> or <b>CG Lab</b>, and line it up with your floor plan. Tagged places become navigation destinations.</p>
        </div>
        <div className="flex items-center gap-2">
          <select value={rid || ""} onChange={(e) => { setRid(e.target.value); setPick(null); setSelPlace(null); setDraft(null); setTour(null); }} className="h-9 max-w-[320px] rounded-full border bg-card px-3 text-sm">
            {rows.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status}{r.gaussians ? ` · ${r.gaussians.toLocaleString()} splats` : ""}</option>)}
          </select>
          <Link href="/studio?tab=video"><Button variant="outline" className="rounded-full"><Plus /> New scan</Button></Link>
        </div>
      </div>
      {list.error && <ErrorState message={list.error} onRetry={list.reload} />}
      {list.data && !rows.length && <Empty title="No reconstructions yet">Upload a walkthrough video in AI Studio to build one.</Empty>}
      {s.loading && <Loading label="Loading reconstruction…" />}
      {s.error && <ErrorState message={s.error} />}
      {s.scene && (
        <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
          <div className="surface relative h-[calc(100vh-190px)] min-h-[560px] overflow-hidden p-0">
            <div className={cn("grid h-full", view === "split" ? "grid-rows-2 lg:grid-cols-2 lg:grid-rows-1" : "grid-cols-1")}>
              {view !== "plan" && <div className="relative min-h-0">{viewer}</div>}
              {view !== "3d" && (
                <div className={cn("relative min-h-0 bg-slate-50", view === "split" && "border-t lg:border-l lg:border-t-0")}>
                  {planEl || <Loading label="Generating floor plan…" />}
                </div>
              )}
            </div>
            {/* top toolbar */}
            <div className="pointer-events-none absolute inset-x-3 top-3 flex flex-wrap items-start gap-2">
              <div className="pointer-events-auto"><Seg value={view} onChange={setView} options={[{ v: "3d", label: "3D", icon: <Box className="size-3.5" /> }, { v: "split", label: "Split" }, { v: "plan", label: "Plan", icon: <MapIcon className="size-3.5" /> }]} /></div>
              {view !== "plan" && (
                <div className="pointer-events-auto"><Seg value={tour ? ("tour" as ViewMode) : mode} onChange={(v) => { setTour(null); setMode(v as ViewMode); }}
                  options={[{ v: "orbit", label: "Orbit", icon: <Rotate3d className="size-3.5" /> }, { v: "walk", label: "Walk", icon: <Footprints className="size-3.5" /> }]} /></div>
              )}
              {view !== "plan" && (
                <button className="glass pointer-events-auto flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-medium" onClick={() => setTour(tour ? null : { path, nonce: Date.now() })}>
                  {tour ? <><X className="size-3.5" /> Stop tour</> : <><Play className="size-3.5" /> Replay walk</>}
                </button>
              )}
              <div className="pointer-events-auto relative">
                <button className="glass flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-medium" onClick={() => setShowLayers(!showLayers)}><Layers className="size-3.5" /> Layers</button>
                {showLayers && (
                  <div className="glass absolute left-0 top-10 z-10 w-52 space-y-2 rounded-2xl p-3 text-xs">
                    {(["splat", "points", "trajectory", "facilities"] as const).map((k) => (
                      <label key={k} className="flex items-center justify-between capitalize">{k === "points" ? "point cloud" : k === "trajectory" ? "walked path" : k}
                        <Switch checked={layers[k]} onCheckedChange={(v: boolean) => setLayers({ ...layers, [k]: v })} /></label>
                    ))}
                    <label className="flex items-center justify-between">problem spots<Switch checked={showIssues} onCheckedChange={(v: boolean) => setShowIssues(v)} /></label>
                    {plan?.uploaded && (["generated", "uploaded"] as const).map((k) => (
                      <label key={k} className="flex items-center justify-between">{k === "generated" ? "AI plan" : "Your plan"}
                        <Switch checked={planLayers[k]} onCheckedChange={(v: boolean) => setPlanLayers({ ...planLayers, [k]: v })} /></label>
                    ))}
                  </div>
                )}
              </div>
              <div className="ml-auto flex gap-2">
                {mode === "walk" && view !== "plan" && !tour && (
                  <button className="glass pointer-events-auto flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-semibold text-violet-700" onClick={tagHere}><Crosshair className="size-3.5" /> Tag where I&apos;m standing</button>
                )}
                <button onClick={() => (placing ? setPlacing(false) : startPlacing())}
                  className={cn("pointer-events-auto flex items-center gap-1.5 rounded-full px-4 py-2 text-xs font-semibold shadow-lg transition", placing ? "bg-violet-600 text-white" : "bg-primary text-primary-foreground")}>
                  {placing ? <><X className="size-3.5" /> Cancel tagging</> : <><MapPin className="size-3.5" /> Tag a place</>}
                </button>
              </div>
            </div>
            {placing && (
              <div className="pointer-events-none absolute inset-x-0 top-16 flex justify-center">
                <div className="rounded-full bg-violet-600 px-4 py-2 text-xs font-medium text-white shadow-lg">Click on the walked area in 3D or on the plan to drop a pin</div>
              </div>
            )}
            {/* walk HUD */}
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
                <PlanView plan={planForView!} path={path} places={pins} camera={liveCam} selectedPlace={selPlace} corridor={corridor} issues={issues} />
              </div>
            )}
            <div className="glass absolute bottom-3 left-1/2 hidden -translate-x-1/2 rounded-full px-3 py-1 text-[11px] text-foreground/70 lg:block" style={{ display: mode === "walk" && view !== "plan" ? "none" : undefined }}>
              {s.recon?.alignment?.scale_status === "relative" ? "Relative units (no metric scale in this video)" : `Metric · ${s.recon?.alignment?.scale_status?.replaceAll("_", " ")}`}
            </div>
          </div>

          {/* side panel */}
          <div className="surface flex max-h-[calc(100vh-190px)] min-h-[560px] flex-col overflow-hidden p-0">
            <div className="flex gap-1 border-b p-2">
              {([["access", access.data ? `Score ${access.data.score}` : "Score"], ["places", `Places (${places.length})`], ["plan", "Plan"], ["facilities", "Facilities"], ["details", "Details"]] as [Panel, string][]).map(([k, l]) => (
                <button key={k} onClick={() => setPanel(k)} className={cn("flex flex-1 items-center justify-center gap-1 rounded-lg px-1.5 py-1.5 text-xs font-medium", panel === k ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-muted")}>
                  {k === "access" && <Gauge className="size-3.5" />}{l}</button>
              ))}
            </div>
            <div className="flex-1 space-y-3 overflow-auto p-4 text-sm">
              {err && <ErrorState message={err} />}
              {panel === "access" && (
                <>
                  {access.loading && <Loading label="Scoring accessibility…" />}
                  {access.error && <ErrorState message={access.error} onRetry={access.reload} />}
                  {access.data && <AccessibilityPanel data={access.data} selected={selIssue} onIssue={showIssue} />}
                </>
              )}
              {panel === "places" && (
                <>
                  {draft && (
                    <div className="rounded-2xl border border-violet-200 bg-violet-50/60 p-3">
                      <div className="mb-2 flex items-center justify-between">
                        <div className="text-sm font-semibold">{draft.id ? "Edit place" : "New place"}</div>
                        <button onClick={() => setDraft(null)} className="text-muted-foreground hover:text-foreground"><X className="size-4" /></button>
                      </div>
                      <div className="space-y-2">
                        <input autoFocus value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="Name, e.g. CG Lab" className="field" />
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(PLACE_KINDS).map(([k, v]) => (
                            <button key={k} onClick={() => setDraft({ ...draft, kind: k })} className={cn("rounded-full border px-2 py-0.5 text-[11px]", draft.kind === k ? "border-transparent text-white" : "bg-white hover:bg-muted")}
                              style={draft.kind === k ? { background: PLACE_COLORS[k] } : undefined}>{v}</button>
                          ))}
                        </div>
                        {(draft.kind === "stairs" || draft.kind === "ramp") && (
                          <div className="rounded-lg bg-amber-50 px-2 py-1 text-[11px] text-amber-900">
                            {draft.kind === "stairs" ? "Routes through this spot (about 2 m around it) count as stairs: wheelchair routes avoid it." : "Routes through this spot get a ramp warning."}
                          </div>
                        )}
                        <input value={draft.aliases} onChange={(e) => setDraft({ ...draft, aliases: e.target.value })} placeholder="Also called (comma separated), e.g. graphics lab" className="field" />
                        <textarea value={draft.description} onChange={(e) => setDraft({ ...draft, description: e.target.value })} placeholder="Directions hint shown on arrival, e.g. Second door on the left, room 204" rows={2} className="field" />
                        <div className="flex items-center gap-2 text-xs">
                          <span className="text-muted-foreground">Step-free access</span>
                          {([[true, "Yes"], [false, "No"], [null, "Unknown"]] as [boolean | null, string][]).map(([v, l]) => (
                            <button key={l} onClick={() => setDraft({ ...draft, step_free: v })} className={cn("rounded-full border px-2 py-0.5", draft.step_free === v ? "bg-foreground text-background" : "bg-white")}>{l}</button>
                          ))}
                        </div>
                        <input value={draft.notes} onChange={(e) => setDraft({ ...draft, notes: e.target.value })} placeholder="Accessibility notes, e.g. heavy door, ask at front desk" className="field" />
                        <label className="flex items-center gap-2 text-xs"><Switch checked={draft.is_start} onCheckedChange={(v: boolean) => setDraft({ ...draft, is_start: v })} /> Default starting point (e.g. main entrance)</label>
                        <div className="font-mono text-[10px] text-muted-foreground">at {draft.position.map((v) => v.toFixed(2)).join(", ")}</div>
                        <div className="flex gap-2">
                          <Button className="flex-1 rounded-full" disabled={!draft.name.trim() || busy === "save"} onClick={save}>{busy === "save" ? <Loader2 className="animate-spin" /> : <MapPin />} Save place</Button>
                          {!draft.id && <Button variant="outline" className="rounded-full" onClick={startPlacing}>Re-pick</Button>}
                        </div>
                      </div>
                    </div>
                  )}
                  {!draft && !places.length && (
                    <div className="rounded-2xl border border-dashed p-5 text-center">
                      <MapPin className="mx-auto mb-2 size-6 text-violet-500" />
                      <div className="font-medium">No places tagged yet</div>
                      <p className="mt-1 text-xs text-muted-foreground">Press <b>Tag a place</b> and click where CC Lab is, or switch to <b>Walk</b>, go there and press <b>Tag where I&apos;m standing</b>.</p>
                    </div>
                  )}
                  <div className="space-y-2">
                    {places.map((p) => (
                      <div key={p.id} onClick={() => { setSelPlace(p.id); goTo(p.position, p.id); }}
                        className={cn("group cursor-pointer rounded-2xl border p-3 transition", selPlace === p.id ? "border-primary/40 bg-accent/60" : "hover:bg-muted/60")}>
                        <div className="flex items-start gap-3">
                          <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-xl text-white" style={{ background: PLACE_COLORS[p.kind] }}><MapPin className="size-4" /></span>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1.5 font-medium">{p.name}{p.is_start && <span className="flex items-center gap-0.5 rounded-full bg-emerald-100 px-1.5 text-[10px] font-semibold text-emerald-700"><Star className="size-2.5" />START</span>}</div>
                            <div className="text-xs text-muted-foreground">{PLACE_KINDS[p.kind]}{p.aliases.length ? ` · ${p.aliases.join(", ")}` : ""}{p.access?.step_free === true ? " · step-free" : p.access?.step_free === false ? " · not step-free" : ""}</div>
                            {p.description && <div className="mt-1 line-clamp-2 text-xs text-foreground/70">{p.description}</div>}
                          </div>
                        </div>
                        <div className="mt-2 flex gap-1 opacity-80 group-hover:opacity-100">
                          <Link href={`/navigation?id=${rid}&to=${encodeURIComponent(p.name)}`} onClick={(e) => e.stopPropagation()} className="chip"><Navigation className="size-3" /> Directions</Link>
                          <button onClick={(e) => { e.stopPropagation(); edit(p); }} className="chip"><Pencil className="size-3" /> Edit</button>
                          <button onClick={(e) => { e.stopPropagation(); remove(p); }} className="chip text-rose-600"><Trash2 className="size-3" /></button>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
              {panel === "plan" && plan?.generated && (
                <>
                  <div className="rounded-2xl bg-muted/60 p-3 text-xs">
                    <div className="font-semibold text-foreground">AI-generated plan</div>
                    <p className="mt-1 text-muted-foreground">Top-down projection of {plan.generated.points_used.toLocaleString()} reconstructed points on the walked storey. Teal band = walked corridor.</p>
                    <div className="mt-2 flex gap-2">
                      <Button size="sm" variant="outline" className="rounded-full" disabled={!!busy} onClick={() => planCall("/generate")}>{busy === "/generate" ? <Loader2 className="animate-spin" /> : <RefreshCw />} Regenerate</Button>
                      <a href={apiUrl(plan.generated.url)} download={`plan-${rid}.png`}><Button size="sm" variant="outline" className="rounded-full"><Download /> PNG</Button></a>
                      {view === "3d" && <Button size="sm" variant="outline" className="rounded-full" onClick={() => setView("split")}><Eye /> Show</Button>}
                    </div>
                  </div>
                  <div className="rounded-2xl border p-3 text-xs">
                    <div className="font-semibold">Your floor plan</div>
                    {!plan.uploaded ? (
                      <>
                        <p className="mt-1 text-muted-foreground">Upload the building&apos;s plan (PNG/JPG). You can then drag, scale and rotate it until it lines up with the AI plan.</p>
                        <Button size="sm" className="mt-2 rounded-full" disabled={busy === "upload"} onClick={() => fileRef.current?.click()}>{busy === "upload" ? <Loader2 className="animate-spin" /> : <Upload />} Upload plan</Button>
                      </>
                    ) : transform && (
                      <div className="mt-2 space-y-2">
                        <div className="truncate text-muted-foreground">{plan.uploaded.name}</div>
                        <label className="flex items-center gap-2"><Switch checked={align} onCheckedChange={(v: boolean) => { setAlign(v); if (v && view === "3d") setView("split"); }} /><Move className="size-3" /> Drag to align on the plan</label>
                        {([["scale", "Scale", 0.05, 5, 0.01], ["rotation", "Rotate°", -180, 180, 0.5], ["opacity", "Opacity", 0, 1, 0.05]] as [keyof PlanTransform, string, number, number, number][]).map(([k, l, lo, hi, st]) => (
                          <label key={k} className="grid grid-cols-[60px_1fr_44px] items-center gap-2">
                            <span className="text-muted-foreground">{l}</span>
                            <input type="range" min={lo} max={hi} step={st} value={transform[k]} onChange={(e) => setTr({ ...transform, [k]: +e.target.value })} className="accent-[var(--primary)]" />
                            <span className="text-right tabular-nums">{transform[k].toFixed(k === "rotation" ? 0 : 2)}</span>
                          </label>
                        ))}
                        <div className="flex flex-wrap gap-2">
                          <Button size="sm" className="rounded-full" disabled={!tr || !!busy} onClick={() => planCall("/transform", tr, "PATCH")}>Save alignment</Button>
                          <Button size="sm" variant="outline" className="rounded-full" onClick={() => fileRef.current?.click()}><Upload /> Replace</Button>
                          <Button size="sm" variant="ghost" className="rounded-full text-rose-600" onClick={() => planCall("/uploaded", undefined, "DELETE")}><Trash2 /></Button>
                        </div>
                      </div>
                    )}
                    <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadPlan(f); e.target.value = ""; }} />
                  </div>
                </>
              )}
              {panel === "plan" && planApi.loading && <Loading label="Generating floor plan…" />}
              {panel === "facilities" && (
                <>
                  {picked && (
                    <div className="rounded-2xl border p-3 text-xs">
                      <div className="mb-1 font-semibold capitalize">{picked.class_name.replaceAll("_", " ")} · {picked.facility_id.split("-").pop()}</div>
                      <div>registration confidence {fmt(picked.registration_confidence)} · {picked.support_points} support points</div>
                      <div>{picked.n_observations} observations · merged from {picked.merged_from} track(s)</div>
                      {picked.measurements?.map((mm: { name: string; value: unknown; unit?: string; status: string }) => <div key={mm.name} className="flex justify-between"><span>{mm.name}</span><span>{fmt(mm.value, mm.unit)} <StatusBadge status={mm.status} /></span></div>)}
                    </div>
                  )}
                  <div className="space-y-1 text-xs">
                    {s.facilities?.map((f) => (
                      <button key={f.facility_id} onClick={() => { setPick(f.facility_id); if (f.position_local) goTo(f.position_local); }} disabled={!f.position_sfm}
                        className={cn("flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left", (pick ?? sp.get("facility")) === f.facility_id ? "bg-accent" : "hover:bg-muted", "disabled:opacity-50")}>
                        <span className="flex items-center gap-2 capitalize"><span className="size-2 rounded-full" style={{ background: CLASS_COLORS[f.class_name] }} />{f.class_name.replaceAll("_", " ")}</span>
                        <span className="text-muted-foreground">{f.position_sfm ? `conf ${fmt(f.registration_confidence)}` : "not in 3D"}</span>
                      </button>
                    ))}
                  </div>
                </>
              )}
              {panel === "details" && (
                <>
                  {rid && <MobilePublish rid={rid} />}
                  <div className="grid grid-cols-2 gap-x-3 gap-y-3 text-xs">
                    {[
                      ["Capture quality", `${m?.capture_quality?.suitability} (${m?.capture_quality?.suitability_score})`],
                      ["Registered frames", `${m?.registration?.registered_primary}/${m?.registration?.frames}`],
                      ["Pose completeness", fmt(m?.registration?.pose_completeness)],
                      ["Reprojection error", fmt(m?.registration?.mean_reprojection_error_px, "px")],
                      ["Sparse points", m?.registration?.points?.toLocaleString()],
                      ["Splat backend", m?.gaussian_splat?.backend],
                      ["Gaussians", m?.gaussian_splat?.num_gaussians?.toLocaleString()],
                      ["Held-out PSNR", fmt(m?.gaussian_splat?.test_psnr, "dB")],
                      ["Facilities in 3D", `${m?.facilities?.registered}/${m?.facilities?.total}`],
                      ["Scale", s.recon?.alignment?.scale_status?.replaceAll("_", " ")],
                    ].map(([k, v]) => <div key={k}><div className="text-muted-foreground">{k}</div><div className="font-medium">{v ?? "—"}</div></div>)}
                  </div>
                  <div className="space-y-1 text-xs">
                    <div className="eyebrow mt-2">Downloads</div>
                    {s.assets && Object.entries(s.assets).map(([k, a]) => (
                      <a key={k} href={apiUrl(a.url)} className="flex items-center justify-between rounded-lg px-2 py-1 hover:bg-muted" title={a.description}>
                        <span className="flex items-center gap-2"><Download className="size-3" />{k}</span><span className="text-muted-foreground">{(a.bytes / 1e6).toFixed(2)} MB</span>
                      </a>
                    ))}
                  </div>
                  {rid && <DeleteScan rid={rid} onDeleted={() => { setRid(null); setSelPlace(null); setSelIssue(null); list.reload(); }} />}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

type GeoPin = { lat: number; lng: number; radius: number };
type MapRow = { id: string; name: string; description: string | null; hidden: boolean; version: string; places: number; location: GeoPin | null };

/** Name / hide this scan as it appears in the AccessMap phone app (synced via GET /maps). */
function MobilePublish({ rid }: { rid: string }) {
  const maps = useApi<{ maps: MapRow[] }>("/maps?include_hidden=true");
  const m = maps.data?.maps.find((x) => x.id === rid);
  const [name, setName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (!maps.data) return null;
  if (!m) return <div className="rounded-2xl bg-muted/60 p-3 text-xs text-muted-foreground">Not available to the phone app yet (the scan has to finish first).</div>;
  async function patch(body: Record<string, unknown>) {
    setBusy(true); setErr(null);
    try { await apiSend(`/maps/${rid}`, body, "PATCH"); maps.reload(); setName(null); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  const value = name ?? m.name;
  return (
    <div className="rounded-2xl border p-3 text-xs">
      <div className="flex items-center justify-between">
        <div className="font-semibold">Phone app</div>
        <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-semibold", m.hidden ? "bg-slate-100 text-slate-600" : "bg-emerald-100 text-emerald-700")}>{m.hidden ? "HIDDEN" : "SYNCING TO PHONES"}</span>
      </div>
      <p className="mt-1 text-muted-foreground">Phones download this map ({m.places} places, version {m.version.slice(0, 6)}) for offline use. Edits to places re-sync automatically.</p>
      <div className="mt-2 flex gap-2">
        <input value={value} onChange={(e) => setName(e.target.value)} className="field" aria-label="Name shown in the app" />
        <Button size="sm" className="rounded-full" disabled={busy || !value.trim() || value === m.name} onClick={() => patch({ name: value })}>Save</Button>
      </div>
      <label className="mt-2 flex items-center gap-2"><Switch checked={!m.hidden} onCheckedChange={(v: boolean) => patch({ hidden: !v })} /> Show in the phone app</label>
      <GeoEditor value={m.location} busy={busy} onSave={(loc) => patch({ location: loc })} onError={setErr} />
      {err && <div className="mt-2 text-rose-600">{err}</div>}
    </div>
  );
}

/** Where the building is: phones use GPS to open this map automatically when they are inside this circle. */
function GeoEditor({ value, busy, onSave, onError }: { value: GeoPin | null; busy: boolean; onSave: (v: GeoPin | null) => void; onError: (e: string) => void }) {
  const [draft, setDraft] = useState<{ lat: string; lng: string; radius: string } | null>(null);
  const [locating, setLocating] = useState(false);
  const d = draft ?? { lat: value ? String(value.lat) : "", lng: value ? String(value.lng) : "", radius: String(value?.radius ?? 80) };
  function here() {
    if (!navigator.geolocation) return onError("This browser cannot share its location.");
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (p) => { setLocating(false); setDraft({ lat: p.coords.latitude.toFixed(6), lng: p.coords.longitude.toFixed(6), radius: String(Math.min(300, Math.max(60, Math.round(p.coords.accuracy * 2)))) }); },
      (e) => { setLocating(false); onError(`Location unavailable: ${e.message}`); },
      { enableHighAccuracy: true, timeout: 12000 },
    );
  }
  const valid = d.lat.trim() !== "" && d.lng.trim() !== "" && !isNaN(+d.lat) && !isNaN(+d.lng);
  return (
    <div className="mt-3 border-t pt-3">
      <div className="flex items-center justify-between">
        <div className="font-medium">Building location</div>
        {value && <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">PINNED</span>}
      </div>
      <p className="mt-1 text-muted-foreground">Phones inside this circle open this map automatically (&quot;You are here&quot;).</p>
      <div className="mt-2 grid grid-cols-[1fr_1fr_64px] gap-1.5">
        <input value={d.lat} onChange={(e) => setDraft({ ...d, lat: e.target.value })} placeholder="Latitude" className="field" aria-label="Latitude" />
        <input value={d.lng} onChange={(e) => setDraft({ ...d, lng: e.target.value })} placeholder="Longitude" className="field" aria-label="Longitude" />
        <input value={d.radius} onChange={(e) => setDraft({ ...d, radius: e.target.value })} placeholder="m" className="field" aria-label="Radius in metres" />
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <Button size="sm" variant="outline" className="rounded-full" disabled={locating} onClick={here}>{locating ? <Loader2 className="animate-spin" /> : <Crosshair />} Use my location</Button>
        <Button size="sm" className="rounded-full" disabled={busy || !valid || !draft} onClick={() => { onSave({ lat: +d.lat, lng: +d.lng, radius: +d.radius || 80 }); setDraft(null); }}>Save location</Button>
        {value && <Button size="sm" variant="ghost" className="rounded-full text-rose-600" disabled={busy} onClick={() => { onSave(null); setDraft(null); }}>Clear</Button>}
      </div>
    </div>
  );
}

/** Permanently delete this reconstruction (outputs, places, map on phones). The source video is kept. */
function DeleteScan({ rid, onDeleted }: { rid: string; onDeleted: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function del() {
    if (!confirm(`Delete map ${rid} permanently?\n\nThis removes the 3D model, tagged places, floor plan and score, and the map disappears from phones on their next sync. The source video is kept so you can process it again. This cannot be undone.`)) return;
    setBusy(true); setErr(null);
    try { await apiSend(`/spatial/jobs/${rid}`, undefined, "DELETE"); onDeleted(); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  return (
    <div className="rounded-2xl border border-rose-200 bg-rose-50/40 p-3 text-xs">
      <div className="font-semibold text-rose-700">Delete this map</div>
      <p className="mt-1 text-muted-foreground">Removes the 3D model, places, plan and score, and removes it from phones. The source video stays.</p>
      <Button size="sm" variant="destructive" className="mt-2 rounded-full" disabled={busy} onClick={del}>{busy ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete map</Button>
      {err && <div className="mt-2 text-rose-600">{err}</div>}
    </div>
  );
}

export default function Page() {
  return <Suspense fallback={<Loading />}><Explorer /></Suspense>;
}
