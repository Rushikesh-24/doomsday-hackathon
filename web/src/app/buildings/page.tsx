"use client";
// Multi-floor buildings: one scan per floor, lifts / stairs linking them (auto-linked by name), the stacked 3D
// building, and routes across floors ("from the ground-floor entrance to CG Lab on the 2nd floor").
import dynamic from "next/dynamic";
import Link from "next/link";
import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ArrowDownUp, CheckCircle2, Layers, Loader2, Navigation, Plus, Sparkles, Trash2, TriangleAlert, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { useApi } from "@/hooks/use-api";
import { apiSend } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Empty, ErrorState, Loading } from "@/components/app/common";
import type { BModel } from "@/components/three/building-view";

const BuildingView = dynamic(() => import("@/components/three/building-view"), { ssr: false, loading: () => <Loading label="Starting 3D view…" /> });

type Place = { id: string; name: string; kind: string; position: number[]; scan: string };
type Scan = { scan: string; name: string; floor_id: string; to_floor_id: string; level: number; to_level: number; ready: boolean;
  manual_align: { yaw: number; dx: number; dz: number } | null; transform: { yaw_deg: number; dx: number; dz: number; source: string } | null };
type Stitch = { created: string; pairs: { x: string; y: string; x_name: string; y_name: string; shared_levels: number[]; queries: number; matches: number; linked: boolean; agreeing: number }[] } | null;
type Floor = { id: string; name: string; level: number; label: string; scan: string | null; ready: boolean; places: Place[];
  manual_align: { yaw: number; dx: number; dz: number } | null; transform: { yaw_deg: number; dx: number; dz: number; source: string } | null };
type Connector = { id: string; kind: "lift" | "stairs"; name: string; landings: { floor_id: string; place_id: string }[]; out_of_service: boolean; floors: number[] };
type Building = { id: string; name: string; floor_height: number; floors: Floor[]; connectors: Connector[]; scans: Scan[]; stitch: Stitch;
  summary: { step_free_floors: number; total_floors: number; floors: { level: number; name: string; step_free_from_entrance: boolean; score: { score: number; grade: string } | null }[] } | null; error?: string };
type MapRow = { id: string; name: string; kind?: string };
type Step = { kind: string; text: string; notes?: string[]; floor: number; to_floor?: number };
type Route = { feasible: boolean; reason?: string; length?: number; directions: Step[]; path: number[][]; path_floors: number[]; floors: number[];
  rides?: { name: string; kind: string; from_floor: number; to_floor: number }[]; start_name?: string; goal_name?: string };
type Tab = "floors" | "connectors" | "route";

const CONNECTOR_PLACE_KINDS = ["lift", "stairwell", "stairs"];

function Buildings() {
  const sp = useSearchParams();
  const list = useApi<{ buildings: { id: string; name: string; floors: number; floors_ready: number; connectors: number }[] }>("/buildings");
  const [chosen, setChosen] = useState<string | null>(sp.get("id"));
  const bid = chosen ?? list.data?.buildings[0]?.id ?? null;
  const b = useApi<Building>(bid ? `/buildings/${bid}` : null);
  const model = useApi<BModel>(bid && b.data?.floors.some((f) => f.ready) ? `/buildings/${bid}/model?v=${b.data?.scans.map((x) => x.scan + x.level + x.to_level + JSON.stringify(x.manual_align)).join()}${b.data?.connectors.map((c) => c.id + c.out_of_service + c.landings.length).join()}${b.data?.stitch?.created ?? ""}` : null);
  const maps = useApi<{ maps: MapRow[] }>("/maps?include_hidden=true");
  const [tab, setTab] = useState<Tab>("floors");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [newFloors, setNewFloors] = useState(4);
  const [activeFloor, setActiveFloor] = useState<number | null>(null);
  const [gap, setGap] = useState(2);   // floors spread apart: corridors are long, floors only ~3.5 m apart
  const [route, setRoute] = useState<Route | null>(null);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [profile, setProfile] = useState("wheelchair");

  const scans = (maps.data?.maps ?? []).filter((m) => m.kind !== "building");
  const allPlaces = useMemo(() => (b.data?.floors ?? []).flatMap((f) => f.places.map((p) => ({ ...p, level: f.level, floor: f.name }))), [b.data]);

  async function call<T>(key: string, fn: () => Promise<T>, after?: (r: T) => void) {
    setBusy(key); setErr(null); setMsg(null);
    try { const r = await fn(); after?.(r); b.reload(); list.reload(); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }
  const patchFloor = (fid: string, body: unknown) => call(`floor:${fid}`, () => apiSend(`/buildings/${bid}/floors/${fid}`, body, "PATCH"), () => setRoute(null));

  async function go() {
    if (!bid || !to) return;
    setBusy("route"); setErr(null);
    try {
      const r = await apiSend<Route>(`/buildings/${bid}/route`, { from: from || null, to, profile });
      setRoute(r);
      if (r.feasible && r.floors.length) setActiveFloor(null);
    } catch (e) { setErr((e as Error).message); setRoute(null); } finally { setBusy(null); }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="eyebrow">Digital twin · multi-floor</div>
          <h1 className="font-display text-3xl font-semibold tracking-tight">Buildings</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">Stack one scan per floor into a whole building. Tag the lift doors and stairwell landings with the same name on every floor (e.g. <b>Lift 1</b>, <b>Main stairs</b>) and they are linked automatically. Routes then go across floors: wheelchair routes take the lift.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {!!list.data?.buildings.length && (
            <select value={bid ?? ""} onChange={(e) => { setChosen(e.target.value); setRoute(null); setActiveFloor(null); }} className="h-9 rounded-full border bg-card px-3 text-sm">
              {list.data.buildings.map((x) => <option key={x.id} value={x.id}>{x.name} · {x.floors_ready}/{x.floors} floors</option>)}
            </select>
          )}
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="New building, e.g. Main Block" className="field h-9 w-56" />
          <select value={newFloors} onChange={(e) => setNewFloors(+e.target.value)} className="h-9 rounded-full border bg-card px-2 text-sm" aria-label="Number of floors">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => <option key={n} value={n}>{n} floor{n > 1 ? "s" : ""}</option>)}
          </select>
          <Button className="rounded-full" disabled={!newName.trim() || !!busy} onClick={() => call("new", () => apiSend<Building>("/buildings", { name: newName, floors: newFloors }), (r) => { setChosen(r.id); setNewName(""); })}><Plus /> Create</Button>
        </div>
      </div>
      {list.error && <ErrorState message={list.error} onRetry={list.reload} />}
      {err && <ErrorState message={err} />}
      {msg && <div className="rounded-xl bg-emerald-50 px-3 py-2 text-sm text-emerald-800">{msg}</div>}
      {list.data && !list.data.buildings.length && <Empty title="No buildings yet">Create one above, e.g. <b>Main Block</b> with 4 floors, then pick each floor&apos;s scan.</Empty>}
      {b.data && (
        <div className="grid gap-4 xl:grid-cols-[1fr_420px]">
          <div className="surface relative h-[calc(100vh-200px)] min-h-[560px] overflow-hidden p-0">
            {model.data && model.data.floors.length ? (
              <BuildingView model={model.data} route={route?.feasible ? route : null} activeFloor={activeFloor} gap={gap} />
            ) : (
              <div className="grid h-full place-items-center p-8 text-center text-sm text-muted-foreground">
                {model.loading ? <Loading label="Stacking floors…" /> : <div><Layers className="mx-auto mb-2 size-8 text-violet-400" />Assign a finished scan to at least one floor to see the building.</div>}
              </div>
            )}
            <div className="pointer-events-none absolute inset-x-3 top-3 flex flex-wrap items-center gap-2">
              <div className="glass pointer-events-auto inline-flex rounded-full p-1 text-xs">
                <button onClick={() => setActiveFloor(null)} className={cn("rounded-full px-3 py-1.5 font-medium", activeFloor === null ? "bg-primary text-primary-foreground" : "")}>All floors</button>
                {b.data.floors.map((f) => (
                  <button key={f.id} onClick={() => setActiveFloor(f.level)} disabled={!f.ready}
                    className={cn("rounded-full px-3 py-1.5 font-medium disabled:opacity-40", activeFloor === f.level ? "bg-primary text-primary-foreground" : "")}>{f.level === 0 ? "G" : f.level}</button>
                ))}
              </div>
              <label className="glass pointer-events-auto flex items-center gap-2 rounded-full px-3 py-1.5 text-xs"><ArrowDownUp className="size-3.5" /> Spread
                <input type="range" min={1} max={3} step={0.1} value={gap} onChange={(e) => setGap(+e.target.value)} className="w-24 accent-[var(--primary)]" />
              </label>
            </div>
            {b.data.summary && (
              <div className="glass absolute bottom-3 left-3 rounded-2xl px-3 py-2 text-xs">
                <div className="font-semibold">{b.data.summary.step_free_floors}/{b.data.summary.total_floors} floors reachable step-free from the entrance</div>
                <div className="mt-0.5 flex flex-wrap gap-1">
                  {b.data.summary.floors.map((f) => (
                    <span key={f.level} className={cn("rounded-full px-1.5 py-0.5 font-medium", f.step_free_from_entrance ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-700")}>
                      {f.level === 0 ? "G" : f.level}{f.score ? ` · ${f.score.score}` : ""}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="surface flex max-h-[calc(100vh-200px)] min-h-[560px] flex-col overflow-hidden p-0">
            <div className="flex gap-1 border-b p-2">
              {([["floors", `Floors (${b.data.floors.length})`], ["connectors", `Lifts & stairs (${b.data.connectors.length})`], ["route", "Route"]] as [Tab, string][]).map(([k, l]) => (
                <button key={k} onClick={() => setTab(k)} className={cn("flex-1 rounded-lg px-2 py-1.5 text-xs font-medium", tab === k ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-muted")}>{l}</button>
              ))}
            </div>
            <div className="flex-1 space-y-3 overflow-auto p-4 text-sm">
              {tab === "floors" && (
                <>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">Floor height</span>
                    <input type="number" min={2} max={10} step={0.1} defaultValue={b.data.floor_height} key={b.data.floor_height}
                      onBlur={(e) => +e.target.value !== b.data!.floor_height && call("fh", () => apiSend(`/buildings/${bid}`, { floor_height: +e.target.value }, "PATCH"))}
                      className="field h-8 w-20" /> m
                    <Button size="sm" variant="ghost" className="ml-auto rounded-full text-rose-600" onClick={() => confirm(`Delete building "${b.data!.name}"? The floor scans themselves are kept.`) && call("del", () => apiSend(`/buildings/${bid}`, undefined, "DELETE"), () => setChosen(null))}><Trash2 /> Delete building</Button>
                  </div>
                  <div className="rounded-2xl border border-violet-200 bg-violet-50/50 p-3 text-xs">
                    <div className="font-semibold">Join the scans</div>
                    <p className="mt-0.5 text-muted-foreground">Finds where videos overlap (same corridor, bottom and top of the stairs) by matching their frames, then joins and lines them up. Takes a few seconds per pair.</p>
                    <Button size="sm" className="mt-2 w-full rounded-full" disabled={!!busy || b.data.scans.filter((x) => x.ready).length < 2}
                      onClick={() => call("stitch", () => apiSend<{ stitch: Stitch }>(`/buildings/${bid}/stitch`), (r) => setMsg(`Stitched: ${r.stitch?.pairs.filter((x) => x.linked).length ?? 0} of ${r.stitch?.pairs.length ?? 0} overlapping pairs joined.`))}>
                      {busy === "stitch" ? <><Loader2 className="animate-spin" /> Matching frames…</> : <><Wand2 /> Stitch scans</>}
                    </Button>
                    {b.data.stitch && (
                      <div className="mt-2 space-y-0.5">
                        {b.data.stitch.pairs.map((x, i) => (
                          <div key={i} className={cn("flex justify-between gap-2", x.linked ? "text-emerald-800" : "text-muted-foreground")}>
                            <span className="truncate">{x.x_name} ↔ {x.y_name}</span><span className="shrink-0">{x.linked ? `joined · ${x.agreeing} views` : `no overlap found (${x.matches})`}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  {[...b.data.floors].sort((x, y) => y.level - x.level).map((f) => {
                    const mine = b.data!.scans.filter((x) => x.level === f.level);
                    const arriving = b.data!.scans.filter((x) => x.to_level === f.level && x.level !== f.level);
                    return (
                    <div key={f.id} className={cn("rounded-2xl border p-3", activeFloor === f.level && "border-primary/40 bg-accent/40")} onMouseEnter={() => f.ready && setActiveFloor(f.level)}>
                      <div className="flex items-center gap-2">
                        <span className="grid size-8 place-items-center rounded-xl bg-slate-900 text-xs font-bold text-white">{f.level === 0 ? "G" : f.level}</span>
                        <input defaultValue={f.name} key={f.name} onBlur={(e) => e.target.value.trim() && e.target.value !== f.name && patchFloor(f.id, { name: e.target.value })} className="field h-8 flex-1 font-medium" aria-label="Floor name" />
                        {f.ready ? <CheckCircle2 className="size-4 text-emerald-600" /> : <TriangleAlert className="size-4 text-amber-500" />}
                        <button className="text-muted-foreground hover:text-rose-600" title="Remove floor" onClick={() => confirm(`Remove ${f.name}? Its scans are kept.`) && call("df", () => apiSend(`/buildings/${bid}/floors/${f.id}`, undefined, "DELETE"))}><Trash2 className="size-3.5" /></button>
                      </div>
                      <div className="mt-2 space-y-1.5">
                        {mine.map((x) => <ScanRow key={x.scan} x={x} b={b.data!} onRemove={() => call("rs", () => apiSend(`/buildings/${bid}/scans/${x.scan}`, undefined, "DELETE"))}
                          onAlign={(a) => call("al", () => apiSend(`/buildings/${bid}/scans/${x.scan}`, { align: a }, "PATCH"))} />)}
                        {arriving.map((x) => <div key={x.scan} className="rounded-lg bg-orange-50 px-2 py-1 text-[11px] text-orange-900">Stairs arrive here from {x.level === 0 ? "the ground floor" : `floor ${x.level}`}: {x.name}</div>)}
                        {!mine.length && !arriving.length && <div className="text-[11px] text-amber-700">No scan on this floor yet.</div>}
                      </div>
                      <AddScan floors={b.data!.floors} floorId={f.id} options={scans.filter((m) => !b.data!.scans.some((x) => x.scan === m.id))}
                        onAdd={(scan, to) => call("as", () => apiSend(`/buildings/${bid}/scans`, { scan, floor_id: f.id, to_floor_id: to }), () => setRoute(null))} />
                      {f.ready && (
                        <div className="mt-2 text-[11px] text-muted-foreground">{f.places.length} places{f.places.some((p) => CONNECTOR_PLACE_KINDS.includes(p.kind)) ? ` · ${f.places.filter((p) => CONNECTOR_PLACE_KINDS.includes(p.kind)).map((p) => p.name).join(", ")}` : ""}</div>
                      )}
                    </div>
                    );
                  })}
                  <Button size="sm" variant="outline" className="rounded-full" onClick={() => call("af", () => apiSend(`/buildings/${bid}/floors`, { level: Math.max(-1, ...b.data!.floors.map((f) => f.level)) + 1 }))}><Plus /> Add floor</Button>
                </>
              )}

              {tab === "connectors" && (
                <>
                  <div className="rounded-2xl bg-muted/60 p-3 text-xs">
                    On every floor tag the lift door as kind <b>Lift</b> and the stairwell landing as <b>Stairwell (landing)</b>, with the same name on each floor. Then:
                    <Button size="sm" className="mt-2 w-full rounded-full" disabled={!!busy} onClick={() => call("auto", () => apiSend<{ created: string[]; updated: string[]; only_one_floor: string[] }>(`/buildings/${bid}/connectors/auto`), (r) => setMsg(`Linked: ${[...r.created, ...r.updated].join(", ") || "nothing new"}${r.only_one_floor.length ? ` · tagged on one floor only: ${r.only_one_floor.join(", ")}` : ""}`))}>
                      {busy === "auto" ? <Loader2 className="animate-spin" /> : <Wand2 />} Link lifts &amp; stairs automatically
                    </Button>
                  </div>
                  {b.data.connectors.map((c) => <ConnectorCard key={c.id} c={c} b={b.data!} onSave={(body) => call(`c:${c.id}`, () => apiSend(`/buildings/${bid}/connectors/${c.id}`, body, "PATCH"), () => setRoute(null))}
                    onDelete={() => confirm(`Remove ${c.name}?`) && call("dc", () => apiSend(`/buildings/${bid}/connectors/${c.id}`, undefined, "DELETE"))} />)}
                  <NewConnector b={b.data} onCreate={(body) => call("nc", () => apiSend(`/buildings/${bid}/connectors`, body))} />
                </>
              )}

              {tab === "route" && (
                <>
                  <div className="space-y-2">
                    <select value={from} onChange={(e) => setFrom(e.target.value)} className="field h-9" aria-label="From">
                      <option value="">From: building entrance (default start)</option>
                      {allPlaces.map((p) => <option key={`f${p.level}${p.id}`} value={`${p.scan}|${p.id}`}>{p.name} · {p.floor}</option>)}
                    </select>
                    <select value={to} onChange={(e) => setTo(e.target.value)} className="field h-9" aria-label="To">
                      <option value="">To…</option>
                      {allPlaces.filter((p) => !CONNECTOR_PLACE_KINDS.includes(p.kind)).map((p) => <option key={`t${p.level}${p.id}`} value={`${p.scan}|${p.id}`}>{p.name} · {p.floor}</option>)}
                    </select>
                    <div className="inline-flex w-full rounded-full bg-muted p-1 text-xs">
                      {[["standard", "Walking"], ["wheelchair", "Wheelchair"], ["low_vision", "Low vision"]].map(([k, l]) => (
                        <button key={k} onClick={() => setProfile(k)} className={cn("flex-1 rounded-full px-3 py-1.5", profile === k ? "bg-white font-semibold shadow-sm" : "text-muted-foreground")}>{l}</button>
                      ))}
                    </div>
                    <Button className="w-full rounded-full" disabled={!to || busy === "route"} onClick={go}>{busy === "route" ? <Loader2 className="animate-spin" /> : <Navigation />} Show route</Button>
                  </div>
                  {route && !route.feasible && <div className="rounded-xl bg-rose-50 p-3 text-xs text-rose-800"><b>No route.</b> {route.reason}</div>}
                  {route?.feasible && (
                    <div className="space-y-2">
                      <div className="rounded-xl bg-accent/60 p-3 text-xs">
                        <div className="font-semibold">{route.start_name} → {route.goal_name}</div>
                        <div className="text-muted-foreground">{route.length?.toFixed(0)} m equivalent · {route.floors.length} floor{route.floors.length > 1 ? "s" : ""}{route.rides?.length ? ` · ${route.rides.map((r) => r.name).join(", ")}` : ""}</div>
                      </div>
                      {route.directions.map((s, i) => (
                        <div key={i} className={cn("flex gap-2 rounded-xl p-2 text-xs", s.kind === "lift" || s.kind === "stairs" ? "bg-blue-50" : "")}>
                          <span className="grid size-6 shrink-0 place-items-center rounded-lg bg-slate-900 text-[10px] font-bold text-white">{s.floor === 0 ? "G" : s.floor}</span>
                          <div>
                            <div className={cn(s.kind === "lift" || s.kind === "stairs" ? "font-semibold text-blue-900" : "")}>{s.text}</div>
                            {s.notes?.map((n, j) => <div key={j} className="mt-0.5 text-amber-800">{n}</div>)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="flex gap-1.5 text-[11px] text-muted-foreground"><Sparkles className="mt-0.5 size-3 shrink-0" /> The same building is available offline in the phone app (it syncs as one map with all floors).</p>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ScanRow({ x, b, onRemove, onAlign }: { x: Scan; b: Building; onRemove: () => void; onAlign: (a: { yaw: number; dx: number; dz: number } | null) => void }) {
  const to = b.floors.find((f) => f.id === x.to_floor_id);
  return (
    <div className="rounded-xl bg-muted/50 px-2 py-1.5 text-[11px]">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate font-medium" title={x.scan}>{x.name}</span>
        {x.to_level !== x.level && <span className="shrink-0 rounded-full bg-orange-100 px-1.5 font-semibold text-orange-800">stairs → {to?.level === 0 ? "G" : to?.level}</span>}
        {!x.ready && <span className="shrink-0 text-amber-700">processing…</span>}
        <Link href={`/explorer?id=${x.scan}&panel=places`} className="shrink-0 text-violet-700 underline">Tag places</Link>
        <button className="shrink-0 text-muted-foreground hover:text-rose-600" title="Remove from building" onClick={onRemove}><Trash2 className="size-3" /></button>
      </div>
      {x.transform && <div className={cn("mt-0.5", x.transform.source.startsWith("not linked") ? "text-amber-700" : "text-muted-foreground")}>Placement: {x.transform.source}</div>}
      {x.ready && <details><summary className="cursor-pointer text-violet-700">Adjust placement by hand</summary><AlignEditor f={x} onSave={onAlign} /></details>}
    </div>
  );
}

function AddScan({ floors, floorId, options, onAdd }: { floors: Floor[]; floorId: string; options: MapRow[]; onAdd: (scan: string, to: string | null) => void }) {
  const [scan, setScan] = useState("");
  const [to, setTo] = useState("");
  if (!options.length) return null;
  return (
    <div className="mt-2 flex gap-1.5">
      <select value={scan} onChange={(e) => setScan(e.target.value)} className="field h-8 min-w-0 flex-1 text-xs" aria-label="Add a scan">
        <option value="">+ add a scan of this floor…</option>
        {options.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select>
      <select value={to} onChange={(e) => setTo(e.target.value)} className="field h-8 w-28 text-xs" aria-label="Ends on floor" title="For stairs videos: the floor the video ends on">
        <option value="">stays here</option>
        {floors.filter((f) => f.id !== floorId).map((f) => <option key={f.id} value={f.id}>ends on {f.level === 0 ? "G" : f.level}</option>)}
      </select>
      <Button size="sm" className="h-8 rounded-full" disabled={!scan} onClick={() => { onAdd(scan, to || null); setScan(""); setTo(""); }}><Plus /></Button>
    </div>
  );
}

function AlignEditor({ f, onSave }: { f: { manual_align: { yaw: number; dx: number; dz: number } | null; transform: { yaw_deg: number; dx: number; dz: number } | null }; onSave: (a: { yaw: number; dx: number; dz: number } | null) => void }) {
  const t = f.manual_align ?? (f.transform ? { yaw: f.transform.yaw_deg, dx: f.transform.dx, dz: f.transform.dz } : { yaw: 0, dx: 0, dz: 0 });
  const [a, setA] = useState(t);
  return (
    <div className="mt-2 space-y-1.5 rounded-xl bg-muted/50 p-2">
      {([["yaw", "Rotate°", -180, 180, 1], ["dx", "Shift x (m)", -60, 60, 0.1], ["dz", "Shift z (m)", -60, 60, 0.1]] as [keyof typeof a, string, number, number, number][]).map(([k, l, lo, hi, st]) => (
        <label key={k} className="grid grid-cols-[72px_1fr_44px] items-center gap-2">
          <span>{l}</span>
          <input type="range" min={lo} max={hi} step={st} value={a[k]} onChange={(e) => setA({ ...a, [k]: +e.target.value })} className="accent-[var(--primary)]" />
          <span className="text-right tabular-nums">{a[k].toFixed(k === "yaw" ? 0 : 1)}</span>
        </label>
      ))}
      <div className="flex gap-2">
        <Button size="sm" className="rounded-full" onClick={() => onSave(a)}>Save placement</Button>
        {f.manual_align && <Button size="sm" variant="ghost" className="rounded-full" onClick={() => onSave(null)}>Back to automatic</Button>}
      </div>
    </div>
  );
}

function ConnectorCard({ c, b, onSave, onDelete }: { c: Connector; b: Building; onSave: (body: unknown) => void; onDelete: () => void }) {
  const landing = (fid: string) => c.landings.find((x) => x.floor_id === fid)?.place_id ?? "";
  function setLanding(fid: string, pid: string) {
    const rest = c.landings.filter((x) => x.floor_id !== fid);
    onSave({ ...c, landings: pid ? [...rest, { floor_id: fid, place_id: pid }] : rest });
  }
  return (
    <div className={cn("rounded-2xl border p-3", c.out_of_service && "border-amber-300 bg-amber-50/50")}>
      <div className="flex items-center gap-2">
        <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-bold uppercase text-white", c.kind === "lift" ? "bg-blue-600" : "bg-orange-600")}>{c.kind}</span>
        <input defaultValue={c.name} key={c.name} onBlur={(e) => e.target.value.trim() && e.target.value !== c.name && onSave({ ...c, name: e.target.value })} className="field h-8 flex-1 font-semibold" aria-label="Name" />
        <button className="text-muted-foreground hover:text-rose-600" onClick={onDelete} title="Remove"><Trash2 className="size-3.5" /></button>
      </div>
      <div className="mt-2 text-[11px] text-muted-foreground">{c.kind === "lift" ? "Stops at" : "Connects"}: {c.floors.map((l) => (l === 0 ? "G" : l)).join(" · ") || "no floors yet"}</div>
      <div className="mt-2 space-y-1">
        {[...b.floors].sort((x, y) => y.level - x.level).map((f) => (
          <label key={f.id} className="grid grid-cols-[34px_1fr] items-center gap-2 text-xs">
            <span className="font-semibold">{f.level === 0 ? "G" : f.level}</span>
            <select value={landing(f.id)} onChange={(e) => setLanding(f.id, e.target.value)} disabled={!f.ready} className="field h-7 text-xs">
              <option value="">{c.kind === "lift" ? "does not stop here" : "not on this floor"}</option>
              {f.places.filter((p) => CONNECTOR_PLACE_KINDS.includes(p.kind) || landing(f.id) === p.id).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              {f.places.filter((p) => !CONNECTOR_PLACE_KINDS.includes(p.kind) && landing(f.id) !== p.id).length > 0 && <option disabled>──────────</option>}
              {f.places.filter((p) => !CONNECTOR_PLACE_KINDS.includes(p.kind) && landing(f.id) !== p.id).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </label>
        ))}
      </div>
      {c.kind === "lift" && (
        <label className="mt-2 flex items-center gap-2 text-xs"><Switch checked={c.out_of_service} onCheckedChange={(v: boolean) => onSave({ ...c, out_of_service: v })} /> Out of service (routes avoid it)</label>
      )}
    </div>
  );
}

function NewConnector({ b, onCreate }: { b: Building; onCreate: (body: unknown) => void }) {
  const [kind, setKind] = useState<"lift" | "stairs">("lift");
  const [name, setName] = useState("");
  return (
    <div className="rounded-2xl border border-dashed p-3 text-xs">
      <div className="mb-2 font-semibold">Add a lift or staircase by hand</div>
      <div className="flex gap-2">
        <select value={kind} onChange={(e) => setKind(e.target.value as "lift" | "stairs")} className="field h-8 w-24"><option value="lift">Lift</option><option value="stairs">Stairs</option></select>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder={kind === "lift" ? "Lift 2" : "Back stairs"} className="field h-8 flex-1" />
        <Button size="sm" className="rounded-full" disabled={!name.trim()} onClick={() => { onCreate({ kind, name, landings: [] }); setName(""); }}><Plus /> Add</Button>
      </div>
      <p className="mt-1 text-muted-foreground">Then choose its door / landing on each floor it serves ({b.floors.length} floors).</p>
    </div>
  );
}

export default function Page() {
  return <Suspense fallback={<Loading />}><Buildings /></Suspense>;
}

