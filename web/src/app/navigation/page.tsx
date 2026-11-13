"use client";
import dynamic from "next/dynamic";
import { useMemo, useRef, useState } from "react";
import { Accessibility, Camera, Loader2, MapPin, Mic, Navigation as NavIcon, Square, Volume2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useApi } from "@/hooks/use-api";
import { useScene } from "@/hooks/use-scene";
import { LANGUAGES, useRecorder } from "@/hooks/use-recorder";
import { apiSend, apiUpload } from "@/lib/api";
import { CLASS_COLORS, Empty, ErrorState, Loading, PageHeader, StatusBadge } from "@/components/app/common";

const SceneViewer = dynamic(() => import("@/components/three/scene-viewer"), { ssr: false, loading: () => <Loading label="Starting 3D viewer…" /> });

type Node = { id: string; kind: "trajectory" | "facility"; pos: number[]; class_name?: string; frame?: string; confidence?: number };
type Graph = { nodes: Node[]; edges: unknown[]; units: string; scale_status: string; facilities_unregistered: number; note: string; profiles: Record<string, string> };
type Route = {
  feasible: boolean; reason?: string; profile: string; profile_label?: string; goal?: string; path: number[][]; nodes?: string[];
  length?: number; units?: string; steps: { kind: string; length: number; instruction: string }[]; warnings?: string[];
};
type Transcript = { text: string; language: string | null; intent: { destination_class: string; keyword: string } | null; ms: number };
type Loc = { best_node: string; estimated_position_local: number[]; confidence: string; matches: { frame: string; similarity: number }[]; method: string };

export default function NavigationPage() {
  const recons = useApi<{ reconstructions: { id: string; status: string }[] }>("/spatial/reconstructions");
  const [chosen, setRid] = useState<string | null>(null);
  const rid = chosen ?? recons.data?.reconstructions.find((x) => x.status.startsWith("completed"))?.id ?? null;
  const graph = useApi<Graph>(rid ? `/navigation/${rid}/graph` : null);
  const scene = useScene(rid);
  const [profile, setProfile] = useState("wheelchair");
  const [start, setStart] = useState("cam0");
  const [goal, setGoal] = useState("");
  const [route, setRoute] = useState<Route | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [lang, setLang] = useState("");
  const [heard, setHeard] = useState<Transcript | null>(null);
  const [loc, setLoc] = useState<Loc | null>(null);
  const rec = useRecorder();
  const fileRef = useRef<HTMLInputElement>(null);

  const facilityNodes = useMemo(() => (graph.data?.nodes || []).filter((n) => n.kind === "facility"), [graph.data]);
  const classes = useMemo(() => [...new Set(facilityNodes.map((n) => n.class_name!))].sort(), [facilityNodes]);
  const camNodes = useMemo(() => (graph.data?.nodes || []).filter((n) => n.kind === "trajectory"), [graph.data]);
  const nodeById = useMemo(() => new Map((graph.data?.nodes || []).map((n) => [n.id, n])), [graph.data]);

  async function go(g = goal, p = profile) {
    if (!rid || !g) return;
    setBusy("route"); setErr(null);
    try { setRoute(await apiSend<Route>(`/navigation/${rid}/route`, { start, goal: g, profile: p })); }
    catch (e) { setErr((e as Error).message); setRoute(null); } finally { setBusy(null); }
  }

  function speak(r: Route) {
    if (!("speechSynthesis" in window)) return setErr("This browser has no speech synthesis.");
    window.speechSynthesis.cancel();
    const text = r.feasible ? r.steps.map((s) => s.instruction).join(". ") + ". You have arrived." : `No route. ${r.reason}`;
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
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
        const c = t.intent?.destination_class;
        if (c && classes.includes(c)) { setGoal(c); await go(c); }
        else if (c) setErr(`Heard "${c}", but no ${c} is registered in this reconstruction.`);
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
      setLoc(l);
      setStart(l.best_node);
      setRoute(null);
    } catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }

  const startNode = nodeById.get(start);
  const goalNode = route?.goal ? nodeById.get(route.goal) : null;
  const markers = [
    ...(startNode ? [{ pos: startNode.pos, label: "Start", color: "#16a34a" }] : []),
    ...(goalNode ? [{ pos: goalNode.pos, label: "Destination", color: "#e11d48" }] : []),
  ];

  return (
    <>
      <PageHeader title="Accessible navigation" description="Routes over the walkable graph recovered from a reconstruction (the camera's walked path plus facilities registered in 3D). Profiles change edge costs: the wheelchair profile forbids stairs. Unexplored space is unknown, not blocked." />
      {recons.error && <ErrorState message={recons.error} onRetry={recons.reload} />}
      {recons.data && !recons.data.reconstructions.length && <Empty title="No reconstructions">Build one from a walkthrough video in AI Studio first.</Empty>}
      {graph.error && <ErrorState message={graph.error} />}
      {graph.data && (
        <div className="grid gap-4 xl:grid-cols-[380px_1fr]">
          <div className="space-y-4">
            <Card className="shadow-none">
              <CardHeader><CardTitle className="flex items-center gap-2 text-base"><NavIcon className="size-4" /> Plan a route</CardTitle>
                <CardDescription>{camNodes.length} path nodes · {facilityNodes.length} registered facilities · {graph.data.facilities_unregistered} detected but not placed in 3D</CardDescription></CardHeader>
              <CardContent className="space-y-3 text-sm">
                <label className="block"><span className="text-xs text-muted-foreground">Reconstruction</span>
                  <select value={rid || ""} onChange={(e) => { setRid(e.target.value); setRoute(null); setStart("cam0"); }} className="mt-1 h-9 w-full rounded-md border bg-card px-2">
                    {recons.data?.reconstructions.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status}</option>)}
                  </select></label>
                <div>
                  <span className="text-xs text-muted-foreground">Accessibility profile</span>
                  <div className="mt-1 grid grid-cols-3 gap-1">
                    {Object.entries(graph.data.profiles).map(([k, v]) => (
                      <button key={k} onClick={() => { setProfile(k); if (goal) go(goal, k); }} className={`rounded-md border px-2 py-1.5 text-xs ${profile === k ? "border-primary bg-accent font-medium" : "hover:bg-muted"}`}>
                        {k === "wheelchair" && <Accessibility className="mr-1 inline size-3" />}{v}
                      </button>
                    ))}
                  </div>
                </div>
                <label className="block"><span className="text-xs text-muted-foreground">Start</span>
                  <select value={start} onChange={(e) => { setStart(e.target.value); setLoc(null); setRoute(null); }} className="mt-1 h-9 w-full rounded-md border bg-card px-2">
                    {camNodes.map((n, i) => <option key={n.id} value={n.id}>{i === 0 ? "Video start" : `Path point ${i}`} ({n.frame})</option>)}
                  </select></label>
                <Button variant="outline" size="sm" className="w-full" disabled={!!busy} onClick={() => fileRef.current?.click()}>
                  {busy === "locate" ? <Loader2 className="animate-spin" /> : <Camera />} Locate me from a photo (DINOv2)
                </Button>
                <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && locate(e.target.files[0])} />
                {loc && (
                  <div className="rounded-md bg-muted p-2 text-xs">
                    Matched keyframe {loc.matches[0].frame} (similarity {loc.matches[0].similarity}) · confidence <StatusBadge status={loc.confidence} />
                    {loc.confidence === "low" && <div className="mt-1 text-amber-700">Low confidence: the photo may be from outside this reconstruction. Check the start point.</div>}
                  </div>
                )}
                <label className="block"><span className="text-xs text-muted-foreground">Destination</span>
                  <select value={goal} onChange={(e) => { setGoal(e.target.value); setRoute(null); }} className="mt-1 h-9 w-full rounded-md border bg-card px-2">
                    <option value="">Choose…</option>
                    <optgroup label="Nearest of a kind">{classes.map((c) => <option key={c} value={c}>Nearest {c.replaceAll("_", " ")}</option>)}</optgroup>
                    <optgroup label="Specific facility">{facilityNodes.map((n) => <option key={n.id} value={n.id}>{n.class_name} · {n.id}</option>)}</optgroup>
                  </select></label>
                <div className="flex gap-2">
                  <select value={lang} onChange={(e) => setLang(e.target.value)} className="h-9 rounded-md border bg-card px-2 text-xs" title="Spoken language hint for Whisper">
                    {LANGUAGES.map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                  </select>
                  <Button variant={rec.recording ? "destructive" : "outline"} className="flex-1" disabled={busy === "voice"} onClick={voice}>
                    {rec.recording ? <><Square /> Stop</> : busy === "voice" ? <><Loader2 className="animate-spin" /> Transcribing…</> : <><Mic /> Say destination</>}
                  </Button>
                </div>
                {rec.error && <div className="text-xs text-destructive">{rec.error}</div>}
                {heard && <div className="rounded-md bg-muted p-2 text-xs">Heard ({heard.language || "?"}): &ldquo;{heard.text}&rdquo; → {heard.intent ? <b>{heard.intent.destination_class}</b> : "no destination recognised"}</div>}
                <Button className="w-full" disabled={!goal || !!busy} onClick={() => go()}>{busy === "route" ? <Loader2 className="animate-spin" /> : <MapPin />} Find route</Button>
                {err && <ErrorState message={err} />}
              </CardContent>
            </Card>
            {route && (
              <Card className="shadow-none">
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle className="text-base">{route.feasible ? `Route · ${route.length?.toFixed(1)} ${route.units}` : "No accessible route"}</CardTitle>
                  <Button size="sm" variant="outline" onClick={() => speak(route)}><Volume2 /> Read aloud</Button>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {!route.feasible && <div className="rounded-md border border-rose-200 bg-rose-50 p-2 text-rose-800">{route.reason}</div>}
                  <ol className="space-y-1">
                    {route.steps.map((s, i) => (
                      <li key={i} className={`flex gap-2 rounded px-2 py-1 ${s.kind.includes("stairs") ? "bg-amber-50 text-amber-900" : ""}`}><span className="text-muted-foreground tabular-nums">{i + 1}.</span>{s.instruction}</li>
                    ))}
                  </ol>
                  {route.units === "units" && <div className="text-xs text-muted-foreground">Distances are in reconstruction units: this reconstruction has no metric scale reference.</div>}
                </CardContent>
              </Card>
            )}
          </div>
          <Card className="overflow-hidden p-0 shadow-none">
            <div className="flex flex-wrap items-center gap-3 border-b px-4 py-2 text-xs">
              {classes.map((c) => <span key={c} className="flex items-center gap-1 capitalize"><span className="size-2 rounded-full" style={{ background: CLASS_COLORS[c] }} />{c.replaceAll("_", " ")}</span>)}
              <span className="ml-auto text-muted-foreground">{graph.data.note}</span>
            </div>
            <div className="h-[72vh]">
              {scene.scene ? (
                <SceneViewer data={scene.scene} layers={{ splat: true, points: false, trajectory: true, facilities: true }}
                  route={route?.feasible ? route.path : null} highlight={route?.goal} markers={markers} />
              ) : scene.error ? <ErrorState message={scene.error} /> : <Loading label="Loading reconstruction…" />}
            </div>
          </Card>
        </div>
      )}
      {graph.loading && <Loading label="Building navigation graph…" />}
    </>
  );
}
