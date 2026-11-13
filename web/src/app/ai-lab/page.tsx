"use client";
import { useState } from "react";
import { Cpu, FileAudio, ImageIcon, Loader2, Mic, MapPin, Square, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useApi } from "@/hooks/use-api";
import { LANGUAGES, useRecorder } from "@/hooks/use-recorder";
import { apiUpload, apiUrl } from "@/lib/api";
import { ErrorState, PageHeader, StatusBadge } from "@/components/app/common";

type ModelStatus = Record<string, { hf_id: string; license: string; task: string; loaded: boolean; device: string | null }>;
type Describe = { task: string; text: string; regions?: { text: string; quad: number[] }[]; ms: number; model: string };
type Transcript = { text: string; language: string | null; language_source: string; seconds: number; intent: { destination_class: string; keyword: string } | null; ms: number };
type Loc = { best_node: string; confidence: string; similarity_margin: number; matches: { frame: string; similarity: number; node: string }[]; method: string; ms: number };

const TASKS: [string, string][] = [["detailed_caption", "Describe scene"], ["caption", "Short caption"], ["ocr", "Read text (OCR)"], ["ocr_regions", "OCR with regions"]];

function FlorenceDemo() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [size, setSize] = useState<[number, number] | null>(null);
  const [task, setTask] = useState("detailed_caption");
  const [res, setRes] = useState<Describe | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run() {
    if (!file) return;
    setBusy(true); setErr(null); setRes(null);
    const fd = new FormData(); fd.append("file", file); fd.append("task", task);
    try { setRes(await apiUpload<Describe>("/ai/describe", fd)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  return (
    <Card className="shadow-none">
      <CardHeader><CardTitle className="flex items-center gap-2 text-base"><ImageIcon className="size-4" /> Florence-2 · scene description &amp; sign reading</CardTitle>
        <CardDescription>Describe a photo for blind and low-vision users, or read text on signs. First run loads the model (~10 s).</CardDescription></CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-2">
          <label className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-md border px-3 hover:bg-muted"><Upload className="size-4" /> {file ? file.name : "Choose photo"}
            <input type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) { setFile(f); setPreview(URL.createObjectURL(f)); setRes(null); } }} /></label>
          <select value={task} onChange={(e) => setTask(e.target.value)} className="h-9 rounded-md border bg-card px-2">{TASKS.map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select>
          <Button disabled={!file || busy} onClick={run}>{busy && <Loader2 className="animate-spin" />} Run</Button>
        </div>
        {preview && (
          <div className="relative w-fit max-w-full">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="" className="max-h-80 rounded border" onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])} />
            {size && res?.regions && (
              <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${size[0]} ${size[1]}`} preserveAspectRatio="none">
                {res.regions.map((r, i) => <polygon key={i} points={r.quad.join(",")} fill="rgba(250,204,21,0.18)" stroke="#facc15" strokeWidth={Math.max(2, size[0] / 300)} />)}
              </svg>
            )}
          </div>
        )}
        {err && <ErrorState message={err} />}
        {res && (
          <div className="rounded-md bg-muted p-3">
            <div className="whitespace-pre-wrap">{res.text || <span className="text-muted-foreground">No text found.</span>}</div>
            <div className="mt-2 text-xs text-muted-foreground">{res.model} · {res.ms} ms{res.regions ? ` · ${res.regions.length} text regions` : ""}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function WhisperDemo() {
  const [lang, setLang] = useState("");
  const [res, setRes] = useState<Transcript | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const rec = useRecorder(20);
  async function send(blob: Blob, name: string) {
    setBusy(true); setErr(null); setRes(null);
    const fd = new FormData(); fd.append("file", blob, name); if (lang) fd.append("language", lang);
    try { setRes(await apiUpload<Transcript>("/ai/transcribe", fd)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  return (
    <Card className="shadow-none">
      <CardHeader><CardTitle className="flex items-center gap-2 text-base"><Mic className="size-4" /> Whisper · multilingual voice requests</CardTitle>
        <CardDescription>Speak a destination (&ldquo;where is the lift?&rdquo;, &ldquo;लिफ्ट कहाँ है&rdquo;). The transcript is matched to a facility class used by Navigation. whisper-base is weak on Hindi without a language hint.</CardDescription></CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-2">
          <select value={lang} onChange={(e) => setLang(e.target.value)} className="h-9 rounded-md border bg-card px-2">{LANGUAGES.map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select>
          <Button variant={rec.recording ? "destructive" : "default"} disabled={busy} onClick={() => rec.recording ? rec.stop() : rec.start((b) => send(b, "speech.webm"))}>
            {rec.recording ? <><Square /> Stop recording</> : <><Mic /> Record</>}
          </Button>
          <label className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-md border px-3 hover:bg-muted"><FileAudio className="size-4" /> Upload audio
            <input type="file" accept="audio/*,video/webm" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) send(f, f.name); }} /></label>
          {busy && <span className="flex items-center gap-1 text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Transcribing…</span>}
        </div>
        {rec.error && <ErrorState message={rec.error} />}
        {err && <ErrorState message={err} />}
        {res && (
          <div className="rounded-md bg-muted p-3">
            <div>&ldquo;{res.text}&rdquo;</div>
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted-foreground">
              <span>language: {res.language || "?"} ({res.language_source})</span><span>{res.seconds}s audio</span><span>{res.ms} ms</span>
              <span>intent: {res.intent ? <b className="text-foreground">{res.intent.destination_class}</b> : "none recognised"}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function LocalizeDemo() {
  const recons = useApi<{ reconstructions: { id: string; status: string }[] }>("/spatial/reconstructions");
  const done = recons.data?.reconstructions.filter((r) => r.status.startsWith("completed")) || [];
  const [rid, setRid] = useState("");
  const [res, setRes] = useState<Loc | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const r = rid || done[0]?.id || "";
  async function run(f: File) {
    if (!r) return;
    setBusy(true); setErr(null); setRes(null);
    const fd = new FormData(); fd.append("file", f);
    try { setRes(await apiUpload<Loc>(`/ai/localize/${r}`, fd)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  return (
    <Card className="shadow-none">
      <CardHeader><CardTitle className="flex items-center gap-2 text-base"><MapPin className="size-4" /> DINOv2 · visual localisation</CardTitle>
        <CardDescription>Match a photo against a reconstruction&rsquo;s registered keyframes to estimate where it was taken (keyframe-level, not a 6-DoF pose).</CardDescription></CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-2">
          <select value={r} onChange={(e) => setRid(e.target.value)} className="h-9 rounded-md border bg-card px-2">{done.map((x) => <option key={x.id} value={x.id}>{x.id}</option>)}</select>
          <label className={`inline-flex h-9 items-center gap-2 rounded-md border px-3 ${r ? "cursor-pointer hover:bg-muted" : "opacity-50"}`}>{busy ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />} Photo to locate
            <input type="file" accept="image/*" hidden disabled={!r || busy} onChange={(e) => e.target.files?.[0] && run(e.target.files[0])} /></label>
        </div>
        {!done.length && recons.data && <div className="text-xs text-muted-foreground">No completed reconstruction to localise against.</div>}
        {err && <ErrorState message={err} />}
        {res && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">Best match <b>{res.matches[0].frame}</b> · confidence <StatusBadge status={res.confidence} /> <span className="text-xs text-muted-foreground">{res.ms} ms</span></div>
            <div className="grid grid-cols-5 gap-2">
              {res.matches.map((m) => (
                <div key={m.frame} className="text-center text-[10px]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={apiUrl(`/spatial/reconstructions/${r}/frames/${m.frame}`)} alt={m.frame} className="aspect-video w-full rounded border object-cover" onError={(e) => (e.currentTarget.style.display = "none")} />
                  {m.similarity.toFixed(3)}
                </div>
              ))}
            </div>
            <div className="text-xs text-muted-foreground">{res.method}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function AILabPage() {
  const models = useApi<ModelStatus>("/ai/models");
  return (
    <>
      <PageHeader title="AI Lab" description="Additional open-source models running locally on this machine. Licences verified before integration; nothing is sent to a third-party service." />
      <Card className="mb-6 shadow-none">
        <CardContent className="overflow-auto p-0">
          <table className="w-full text-sm">
            <thead className="bg-muted text-left text-xs text-muted-foreground"><tr><th className="px-4 py-2">Model</th><th className="px-4 py-2">Task</th><th className="px-4 py-2">Licence</th><th className="px-4 py-2">State</th></tr></thead>
            <tbody>
              {[["OWLv2 + YOLO11m", "facility detection (AI Studio)", "Apache-2.0 / AGPL-3.0", "pipeline"], ["Depth Anything V2 Metric", "depth for measurements", "Apache-2.0 (Small, per upstream)", "pipeline"], ["COLMAP + Brush", "video → 3D Gaussian splat", "BSD / Apache-2.0", "pipeline"]].map((r) => (
                <tr key={r[0]} className="border-b"><td className="px-4 py-2 font-medium">{r[0]}</td><td className="px-4 py-2">{r[1]}</td><td className="px-4 py-2">{r[2]}</td><td className="px-4 py-2 text-xs text-muted-foreground">{r[3]}</td></tr>
              ))}
              {models.data && Object.entries(models.data).map(([k, m]) => (
                <tr key={k} className="border-b"><td className="px-4 py-2 font-medium">{m.hf_id}</td><td className="px-4 py-2">{m.task}</td><td className="px-4 py-2">{m.license}</td>
                  <td className="px-4 py-2 text-xs">{m.loaded ? <span className="flex items-center gap-1 text-emerald-700"><Cpu className="size-3" /> loaded on {m.device}</span> : <span className="text-muted-foreground">loads on first use</span>}</td></tr>
              ))}
            </tbody>
          </table>
          {models.error && <div className="p-4"><ErrorState message={models.error} onRetry={models.reload} /></div>}
        </CardContent>
      </Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <FlorenceDemo />
        <div className="space-y-6"><WhisperDemo /><LocalizeDemo /></div>
      </div>
    </>
  );
}
