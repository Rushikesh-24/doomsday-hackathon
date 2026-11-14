"use client";
// AI Studio processing queue: every walkthrough-video job with live progress and Pause / Resume / Stop / Start / Delete.
import Link from "next/link";
import { useState } from "react";
import { Box, ChevronDown, Loader2, Pause, Play, RotateCcw, Square, Trash2 } from "lucide-react";
import { usePoll } from "@/hooks/use-api";
import { apiSend, apiUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ErrorState, timeAgo } from "@/components/app/common";
import { JobMonitor } from "@/components/studio/video-panel";

export type JobRow = {
  id: string; video_id: string; video_name: string | null; status: string; preset: string; created: string; updated: string;
  overall: number | null; stage: string | null; message: string | null; error: string | null; duration_s: number | null; thumbnail: string;
};

const ACTIVE = ["queued", "running", "paused"];
const TONE: Record<string, string> = {
  running: "bg-sky-100 text-sky-700", queued: "bg-slate-100 text-slate-600", paused: "bg-amber-100 text-amber-800",
  stopped: "bg-slate-200 text-slate-700", interrupted: "bg-amber-100 text-amber-800", failed: "bg-rose-100 text-rose-700",
  completed: "bg-emerald-100 text-emerald-700", completed_with_failures: "bg-amber-100 text-amber-800",
};
const STAGE: Record<string, string> = { quality: "Checking capture", frames: "Picking keyframes", tracking: "Finding facilities", sfm: "Camera poses", reconstruction: "Training 3D splat", alignment: "Scale & gravity", registration: "Placing facilities", export: "Exporting" };

function Thumb({ src }: { src: string }) {
  const [bad, setBad] = useState(false);
  return bad ? <div className="grid size-full place-items-center bg-muted"><Box className="size-5 text-muted-foreground" /></div>
    // eslint-disable-next-line @next/next/no-img-element -- backend-served preview, not a static asset
    : <img src={apiUrl(src)} alt="" className="size-full object-cover" onError={() => setBad(true)} />;
}

export function JobQueue({ highlight }: { highlight?: string | null }) {
  const q = usePoll<{ jobs: JobRow[] }>("/spatial/jobs", (d) => !d || d.jobs.some((j) => ACTIVE.includes(j.status)), 2500);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(highlight ?? null);
  const [filter, setFilter] = useState<"all" | "active" | "done">("all");
  const jobs = q.data?.jobs ?? [];
  const shown = jobs.filter((j) => filter === "all" || (filter === "active" ? ACTIVE.includes(j.status) || ["stopped", "interrupted", "failed"].includes(j.status) : j.status.startsWith("completed")));
  const counts = { active: jobs.filter((j) => ACTIVE.includes(j.status)).length, total: jobs.length };

  async function act(j: JobRow, action: "pause" | "resume" | "stop" | "start", body?: unknown) {
    if (action === "stop" && !confirm(`Stop processing "${j.video_name ?? j.id}"? Finished stages are kept, so you can start it again later.`)) return;
    setBusy(`${j.id}:${action}`); setErr(null);
    try { await apiSend(`/spatial/jobs/${j.id}/${action}`, body); q.reload(); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }
  async function remove(j: JobRow) {
    const name = j.video_name ?? j.id;
    const active = ACTIVE.includes(j.status);
    if (!confirm(`Delete "${name}" (${j.status.replaceAll("_", " ")}) permanently?\n\n${active ? "Processing is stopped first. " : ""}All its outputs (3D model, places, plan, score) are removed${j.status.startsWith("completed") ? " and the map disappears from phones on their next sync" : ""}. The uploaded video is kept so you can process it again. This cannot be undone.`)) return;
    setBusy(`${j.id}:delete`); setErr(null);
    try { await apiSend(`/spatial/jobs/${j.id}`, undefined, "DELETE"); if (open === j.id) setOpen(null); q.reload(); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(null); }
  }
  const Btn = ({ j, action, icon, label, tone = "", body }: { j: JobRow; action: "pause" | "resume" | "stop" | "start"; icon: React.ReactNode; label: string; tone?: string; body?: unknown }) => (
    <button onClick={(e) => { e.stopPropagation(); act(j, action, body); }} disabled={!!busy} title={label} aria-label={`${label} ${j.video_name ?? j.id}`}
      className={cn("flex items-center gap-1 rounded-full border bg-white px-2.5 py-1 text-[11px] font-medium transition hover:bg-muted disabled:opacity-50", tone)}>
      {busy === `${j.id}:${action}` ? <Loader2 className="size-3 animate-spin" /> : icon}{label}
    </button>
  );

  return (
    <div className="surface p-5">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div>
          <div className="eyebrow">Processing queue</div>
          <div className="font-display text-lg font-semibold">{counts.active ? `${counts.active} active · ` : ""}{counts.total} video job{counts.total === 1 ? "" : "s"}</div>
        </div>
        <div className="ml-auto inline-flex rounded-full bg-muted p-1 text-xs">
          {(["all", "active", "done"] as const).map((f) => (
            <button key={f} onClick={() => setFilter(f)} className={cn("rounded-full px-3 py-1 capitalize", filter === f ? "bg-white font-medium shadow-sm" : "text-muted-foreground")}>{f === "active" ? "In progress" : f}</button>
          ))}
        </div>
      </div>
      {q.error && <ErrorState message={q.error} onRetry={q.reload} />}
      {err && <div className="mb-3"><ErrorState message={err} /></div>}
      {!q.data && !q.error && <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Loading jobs…</div>}
      {q.data && !shown.length && <div className="rounded-2xl border border-dashed p-6 text-center text-sm text-muted-foreground">No jobs here yet. Upload a walkthrough video to start one.</div>}
      <div className="space-y-2">
        {shown.map((j) => {
          const pct = Math.round((j.overall ?? (j.status.startsWith("completed") ? 1 : 0)) * 100);
          const live = ["running", "queued"].includes(j.status);
          return (
            <div key={j.id} className={cn("rounded-2xl border transition", open === j.id ? "border-primary/30 bg-accent/30" : "hover:bg-muted/40")}>
              <div className="flex cursor-pointer items-center gap-3 p-3" onClick={() => setOpen(open === j.id ? null : j.id)}>
                <div className="size-14 shrink-0 overflow-hidden rounded-xl bg-muted"><Thumb src={j.thumbnail} /></div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-semibold">{j.video_name ?? j.video_id}</span>
                    <span className={cn("shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide", TONE[j.status] ?? "bg-muted")}>{j.status.replaceAll("_", " ")}</span>
                  </div>
                  <div className="mt-0.5 truncate text-xs text-muted-foreground">
                    {j.preset} · {j.duration_s ? `${j.duration_s.toFixed(0)} s video · ` : ""}{live || j.status === "paused" ? `${STAGE[j.stage ?? ""] ?? "Waiting"}${j.message ? ` — ${j.message}` : ""}` : j.error ?? `updated ${timeAgo(j.updated)}`}
                  </div>
                  <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
                    <div className={cn("h-full rounded-full transition-all", j.status === "paused" ? "bg-amber-400" : j.status === "failed" ? "bg-rose-400" : "bg-gradient-to-r from-violet-500 to-teal-400", live && "animate-pulse")} style={{ width: `${pct}%` }} />
                  </div>
                </div>
                <div className="w-10 shrink-0 text-right text-xs font-semibold tabular-nums">{pct}%</div>
                <div className="flex shrink-0 flex-wrap justify-end gap-1">
                  {j.status === "running" && <Btn j={j} action="pause" icon={<Pause className="size-3" />} label="Pause" />}
                  {j.status === "paused" && <Btn j={j} action="resume" icon={<Play className="size-3" />} label="Resume" tone="text-emerald-700" />}
                  {ACTIVE.includes(j.status) && <Btn j={j} action="stop" icon={<Square className="size-3" />} label="Stop" tone="text-rose-600" />}
                  {["stopped", "interrupted", "failed", "completed_with_failures"].includes(j.status) && <Btn j={j} action="start" icon={<Play className="size-3" />} label={j.status === "stopped" ? "Start" : "Resume"} tone="text-primary" />}
                  {j.status.startsWith("completed") && (
                    <Link href={`/explorer?id=${j.id}`} onClick={(e) => e.stopPropagation()} className="flex items-center gap-1 rounded-full bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground"><Box className="size-3" /> Open</Link>
                  )}
                  {j.status === "completed" && <Btn j={j} action="start" icon={<RotateCcw className="size-3" />} label="Re-run HQ" body={{ preset: "balanced", force_stages: ["reconstruction", "export"] }} />}
                  <button onClick={(e) => { e.stopPropagation(); remove(j); }} disabled={!!busy} title="Delete" aria-label={`Delete ${j.video_name ?? j.id}`}
                    className="flex items-center gap-1 rounded-full border bg-white px-2 py-1 text-[11px] font-medium text-rose-600 transition hover:bg-rose-50 disabled:opacity-50">
                    {busy === `${j.id}:delete` ? <Loader2 className="size-3 animate-spin" /> : <Trash2 className="size-3" />}
                  </button>
                </div>
                <ChevronDown className={cn("size-4 shrink-0 text-muted-foreground transition", open === j.id && "rotate-180")} />
              </div>
              {open === j.id && <div className="border-t p-3"><JobMonitor jobId={j.id} /></div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
