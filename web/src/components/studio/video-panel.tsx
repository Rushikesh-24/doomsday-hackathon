"use client";
import Link from "next/link";
import { useRef, useState } from "react";
import { Box, CheckCircle2, Circle, Clapperboard, Loader2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { apiSend, apiUpload } from "@/lib/api";
import { usePoll } from "@/hooks/use-api";
import type { JobProgress } from "@/lib/types";
import { ErrorState, StatusBadge } from "@/components/app/common";

const STAGES = ["quality", "frames", "tracking", "sfm", "reconstruction", "alignment", "registration", "export"];
const LABEL: Record<string, string> = { quality: "Capture quality", frames: "Keyframes", tracking: "Facility tracking", sfm: "Camera poses (COLMAP)", reconstruction: "Gaussian splat", alignment: "Scale & gravity", registration: "Facility → 3D", export: "Export assets" };

export function JobMonitor({ jobId }: { jobId: string }) {
  const { data, error } = usePoll<JobProgress>(`/spatial/jobs/${jobId}/progress`, (d) => !d || ["queued", "running"].includes(d.status), 2500);
  if (error) return <ErrorState message={error} />;
  if (!data) return null;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 text-sm">
        <StatusBadge status={data.status} /> <span className="font-mono text-xs text-muted-foreground">job {jobId}</span>
        <span className="ml-auto tabular-nums text-xs">{Math.round((data.overall || 0) * 100)}%</span>
      </div>
      <Progress value={(data.overall || 0) * 100} />
      <div className="grid gap-1.5 sm:grid-cols-2">
        {STAGES.map((s) => {
          const st = data.stages?.[s];
          const icon = st?.status === "done" ? <CheckCircle2 className="size-4 text-emerald-600" /> :
            st?.status === "running" ? <Loader2 className="size-4 animate-spin text-sky-600" /> :
            ["failed", "blocked"].includes(st?.status || "") ? <XCircle className="size-4 text-rose-600" /> : <Circle className="size-4 text-slate-300" />;
          return (
            <div key={s} className="flex items-start gap-2 rounded-md border px-2.5 py-1.5 text-xs">
              {icon}
              <div className="min-w-0">
                <div className="font-medium">{LABEL[s]} {st?.seconds != null && <span className="font-normal text-muted-foreground">· {st.seconds}s</span>}</div>
                {(st?.message || st?.error) && <div className={`truncate ${st?.error ? "text-rose-600" : "text-muted-foreground"}`}>{st?.error || st?.message}</div>}
              </div>
            </div>
          );
        })}
      </div>
      {["completed", "completed_with_failures"].includes(data.status) && (
        <Link href={`/explorer?id=${jobId}`}><Button><Box /> Open in 3D Explorer</Button></Link>
      )}
    </div>
  );
}

export function VideoPanel() {
  const ref = useRef<HTMLInputElement>(null);
  const [video, setVideo] = useState<{ video_id: string; metadata: Record<string, unknown> } | null>(null);
  const [pct, setPct] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [preset, setPreset] = useState("fast");
  const [job, setJob] = useState<string | null>(null);
  async function upload(f: File) {
    setErr(null); setPct(0); setJob(null);
    const fd = new FormData(); fd.append("file", f);
    try { setVideo(await apiUpload("/spatial/videos/upload", fd, setPct)); } catch (e) { setErr((e as Error).message); } finally { setPct(null); }
  }
  async function start() {
    setErr(null);
    try { setJob((await apiSend<{ job_id: string }>(`/spatial/videos/${video!.video_id}/analyze`, { preset })).job_id); }
    catch (e) { setErr((e as Error).message); }
  }
  const m = video?.metadata as Record<string, number | string> | undefined;
  return (
    <Card className="shadow-none">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base"><Clapperboard className="size-4" /> Walkthrough video → 3D</CardTitle>
        <CardDescription>Quality analysis, facility tracking, COLMAP camera poses, Gaussian-splat training (Brush) and facility registration. A fast run takes ~10–15 min on Apple M2.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <button onClick={() => ref.current?.click()} className="flex w-full flex-col items-center gap-1 rounded-lg border-2 border-dashed p-6 hover:bg-muted">
          <Clapperboard className="size-6 text-primary" />
          <span className="text-sm font-medium">Upload an MP4 / MOV walkthrough</span>
          <span className="text-xs text-muted-foreground">Walk slowly, phone upright, 30–60 s works best</span>
        </button>
        <input ref={ref} type="file" accept="video/mp4,video/quicktime" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        {pct !== null && <Progress value={pct * 100} />}
        {err && <ErrorState message={err} />}
        {video && m && (
          <div className="rounded-lg border p-3 text-sm">
            <div className="font-medium">{String(m.file)}</div>
            <div className="text-xs text-muted-foreground">{m.width}×{m.height} · {Number(m.fps).toFixed(1)} fps · {Number(m.duration_s).toFixed(1)} s · {String(m.codec)}</div>
            {!job && (
              <div className="mt-3 flex items-center gap-2">
                <select value={preset} onChange={(e) => setPreset(e.target.value)} className="h-8 rounded-md border bg-background px-2 text-sm">
                  <option value="fast">Fast preview</option><option value="balanced">Balanced</option><option value="high">High quality</option>
                </select>
                <Button onClick={start}>Start reconstruction</Button>
              </div>
            )}
          </div>
        )}
        {job && <JobMonitor jobId={job} />}
      </CardContent>
    </Card>
  );
}
