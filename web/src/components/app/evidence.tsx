"use client";
// Review evidence for a facility found in a walkthrough video: the frames it was detected in, with the
// detection box, a close-up of the box, and every other sighting as a filmstrip (← / → to step through).
import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Film, MapPin } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { apiUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import { CLASS_COLORS, Loading, fmt } from "@/components/app/common";

type Obs = { frame: string; t: number | null; confidence: number | null; bbox_n: [number, number, number, number]; placed: boolean; reason?: string | null; url: string; width: number; height: number };
type Evidence = { class_name: string; registered: boolean; total: number; observations: Obs[] };

const clock = (t: number | null) => (t == null ? "?" : `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`);

function Boxed({ o, color, className, label }: { o: Obs; color: string; className?: string; label?: string }) {
  const [x1, y1, x2, y2] = o.bbox_n;
  return (
    <div className={cn("relative w-fit", className)}>
      {/* eslint-disable-next-line @next/next/no-img-element -- backend-served video frame */}
      <img src={apiUrl(o.url)} alt={`Video frame at ${clock(o.t)}`} className="block h-full w-auto" />
      <div className="absolute rounded-sm border-[3px] shadow-[0_0_0_9999px_rgba(15,23,42,0.28)]"
        style={{ borderColor: color, left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }}>
        {label && <span className="absolute -top-6 left-0 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-semibold text-white" style={{ background: color }}>{label}</span>}
      </div>
    </div>
  );
}

/** The detection box blown up, with some context around it. */
function CloseUp({ o }: { o: Obs }) {
  const [x1, y1, x2, y2] = o.bbox_n;
  const pad = 0.15;
  const bw = Math.min(1, (x2 - x1) * (1 + 2 * pad)), bh = Math.min(1, (y2 - y1) * (1 + 2 * pad));
  const cx = Math.max(0, Math.min(1 - bw, x1 - (x2 - x1) * pad)), cy = Math.max(0, Math.min(1 - bh, y1 - (y2 - y1) * pad));
  return (
    <div role="img" aria-label="Close-up of the detection" className="aspect-square w-full rounded-lg border bg-muted bg-no-repeat"
      style={{
        backgroundImage: `url(${apiUrl(o.url)})`,
        backgroundSize: `${100 / bw}% ${100 / bh}%`,
        backgroundPosition: `${bw >= 1 ? 0 : (cx / (1 - bw)) * 100}% ${bh >= 1 ? 0 : (cy / (1 - bh)) * 100}%`,
      }} />
  );
}

export function FacilityEvidence({ rid, fid }: { rid: string; fid: string }) {
  const ev = useApi<Evidence>(`/spatial/reconstructions/${rid}/facilities/${encodeURIComponent(fid)}/evidence?limit=24`);
  const [i, setI] = useState(0);
  const obs = ev.data?.observations ?? [];
  const o = obs[Math.min(i, obs.length - 1)];
  const color = CLASS_COLORS[ev.data?.class_name ?? ""] || "#7c3aed";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !obs.length) return;
      if (e.key === "ArrowRight") { setI((k) => (k + 1) % obs.length); e.preventDefault(); }
      if (e.key === "ArrowLeft") { setI((k) => (k - 1 + obs.length) % obs.length); e.preventDefault(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [obs.length]);

  if (ev.loading && !ev.data) return <Loading label="Loading video frames…" />;
  if (ev.error) return <div className="rounded-md bg-muted p-3 text-xs text-muted-foreground">No frames available: {ev.error}</div>;
  if (!o) return <div className="rounded-md bg-muted p-3 text-xs text-muted-foreground">This detection has no stored video frames.</div>;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 font-medium uppercase text-muted-foreground"><Film className="size-3.5" /> Seen in the video</span>
        <span className="tabular-nums text-muted-foreground">{Math.min(i, obs.length - 1) + 1} / {obs.length}{ev.data!.total > obs.length ? ` of ${ev.data!.total}` : ""}</span>
      </div>
      <div className="grid grid-cols-[1fr_128px] gap-2">
        <div className="relative flex h-80 items-center justify-center overflow-hidden rounded-lg border bg-slate-950">
          <Boxed o={o} color={color} className="h-full" label={`${ev.data!.class_name.replaceAll("_", " ")} ${fmt(o.confidence)}`} />
          {obs.length > 1 && (
            <>
              <button onClick={() => setI((k) => (k - 1 + obs.length) % obs.length)} aria-label="Previous frame" className="absolute left-1.5 top-1/2 grid size-8 -translate-y-1/2 place-items-center rounded-full bg-white/85 shadow hover:bg-white"><ChevronLeft className="size-4" /></button>
              <button onClick={() => setI((k) => (k + 1) % obs.length)} aria-label="Next frame" className="absolute right-1.5 top-1/2 grid size-8 -translate-y-1/2 place-items-center rounded-full bg-white/85 shadow hover:bg-white"><ChevronRight className="size-4" /></button>
            </>
          )}
        </div>
        <div className="space-y-2 text-[11px] leading-snug">
          <CloseUp o={o} />
          <div><span className="text-muted-foreground">Time</span> <b className="tabular-nums">{clock(o.t)}</b></div>
          <div><span className="text-muted-foreground">Confidence</span> <b className="tabular-nums">{fmt(o.confidence)}</b></div>
          <div className={cn("rounded px-1.5 py-1", o.placed ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-900")}>
            {o.placed ? <><MapPin className="mr-0.5 inline size-3" />Used for the 3D position</> : <>Not used in 3D{o.reason ? `: ${o.reason}` : ""}</>}
          </div>
        </div>
      </div>
      {obs.length > 1 && (
        <div className="flex gap-1.5 overflow-x-auto pb-1" role="listbox" aria-label="All sightings">
          {obs.map((x, k) => (
            <button key={x.frame} role="option" aria-selected={k === i} onClick={() => setI(k)} title={`${clock(x.t)} · confidence ${fmt(x.confidence)}`}
              className={cn("h-16 shrink-0 overflow-hidden rounded-md border-2 transition", k === i ? "border-primary" : "border-transparent opacity-70 hover:opacity-100")}>
              <Boxed o={x} color={color} className="h-full" />
            </button>
          ))}
        </div>
      )}
      <p className="text-[11px] text-muted-foreground">Use ← → to step through sightings, then mark the detection below (V verify · N needs review · R reject).</p>
    </div>
  );
}
