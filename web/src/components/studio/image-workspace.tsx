"use client";
import { useRef, useState } from "react";
import { ImageUp, Loader2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { apiUpload, apiUrl } from "@/lib/api";
import { useApi } from "@/hooks/use-api";
import type { ImageResult, LibraryItem, MeasurementReport } from "@/lib/types";
import { BBoxOverlay } from "@/components/app/bbox-overlay";
import { Empty, ErrorState, Loading, ProvenanceBadge, StatusBadge, fmt, timeAgo } from "@/components/app/common";
import { MeasurePanel, SignReader } from "./measure-panel";

type Detail = { item: LibraryItem; result: ImageResult; measurement: MeasurementReport | null };

export function ImageWorkspace({ initialId }: { initialId?: string | null }) {
  const lib = useApi<{ items: LibraryItem[] }>("/platform/library/images");
  const [selId, setSelId] = useState<string | null>(initialId || null);
  const [upPct, setUpPct] = useState<number | null>(null);
  const [upErr, setUpErr] = useState<string | null>(null);
  const [selDet, setSelDet] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const id = selId ?? lib.data?.items[0]?.request_id ?? null;
  const d = useApi<Detail>(id ? `/platform/library/images/${id}` : null);
  const detail = d.loading ? null : d.data;
  const loadErr = d.error;
  function select(x: string) { setSelId(x); setSelDet(null); }

  async function upload(f: File) {
    setUpErr(null); setUpPct(0);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await apiUpload<ImageResult>("/vision/analyze-image", fd, (p) => setUpPct(p));
      setUpPct(null);
      lib.reload();
      select(r.request_id);
    } catch (e) {
      setUpPct(null);
      setUpErr((e as Error).message);
    }
  }

  const det = detail?.result.detections.find((d) => d.id === selDet);
  return (
    <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
      <div className="space-y-4">
        <Card className="shadow-none">
          <CardContent className="p-4">
            <button onClick={() => fileRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) upload(f); }}
              className="flex w-full flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 text-center hover:bg-muted">
              <ImageUp className="size-6 text-primary" />
              <div className="text-sm font-medium">Upload a campus photograph</div>
              <div className="text-xs text-muted-foreground">JPEG / PNG / HEIC-converted, ≤ 25 MB. Runs OWLv2 + YOLO11 detection.</div>
            </button>
            <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
            {upPct !== null && (
              <div className="mt-3 space-y-1 text-xs text-muted-foreground">
                {upPct < 1 ? <>Uploading {(upPct * 100).toFixed(0)}%</> : <span className="flex items-center gap-1"><Loader2 className="size-3 animate-spin" /> Detecting facilities…</span>}
                <Progress value={upPct * 100} />
              </div>
            )}
            {upErr && <div className="mt-3"><ErrorState message={upErr} /></div>}
          </CardContent>
        </Card>
        <div>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Analyses</div>
          {lib.loading && <Loading />}
          {lib.error && <ErrorState message={lib.error} onRetry={lib.reload} />}
          <div className="space-y-2">
            {lib.data?.items.map((it) => (
              <button key={it.request_id} onClick={() => select(it.request_id)}
                className={`flex w-full gap-3 rounded-lg border p-2 text-left transition ${id === it.request_id ? "border-primary bg-accent" : "bg-card hover:bg-muted"}`}>
                {it.annotated_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={apiUrl(it.annotated_url)} alt="" className="size-14 shrink-0 rounded object-cover" />
                )}
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">{it.image_name}</div>
                  <div className="text-xs text-muted-foreground">{it.detections} detections · {timeAgo(it.created)}</div>
                  <div className="mt-1"><ProvenanceBadge p={it.provenance} /></div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="min-w-0 space-y-6">
        {loadErr && <ErrorState message={loadErr} />}
        {!id && !lib.loading && <Empty title="No analyses yet">Upload a photograph to run facility detection.</Empty>}
        {id && !detail && !loadErr && <Loading label="Loading analysis…" />}
        {detail && (
          <>
            <Card className="shadow-none">
              <CardHeader>
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <CardTitle className="text-base">{detail.item.image_name}</CardTitle>
                    <CardDescription>
                      {detail.result.detections.length} detections · model {detail.result.model.key} ({detail.result.model.version}) on {detail.result.model.device} · {fmt(detail.result.timing.inference_ms, "ms", 0)} inference
                    </CardDescription>
                  </div>
                  <ProvenanceBadge p={detail.item.provenance} />
                </div>
              </CardHeader>
              <CardContent className="grid gap-4 xl:grid-cols-[1fr_320px]">
                <BBoxOverlay src={apiUrl(detail.item.source_url || detail.item.annotated_url)} detections={detail.result.detections} selected={selDet} onSelect={setSelDet} />
                <div className="space-y-3">
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(detail.result.counts_by_class).map(([k, v]) => <Badge key={k} variant="secondary">{k.replaceAll("_", " ")} · {v}</Badge>)}
                  </div>
                  {detail.result.findings.map((f) => (
                    <div key={f.code} className={`rounded-md border p-2 text-xs ${f.severity === "warning" ? "border-amber-200 bg-amber-50 text-amber-900" : "bg-muted"}`}>
                      <div className="font-semibold">{f.code.replaceAll("_", " ")}</div>{f.message}
                    </div>
                  ))}
                  {det ? (
                    <div className="rounded-md border p-3 text-sm">
                      <div className="mb-2 flex items-center justify-between font-medium capitalize">{det.class_name.replaceAll("_", " ")}<span className="font-mono text-xs">{(det.confidence * 100).toFixed(1)}%</span></div>
                      <div className="space-y-1 text-xs">
                        {Object.entries(det.attributes).map(([k, a]) => (
                          <div key={k} className="flex items-start justify-between gap-2">
                            <span className="text-muted-foreground">{k.replaceAll("_", " ")}</span>
                            <span className="text-right">{typeof a.value === "object" && a.value !== null ? JSON.stringify(a.value).slice(0, 40) : fmt(a.value)} <StatusBadge status={a.status} className="ml-1" /></span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : <div className="text-xs text-muted-foreground">Click a box to inspect its accessibility attributes.</div>}
                </div>
              </CardContent>
            </Card>
            {detail.item.has_source
              ? <MeasurePanel key={detail.item.request_id} result={detail.result} initial={detail.measurement} />
              : <ErrorState message="This older analysis has no stored source image — re-upload the photo to measure it." />}
            <SignReader requestId={detail.item.request_id}
              hasSigns={detail.item.has_source && detail.result.detections.some((d) => ["signboard", "accessibility_symbol", "entrance_exit"].includes(d.class_name))} />
          </>
        )}
      </div>
    </div>
  );
}
