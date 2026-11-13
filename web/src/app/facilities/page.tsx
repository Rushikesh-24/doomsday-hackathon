"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { Check, Search, X, AlertCircle, Wrench } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useApi } from "@/hooks/use-api";
import { apiSend, apiUrl } from "@/lib/api";
import type { Provenance } from "@/lib/types";
import { CLASS_COLORS, Empty, ErrorState, Loading, PageHeader, ProvenanceBadge, StatusBadge, fmt, timeAgo } from "@/components/app/common";

type Fac = {
  id: string; kind: string; class_name: string; confidence: number | null; source_name: string | null; source_request_id?: string;
  reconstruction_id?: string; annotated_url?: string; source_url?: string | null; provenance: Provenance; attributes?: Record<string, unknown>;
  measurements: { name: string; value: unknown; unit?: string; uncertainty?: number; status: string; provisional_value?: number }[];
  review: { status: string; note?: string; reviewer?: string; created: string } | null;
  position_local?: number[]; registered?: boolean; observations?: number; reason?: string; bbox?: { x1n: number; y1n: number; x2n: number; y2n: number };
};

export default function FacilitiesPage() {
  const [q, setQ] = useState("");
  const [cls, setCls] = useState("");
  const [status, setStatus] = useState("");
  const qs = new URLSearchParams({ ...(q && { q }), ...(cls && { class: cls }), ...(status && { status }) }).toString();
  const { data, error, loading, reload } = useApi<{ items: Fac[]; total: number }>(`/platform/facilities${qs ? `?${qs}` : ""}`);
  const all = useApi<{ items: Fac[] }>("/platform/facilities");
  const classes = useMemo(() => [...new Set((all.data?.items || []).map((f) => f.class_name))].sort(), [all.data]);
  const [sel, setSel] = useState<string | null>(null);
  const f = data?.items.find((x) => x.id === sel) || null;
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const history = useApi<{ items: { status: string; note?: string; created: string; reviewer?: string }[] }>(f ? `/platform/facilities/${f.id}/reviews` : null);

  async function review(s: string) {
    if (!f) return;
    setBusy(true); setErr(null);
    try {
      await apiSend(`/platform/facilities/${f.id}/review`, { status: s, note: note || null, reviewer: "web user", kind: f.kind });
      setNote(""); reload(); history.reload();
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }

  return (
    <>
      <PageHeader title="Facility Intelligence" description="Every detected facility across photos and 3D reconstructions, with its measurements and a human review trail. Reviews are stored; detections are never edited." />
      <div className="mb-4 flex flex-wrap gap-2">
        <div className="relative"><Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" /><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search class, file or id" className="w-64 pl-8" /></div>
        <select value={cls} onChange={(e) => setCls(e.target.value)} className="h-9 rounded-md border bg-card px-2 text-sm">
          <option value="">All classes</option>{classes.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)} className="h-9 rounded-md border bg-card px-2 text-sm">
          <option value="">Any review status</option>{["unreviewed", "verified", "needs_review", "rejected", "fixed"].map((s) => <option key={s} value={s}>{s.replace("_", " ")}</option>)}
        </select>
        <span className="self-center text-sm text-muted-foreground">{data?.total ?? "…"} facilities</span>
      </div>
      {error && <ErrorState message={error} onRetry={reload} />}
      <div className="grid gap-4 lg:grid-cols-[1fr_420px]">
        <Card className="overflow-hidden p-0 shadow-none">
          {loading && !data ? <Loading /> : !data?.items.length ? <div className="p-6"><Empty title="No facilities match">Change the filters or analyse more photos.</Empty></div> : (
            <div className="max-h-[70vh] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-muted text-left text-xs text-muted-foreground">
                  <tr><th className="px-3 py-2">Facility</th><th className="px-3 py-2">Source</th><th className="px-3 py-2">Confidence</th><th className="px-3 py-2">Estimated</th><th className="px-3 py-2">Review</th></tr>
                </thead>
                <tbody>
                  {data.items.map((x) => (
                    <tr key={x.id} onClick={() => setSel(x.id)} className={`cursor-pointer border-b ${sel === x.id ? "bg-accent" : "hover:bg-muted/60"}`}>
                      <td className="px-3 py-2"><span className="flex items-center gap-2 capitalize"><span className="size-2 rounded-full" style={{ background: CLASS_COLORS[x.class_name] }} />{x.class_name.replaceAll("_", " ")}</span></td>
                      <td className="max-w-48 truncate px-3 py-2 text-xs text-muted-foreground">{x.kind === "reconstruction_facility" ? "3D · " : ""}{x.source_name}</td>
                      <td className="px-3 py-2 tabular-nums">{fmt(x.confidence)}</td>
                      <td className="px-3 py-2 text-xs">{x.measurements.filter((m) => m.status === "ESTIMATED").length}/{x.measurements.length}</td>
                      <td className="px-3 py-2">{x.review ? <StatusBadge status={x.review.status} /> : <span className="text-xs text-muted-foreground">unreviewed</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
        <div>
          {!f ? <Empty title="Select a facility">Inspect its evidence, measurements and review history.</Empty> : (
            <Card className="shadow-none">
              <CardHeader>
                <CardTitle className="flex items-center justify-between text-base capitalize">{f.class_name.replaceAll("_", " ")}<ProvenanceBadge p={f.provenance} /></CardTitle>
                <div className="font-mono text-xs text-muted-foreground">{f.id}</div>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                {(f.source_url || f.annotated_url) && f.bbox && (
                  <div className="relative overflow-hidden rounded-md border">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={apiUrl((f.source_url || f.annotated_url)!)} alt="" className="w-full" />
                    <div className="absolute border-[3px]" style={{ borderColor: CLASS_COLORS[f.class_name], left: `${f.bbox.x1n * 100}%`, top: `${f.bbox.y1n * 100}%`, width: `${(f.bbox.x2n - f.bbox.x1n) * 100}%`, height: `${(f.bbox.y2n - f.bbox.y1n) * 100}%` }} />
                  </div>
                )}
                {f.kind === "reconstruction_facility" && (
                  <div className="rounded-md bg-muted p-3 text-xs">
                    {f.registered ? <>Placed in 3D at ({f.position_local?.map((v) => v.toFixed(2)).join(", ")}) from {f.observations} observations. <Link className="text-primary underline" href={`/explorer?id=${f.reconstruction_id}`}>View in 3D</Link></> : <>Not placed in 3D: {f.reason}</>}
                  </div>
                )}
                {!!f.measurements.length && (
                  <div className="space-y-1">
                    <div className="text-xs font-medium uppercase text-muted-foreground">Measurements</div>
                    {f.measurements.map((m) => (
                      <div key={m.name} className="flex items-center justify-between gap-2 text-xs">
                        <span>{m.name.replaceAll("_", " ")}</span>
                        <span className="flex items-center gap-2 tabular-nums">{m.status === "REQUIRES_CALIBRATION" ? "withheld" : <>{fmt(m.value, m.unit)}{m.uncertainty ? ` ± ${fmt(m.uncertainty)}` : ""}</>}<StatusBadge status={m.status} /></span>
                      </div>
                    ))}
                  </div>
                )}
                <div className="space-y-2 border-t pt-3">
                  <div className="text-xs font-medium uppercase text-muted-foreground">Review</div>
                  <Textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="Optional note (e.g. 'handrail present on left, verified on site 14 Nov')" rows={2} />
                  <div className="flex flex-wrap gap-2">
                    <Button size="sm" disabled={busy} onClick={() => review("verified")}><Check /> Verify</Button>
                    <Button size="sm" variant="outline" disabled={busy} onClick={() => review("needs_review")}><AlertCircle /> Needs review</Button>
                    <Button size="sm" variant="outline" disabled={busy} onClick={() => review("fixed")}><Wrench /> Issue fixed</Button>
                    <Button size="sm" variant="destructive" disabled={busy} onClick={() => review("rejected")}><X /> Reject detection</Button>
                  </div>
                  {err && <ErrorState message={err} />}
                  {history.data?.items.map((h, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs"><StatusBadge status={h.status} /><span className="text-muted-foreground">{timeAgo(h.created)}{h.note ? ` — ${h.note}` : ""}</span></div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
