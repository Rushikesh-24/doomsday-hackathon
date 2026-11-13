"use client";
import Link from "next/link";
import { Activity, AlertTriangle, ArrowRight, Box, Camera, ClipboardCheck, Ruler, ScanSearch, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useApi } from "@/hooks/use-api";
import { CLASS_COLORS, ErrorState, Loading, PageHeader, Stat, StatusBadge, timeAgo } from "@/components/app/common";

type Overview = {
  images: { distinct: number; analyses_total: number; real_samples: number };
  detections_by_class: Record<string, number>;
  measurements: { runs_total: number; latest_by_status: Record<string, number> };
  spatial: { jobs: number; by_status: Record<string, number>; facilities_3d: number; registered_3d: number };
  compliance: { outcomes: Record<string, number>; gaps: Record<string, number>; issues: number };
  reviews: Record<string, number>;
  activity: { kind: string; title: string; time: string; ref: string; detail?: string }[];
};

function Bars({ data, colors }: { data: Record<string, number>; colors?: Record<string, string> }) {
  const max = Math.max(1, ...Object.values(data));
  return (
    <div className="space-y-2">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="grid grid-cols-[140px_1fr_40px] items-center gap-3 text-sm">
          <span className="truncate capitalize text-muted-foreground">{k.replaceAll("_", " ").toLowerCase()}</span>
          <div className="h-2 rounded-full bg-muted">
            <div className="h-2 rounded-full" style={{ width: `${(v / max) * 100}%`, background: colors?.[k] || "var(--color-primary)" }} />
          </div>
          <span className="text-right tabular-nums">{v}</span>
        </div>
      ))}
    </div>
  );
}

const KIND: Record<string, { icon: typeof Camera; label: string; href: (r: string) => string }> = {
  image_analysis: { icon: Camera, label: "Photo analysed", href: (r) => `/studio?id=${r}` },
  measurement: { icon: Ruler, label: "Measurements", href: () => "/studio" },
  spatial_job: { icon: Box, label: "3D reconstruction", href: (r) => `/explorer?id=${r}` },
};

export default function OverviewPage() {
  const { data, error, loading, reload } = useApi<Overview>("/platform/overview");
  return (
    <>
      <PageHeader title="Overview" description="Live state of the campus accessibility intelligence platform. Every number below is computed from stored AI results (repeated analyses of the same photo count once)."
        actions={<Link href="/studio"><Button><Wand2 /> Open AI Studio</Button></Link>} />
      {loading && <Loading />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Photos analysed" value={data.images.distinct} hint={`${data.images.real_samples} real CC-licensed samples · ${data.images.analyses_total} runs`} icon={<Camera className="size-4" />} />
            <Stat label="Detections" value={Object.values(data.detections_by_class).reduce((a, b) => a + b, 0)} hint="all classes incl. people, latest analysis per photo" icon={<ScanSearch className="size-4" />} />
            <Stat label="3D reconstructions" value={data.spatial.jobs} hint={`${data.spatial.registered_3d} of ${data.spatial.facilities_3d} facilities placed in 3D`} icon={<Box className="size-4" />} />
            <Stat label="Possible issues" value={data.compliance.issues} hint="screening, not certification" icon={<AlertTriangle className="size-4" />} />
          </div>
          <div className="grid gap-6 lg:grid-cols-3">
            <Card className="shadow-none lg:col-span-2">
              <CardHeader><CardTitle className="text-base">Detections by class</CardTitle><CardDescription>OWLv2 + YOLO11 ensemble on the latest analysis of each photo</CardDescription></CardHeader>
              <CardContent><Bars data={data.detections_by_class} colors={CLASS_COLORS} /></CardContent>
            </Card>
            <Card className="shadow-none">
              <CardHeader><CardTitle className="flex items-center gap-2 text-base"><Activity className="size-4" /> Recent activity</CardTitle></CardHeader>
              <CardContent className="space-y-1">
                {data.activity.map((a, i) => {
                  const k = KIND[a.kind];
                  const Icon = k?.icon || Activity;
                  return (
                    <Link key={i} href={k?.href(a.ref) || "#"} className="flex items-center gap-3 rounded-md px-2 py-1.5 text-sm hover:bg-muted">
                      <Icon className="size-4 shrink-0 text-primary" />
                      <div className="min-w-0 flex-1"><div className="truncate">{a.title}</div><div className="text-xs text-muted-foreground">{k?.label} · {timeAgo(a.time)}</div></div>
                      {a.detail && <StatusBadge status={a.detail} />}
                    </Link>
                  );
                })}
              </CardContent>
            </Card>
          </div>
          <div className="grid gap-6 lg:grid-cols-3">
            <Card className="shadow-none">
              <CardHeader><CardTitle className="flex items-center gap-2 text-base"><Ruler className="size-4" /> Measurement status</CardTitle><CardDescription>Absolute sizes are withheld without a scale reference</CardDescription></CardHeader>
              <CardContent><Bars data={data.measurements.latest_by_status} /></CardContent>
            </Card>
            <Card className="shadow-none">
              <CardHeader><CardTitle className="flex items-center gap-2 text-base"><ClipboardCheck className="size-4" /> Screening outcomes</CardTitle><CardDescription>Rules applied to measured facilities</CardDescription></CardHeader>
              <CardContent><Bars data={data.compliance.outcomes} /></CardContent>
            </Card>
            <Card className="shadow-none">
              <CardHeader><CardTitle className="flex items-center gap-2 text-base"><AlertTriangle className="size-4" /> Accessibility insights</CardTitle><CardDescription>Scene-level findings from detection</CardDescription></CardHeader>
              <CardContent className="space-y-3">
                {Object.entries(data.compliance.gaps).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm">
                    <span className="capitalize text-amber-900">{k.replaceAll("_", " ").toLowerCase()}</span><span className="font-semibold tabular-nums text-amber-900">{v}</span>
                  </div>
                ))}
                <Link href="/compliance" className="flex items-center gap-1 text-sm font-medium text-primary hover:underline">Open compliance <ArrowRight className="size-3" /></Link>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
