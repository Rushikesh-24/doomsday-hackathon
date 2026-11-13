"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Download, Layers } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { useApi } from "@/hooks/use-api";
import { useScene } from "@/hooks/use-scene";
import { apiUrl } from "@/lib/api";
import { CLASS_COLORS, Empty, ErrorState, Loading, PageHeader, StatusBadge, fmt } from "@/components/app/common";

const SceneViewer = dynamic(() => import("@/components/three/scene-viewer"), { ssr: false, loading: () => <Loading label="Starting 3D viewer…" /> });

type ReconRow = { id: string; video_id: string; status: string; created: string; scale_status: string | null; gaussians: number | null; registered_frames: number | null };

function Explorer() {
  const sp = useSearchParams();
  const list = useApi<{ reconstructions: ReconRow[] }>("/spatial/reconstructions");
  const [chosen, setRid] = useState<string | null>(sp.get("id"));
  const rows = list.data?.reconstructions || [];
  const rid = chosen ?? (rows.find((r) => r.status.startsWith("completed")) || rows[0])?.id ?? null;
  const s = useScene(rid);
  const [layers, setLayers] = useState({ splat: true, points: false, trajectory: true, facilities: true });
  const [pick, setPick] = useState<string | null>(null);
  const m = s.recon?.metadata;
  const picked = s.facilities?.find((f) => f.facility_id === pick);
  return (
    <>
      <PageHeader title="3D Explorer" description="Reconstructed environments from walkthrough videos: Gaussian splat, COLMAP point cloud, camera trajectory and spatially registered facilities."
        actions={
          <div className="flex items-center gap-2">
            <select value={rid || ""} onChange={(e) => { setRid(e.target.value); setPick(null); }} className="h-9 rounded-md border bg-card px-2 text-sm">
              {list.data?.reconstructions.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status}{r.gaussians ? ` · ${r.gaussians.toLocaleString()} splats` : ""}</option>)}
            </select>
            <Link href="/studio?tab=video"><Button variant="outline">New reconstruction</Button></Link>
          </div>
        } />
      {list.error && <ErrorState message={list.error} onRetry={list.reload} />}
      {list.data && !list.data.reconstructions.length && <Empty title="No reconstructions yet">Upload a walkthrough video in AI Studio to build one.</Empty>}
      {s.loading && <Loading label="Loading reconstruction…" />}
      {s.error && <ErrorState message={s.error} />}
      {s.scene && (
        <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
          <Card className="overflow-hidden p-0 shadow-none">
            <div className="flex flex-wrap items-center gap-4 border-b px-4 py-2 text-sm">
              <Layers className="size-4 text-muted-foreground" />
              {(["splat", "points", "trajectory", "facilities"] as const).map((k) => (
                <label key={k} className="flex items-center gap-2 capitalize">
                  <Switch checked={layers[k]} onCheckedChange={(v: boolean) => setLayers({ ...layers, [k]: v })} /> {k === "points" ? "point cloud" : k}
                </label>
              ))}
              <span className="ml-auto"><StatusBadge status={s.recon?.alignment?.scale_status} /></span>
            </div>
            <div className="h-[620px]"><SceneViewer data={s.scene} layers={layers} highlight={pick} onPick={setPick} /></div>
            <div className="border-t px-4 py-2 text-xs text-muted-foreground">
              Drag to orbit · scroll to zoom · green dot = start of walk. Coordinates are gravity-aligned; units are {s.recon?.alignment?.scale_status === "relative" ? "relative (no metric scale reference in this video)" : "metres (" + s.recon?.alignment?.scale_status + ")"}.
            </div>
          </Card>
          <div className="space-y-4">
            {picked && (
              <Card className="shadow-none">
                <CardHeader className="pb-2"><CardTitle className="text-sm capitalize">{picked.class_name} · {picked.facility_id.split("-").pop()}</CardTitle></CardHeader>
                <CardContent className="space-y-1 text-xs">
                  <div>registration confidence {fmt(picked.registration_confidence)} · {picked.support_points} support points</div>
                  <div>{picked.n_observations} observations · merged from {picked.merged_from} track(s)</div>
                  <div>position (local) {picked.position_local?.map((v: number) => v.toFixed(2)).join(", ")}</div>
                  {picked.measurements?.map((mm: { name: string; value: unknown; unit?: string; status: string }) => <div key={mm.name} className="flex justify-between"><span>{mm.name}</span><span>{fmt(mm.value, mm.unit)} <StatusBadge status={mm.status} /></span></div>)}
                </CardContent>
              </Card>
            )}
            <Card className="shadow-none">
              <CardHeader className="pb-2"><CardTitle className="text-sm">Reconstruction metrics</CardTitle></CardHeader>
              <CardContent className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
                {[
                  ["Capture quality", `${m?.capture_quality?.suitability} (${m?.capture_quality?.suitability_score})`],
                  ["Registered frames", `${m?.registration?.registered_primary}/${m?.registration?.frames}`],
                  ["Pose completeness", fmt(m?.registration?.pose_completeness)],
                  ["Reprojection error", fmt(m?.registration?.mean_reprojection_error_px, "px")],
                  ["Sparse points", m?.registration?.points?.toLocaleString()],
                  ["Splat backend", m?.gaussian_splat?.backend],
                  ["Gaussians", m?.gaussian_splat?.num_gaussians?.toLocaleString()],
                  ["Held-out PSNR", fmt(m?.gaussian_splat?.test_psnr, "dB")],
                  ["Held-out SSIM", fmt(m?.gaussian_splat?.test_ssim)],
                  ["Facilities in 3D", `${m?.facilities?.registered}/${m?.facilities?.total}`],
                ].map(([k, v]) => <div key={k}><div className="text-muted-foreground">{k}</div><div className="font-medium">{v ?? "—"}</div></div>)}
              </CardContent>
            </Card>
            <Card className="shadow-none">
              <CardHeader className="pb-2"><CardTitle className="text-sm">Facilities ({s.facilities?.length})</CardTitle></CardHeader>
              <CardContent className="max-h-64 space-y-1 overflow-auto text-xs">
                {s.facilities?.map((f) => (
                  <button key={f.facility_id} onClick={() => setPick(f.facility_id)} disabled={!f.position_sfm}
                    className={`flex w-full items-center justify-between rounded px-2 py-1 text-left ${pick === f.facility_id ? "bg-accent" : "hover:bg-muted"} disabled:opacity-50`}>
                    <span className="flex items-center gap-2"><span className="size-2 rounded-full" style={{ background: CLASS_COLORS[f.class_name] }} />{f.class_name}</span>
                    <span className="text-muted-foreground">{f.position_sfm ? `conf ${fmt(f.registration_confidence)}` : "not in 3D"}</span>
                  </button>
                ))}
              </CardContent>
            </Card>
            <Card className="shadow-none">
              <CardHeader className="pb-2"><CardTitle className="text-sm">Assets</CardTitle></CardHeader>
              <CardContent className="space-y-1 text-xs">
                {s.assets && Object.entries(s.assets).map(([k, a]) => (
                  <a key={k} href={apiUrl(a.url)} className="flex items-center justify-between rounded px-2 py-1 hover:bg-muted" title={a.description}>
                    <span className="flex items-center gap-2"><Download className="size-3" />{k}</span><span className="text-muted-foreground">{(a.bytes / 1e6).toFixed(2)} MB</span>
                  </a>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}

export default function Page() {
  return <Suspense fallback={<Loading />}><Explorer /></Suspense>;
}
