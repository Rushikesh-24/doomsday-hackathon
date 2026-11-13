"use client";
import Link from "next/link";
import { useState } from "react";
import { Box, Building2, Camera, Layers, Plus, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useApi } from "@/hooks/use-api";
import { apiSend, apiUrl } from "@/lib/api";
import type { Provenance } from "@/lib/types";
import { Empty, ErrorState, Loading, PageHeader, ProvenanceBadge } from "@/components/app/common";

type Asset = { kind: "image" | "reconstruction"; id: string; name: string; thumb?: string | null; status?: string; provenance?: Provenance };
type Building = { id: string; name: string; description?: string | null; floors: { id: string; name: string; level: number }[]; assets: { kind: string; id: string; floor_id: string | null }[] };
type Campus = { buildings: Building[]; assets: Asset[]; unassigned: Asset[] };

export default function CampusPage() {
  const { data, error, loading, reload } = useApi<Campus>("/platform/campus");
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [floorName, setFloorName] = useState<Record<string, string>>({});
  const [assign, setAssign] = useState<Record<string, string>>({});   // asset key -> "bid|fid"

  async function act(fn: () => Promise<unknown>) {
    setErr(null);
    try { await fn(); reload(); } catch (e) { setErr((e as Error).message); }
  }
  const byKey = new Map((data?.assets || []).map((a) => [`${a.kind}:${a.id}`, a]));

  return (
    <>
      <PageHeader title="Campus Explorer" description="Organise analysed photos and 3D reconstructions into buildings and floors. The structure is yours to define; assets come from real stored analyses." />
      {loading && <Loading />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {err && <div className="mb-4"><ErrorState message={err} /></div>}
      {data && (
        <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
          <div className="space-y-4">
            <Card className="shadow-none">
              <CardContent className="flex flex-wrap items-end gap-2 p-4">
                <div className="flex-1"><label className="text-xs text-muted-foreground">Building name</label><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Main Library" /></div>
                <div className="flex-1"><label className="text-xs text-muted-foreground">Description (optional)</label><Input value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="e.g. North entrance, 3 floors" /></div>
                <Button disabled={!name.trim()} onClick={() => act(async () => { await apiSend("/platform/campus/buildings", { name, description: desc || null }); setName(""); setDesc(""); })}><Plus /> Add building</Button>
              </CardContent>
            </Card>
            {!data.buildings.length && <Empty title="No buildings yet">Add a building above, then assign photos and reconstructions to it.</Empty>}
            {data.buildings.map((b) => (
              <Card key={b.id} className="shadow-none">
                <CardHeader className="flex flex-row items-start justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2 text-base"><Building2 className="size-4 text-primary" /> {b.name}</CardTitle>
                    {b.description && <div className="text-sm text-muted-foreground">{b.description}</div>}
                  </div>
                  <Button size="sm" variant="ghost" onClick={() => confirm(`Delete building "${b.name}"? Its floors and assignments are removed; the photos and reconstructions themselves are kept.`) && act(() => apiSend(`/platform/campus/buildings/${b.id}`, undefined, "DELETE"))}><Trash2 /></Button>
                </CardHeader>
                <CardContent className="space-y-3">
                  {[{ id: null as string | null, name: "Unassigned floor", level: NaN }, ...b.floors].map((f) => {
                    const items = b.assets.filter((a) => a.floor_id === f.id);
                    if (f.id === null && !items.length) return null;
                    return (
                      <div key={f.id ?? "none"} className="rounded-md border p-2">
                        <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground"><Layers className="size-3" /> {f.name}{Number.isFinite(f.level) ? ` · level ${f.level}` : ""}</div>
                        <div className="flex flex-wrap gap-2">
                          {!items.length && <span className="text-xs text-muted-foreground">No assets on this floor.</span>}
                          {items.map((a) => {
                            const x = byKey.get(`${a.kind}:${a.id}`);
                            return (
                              <div key={a.id} className="group relative flex items-center gap-2 rounded-md border bg-muted/40 py-1 pl-1 pr-6 text-xs">
                                {x?.thumb ? (
                                  // eslint-disable-next-line @next/next/no-img-element
                                  <img src={apiUrl(x.thumb)} alt="" className="size-8 rounded object-cover" />
                                ) : <Box className="size-4 text-primary" />}
                                <Link href={a.kind === "image" ? `/studio?id=${a.id}` : `/explorer?id=${a.id}`} className="hover:underline">{x?.name || a.id}</Link>
                                <button title="Unassign" onClick={() => act(() => apiSend(`/platform/campus/links/${a.kind}/${a.id}`, undefined, "DELETE"))} className="absolute right-1 text-muted-foreground hover:text-destructive"><X className="size-3" /></button>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                  <div className="flex gap-2">
                    <Input className="h-8 max-w-56 text-xs" placeholder="New floor, e.g. Ground floor" value={floorName[b.id] || ""} onChange={(e) => setFloorName({ ...floorName, [b.id]: e.target.value })} />
                    <Button size="sm" variant="outline" disabled={!floorName[b.id]?.trim()} onClick={() => act(async () => { await apiSend(`/platform/campus/buildings/${b.id}/floors`, { name: floorName[b.id], level: b.floors.length }); setFloorName({ ...floorName, [b.id]: "" }); })}><Plus /> Floor</Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          <Card className="h-fit shadow-none">
            <CardHeader><CardTitle className="text-base">Unassigned assets ({data.unassigned.length})</CardTitle></CardHeader>
            <CardContent className="max-h-[70vh] space-y-2 overflow-auto">
              {data.unassigned.map((a) => {
                const key = `${a.kind}:${a.id}`;
                return (
                  <div key={key} className="rounded-md border p-2 text-xs">
                    <div className="flex items-center gap-2">
                      {a.kind === "image" ? <Camera className="size-3.5 text-primary" /> : <Box className="size-3.5 text-primary" />}
                      <span className="flex-1 truncate font-medium">{a.name}</span>
                      {a.provenance && <ProvenanceBadge p={a.provenance} />}
                    </div>
                    {data.buildings.length > 0 && (
                      <div className="mt-2 flex gap-2">
                        <select value={assign[key] || ""} onChange={(e) => setAssign({ ...assign, [key]: e.target.value })} className="h-7 flex-1 rounded border bg-card px-1">
                          <option value="">Choose building / floor…</option>
                          {data.buildings.map((b) => [
                            <option key={b.id} value={`${b.id}|`}>{b.name}</option>,
                            ...b.floors.map((f) => <option key={f.id} value={`${b.id}|${f.id}`}>{b.name} › {f.name}</option>),
                          ])}
                        </select>
                        <Button size="sm" className="h-7" disabled={!assign[key]} onClick={() => {
                          const [bid, fid] = assign[key].split("|");
                          act(() => apiSend("/platform/campus/links", { asset_kind: a.kind, asset_id: a.id, building_id: bid, floor_id: fid || null }));
                        }}>Assign</Button>
                      </div>
                    )}
                  </div>
                );
              })}
              {!data.buildings.length && <div className="text-xs text-muted-foreground">Create a building to assign assets.</div>}
            </CardContent>
          </Card>
        </div>
      )}
    </>
  );
}
