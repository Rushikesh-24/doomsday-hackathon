"use client";
import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import type { SceneData, SceneFacility } from "@/components/three/scene-viewer";

/* eslint-disable @typescript-eslint/no-explicit-any -- reconstruction metadata/alignment are free-form JSON
   written by the Python pipeline (see docs/SPATIAL.md); pages read individual fields defensively. */
export type Recon = {
  id: string; job: { status: string; video_id: string; options: Record<string, unknown> };
  metadata: Record<string, any> | null; alignment: Record<string, any> | null;
};
export type Asset = { path: string; bytes: number; description: string; url: string };
export type ReconFacility = SceneFacility & Record<string, any>;
/* eslint-enable @typescript-eslint/no-explicit-any */

type Loaded = { rid: string; recon?: Recon; assets?: Record<string, Asset>; scene?: SceneData; facilities?: ReconFacility[]; error?: string };

export function useScene(rid: string | null) {
  const [state, setState] = useState<Loaded | null>(null);
  useEffect(() => {
    if (!rid) return;
    let alive = true;
    (async () => {
      try {
        const recon = await apiGet<Recon>(`/spatial/reconstructions/${rid}`);
        const assets = (await apiGet<{ assets: Record<string, Asset> }>(`/spatial/reconstructions/${rid}/assets`)).assets;
        const traj = assets.trajectory_json ? await apiGet<{ cameras: SceneData["trajectory"] }>(assets.trajectory_json.url) : { cameras: [] };
        const fac = assets.facilities_json ? await apiGet<{ facilities: ReconFacility[] }>(`/spatial/reconstructions/${rid}/facilities`) : { facilities: [] };
        if (!recon.alignment) throw new Error("reconstruction has no alignment yet (still processing?)");
        const scene: SceneData = {
          T: recon.alignment.T_local_from_sfm, splatUrl: assets.splat_ply?.url, pointsUrl: assets.sparse_points_ply?.url,
          trajectory: traj.cameras, facilities: fac.facilities.filter((f) => f.position_sfm),
          metric: recon.alignment.scale_status !== "relative",
        };
        if (alive) setState({ rid, recon, assets, scene, facilities: fac.facilities });
      } catch (e) {
        if (alive) setState({ rid, error: (e as Error).message });
      }
    })();
    return () => { alive = false; };
  }, [rid]);
  const cur = state && state.rid === rid ? state : null;
  return { ...(cur || {}), loading: !!rid && !cur };
}
