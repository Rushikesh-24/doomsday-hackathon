"use client";
import { useMemo, useState } from "react";
import { Loader2, Ruler, ScanText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { apiSend, apiUrl } from "@/lib/api";
import type { ImageResult, MeasurementReport } from "@/lib/types";
import { ErrorState, StatusBadge, fmt } from "@/components/app/common";

type Mode = "priors" | "uncalibrated" | "reference";

export function MeasurePanel({ result, initial }: { result: ImageResult; initial?: MeasurementReport | null }) {
  const [mode, setMode] = useState<Mode>("priors");
  const [profile, setProfile] = useState("ada_2010");
  const doors = useMemo(() => result.detections.filter((d) => ["door", "elevator", "person"].includes(d.class_name)), [result]);
  const [refId, setRefId] = useState<string>(doors[0]?.id || "");
  const [refLen, setRefLen] = useState("2.10");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [rep, setRep] = useState<MeasurementReport | null>(initial || null);

  async function run() {
    setBusy(true);
    setErr(null);
    const calibration =
      mode === "priors" ? { use_priors: true } :
      mode === "uncalibrated" ? { use_priors: false } :
      { use_priors: false, references: [{ type: "object", detection_id: refId, dimension: "height", length_m: parseFloat(refLen) }] };
    try {
      setRep(await apiSend<MeasurementReport>("/measurements/analyze-stored", { vision_request_id: result.request_id, calibration, rules_profile: profile }));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base"><Ruler className="size-4" /> Accessibility measurements</CardTitle>
        <CardDescription>Metric depth + segmentation + geometry on the detected facilities. Absolute sizes need a scale reference; without one they are withheld.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-3">
          {([
            ["priors", "Typical object sizes", "door 2.06 m / person 1.68 m priors"],
            ["uncalibrated", "No calibration", "only scale-free values (slopes, counts, ratios)"],
            ["reference", "Known measurement", "you supply a measured height"],
          ] as [Mode, string, string][]).map(([m, t, d]) => (
            <button key={m} onClick={() => setMode(m)}
              className={`rounded-lg border p-3 text-left text-sm transition ${mode === m ? "border-primary bg-accent" : "hover:bg-muted"}`}>
              <div className="font-medium">{t}</div>
              <div className="mt-0.5 text-xs text-muted-foreground">{d}</div>
            </button>
          ))}
        </div>
        {mode === "reference" && (
          doors.length ? (
            <div className="flex flex-wrap items-center gap-2 text-sm">
              Height of
              <select value={refId} onChange={(e) => setRefId(e.target.value)} className="h-8 rounded-md border bg-background px-2">
                {doors.map((d) => <option key={d.id} value={d.id}>{d.class_name} {d.id.slice(-3)} ({(d.confidence * 100).toFixed(0)}%)</option>)}
              </select>
              =
              <Input value={refLen} onChange={(e) => setRefLen(e.target.value)} className="h-8 w-20" /> m
            </div>
          ) : <div className="text-sm text-muted-foreground">No door, lift or person detected to use as a reference in this image.</div>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <select value={profile} onChange={(e) => setProfile(e.target.value)} className="h-8 rounded-md border bg-background px-2 text-sm">
            <option value="ada_2010">ADA 2010 (US)</option>
            <option value="india_hg_2021">India Harmonised Guidelines 2021</option>
          </select>
          <Button onClick={run} disabled={busy || (mode === "reference" && !doors.length)}>
            {busy ? <Loader2 className="animate-spin" /> : <Ruler />} {busy ? "Measuring (≈5–10 s)…" : "Run measurements"}
          </Button>
        </div>
        {err && <ErrorState message={err} />}
        {rep && <MeasurementView rep={rep} />}
      </CardContent>
    </Card>
  );
}

export function MeasurementView({ rep }: { rep: MeasurementReport }) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <StatusBadge status={rep.scale.status} />
        <span className="text-muted-foreground">scale ×{rep.scale.scale_factor.toFixed(3)} ± {(rep.scale.rel_sigma * 100).toFixed(0)}% · camera {rep.camera.source} ({rep.camera.hfov_deg}° HFOV) · gravity {String(rep.scene.gravity_source)}</span>
      </div>
      {rep.scale.notes.map((n) => <div key={n} className="text-xs text-amber-700">• {n}</div>)}
      {rep.annotated_image && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={apiUrl(rep.annotated_image)} alt="measured" className="w-full rounded-lg border" />
      )}
      {rep.facilities.map((f) => (
        <div key={f.detection_id} className="rounded-lg border">
          <div className="flex items-center justify-between border-b bg-muted/50 px-3 py-2 text-sm">
            <span className="font-medium capitalize">{f.class_name.replace("_", " ")} <span className="font-mono text-xs text-muted-foreground">{f.detection_id.slice(-3)}</span></span>
            <span className="text-xs text-muted-foreground">mask: {f.mask_source} {f.truncated_sides.length ? `· truncated ${f.truncated_sides.join(",")}` : ""}</span>
          </div>
          <table className="w-full text-sm">
            <tbody>
              {f.measurements.map((m) => (
                <tr key={m.id} className="border-b last:border-0 align-top">
                  <td className="px-3 py-2 font-medium">{m.name.replaceAll("_", " ")}</td>
                  <td className="px-3 py-2 tabular-nums">
                    {m.status === "REQUIRES_CALIBRATION"
                      ? <span className="text-muted-foreground">withheld <span className="text-xs">(provisional {fmt(m.provisional_value, "m")})</span></span>
                      : <>{m.lower_bound ? "≥ " : ""}{fmt(m.value, m.unit)}{m.uncertainty != null && <span className="text-muted-foreground"> ± {fmt(m.uncertainty)}</span>}</>}
                  </td>
                  <td className="px-3 py-2"><StatusBadge status={m.status} /></td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">{m.refusal_reason || m.method}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {!!rep.assessment.length && (
        <div className="rounded-lg border">
          <div className="border-b bg-muted/50 px-3 py-2 text-sm font-medium">Screening — {rep.assessment_summary.profile_title}</div>
          {rep.assessment.map((a, i) => (
            <div key={i} className="flex items-start gap-3 border-b px-3 py-2 text-sm last:border-0">
              <StatusBadge status={a.outcome} className="mt-0.5 shrink-0" />
              <div><span className="font-medium">{a.rule_id.replaceAll("_", " ")}</span>{a.clause && <span className="text-xs text-muted-foreground"> · {a.clause}</span>}<div className="text-xs text-muted-foreground">{a.message}</div></div>
            </div>
          ))}
          <div className="px-3 py-2 text-xs text-muted-foreground">{rep.assessment_summary.statement}</div>
        </div>
      )}
    </div>
  );
}

export function SignReader({ requestId, hasSigns }: { requestId: string; hasSigns: boolean }) {
  const [busy, setBusy] = useState(false);
  const [signs, setSigns] = useState<{ detection_id: string; class_name: string; text: string | null; status: string; reason?: string }[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  if (!hasSigns) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base"><ScanText className="size-4" /> Read signs</CardTitle>
        <CardDescription>Florence-2 (MIT) OCR on each detected signboard / accessibility symbol crop.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <Button variant="outline" disabled={busy} onClick={async () => {
          setBusy(true); setErr(null);
          try { setSigns((await apiSend<{ signs: typeof signs }>(`/ai/read-signs/${requestId}`)).signs); }
          catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
        }}>{busy ? <Loader2 className="animate-spin" /> : <ScanText />} Read sign text</Button>
        {err && <ErrorState message={err} />}
        {signs?.map((s) => (
          <div key={s.detection_id} className="flex items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm">
            <span className="text-xs text-muted-foreground">{s.class_name} {s.detection_id.slice(-3)}</span>
            <span className="flex-1 font-medium">{s.text || <span className="text-muted-foreground">{s.reason || "no text found"}</span>}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
