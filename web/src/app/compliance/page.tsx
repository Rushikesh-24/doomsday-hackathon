"use client";
import Link from "next/link";
import { useState } from "react";
import { AlertTriangle, Check, ShieldAlert, Wrench } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useApi } from "@/hooks/use-api";
import { apiSend, apiUrl } from "@/lib/api";
import { Empty, ErrorState, Loading, PageHeader, Stat, StatusBadge, fmt } from "@/components/app/common";

type Issue = {
  rule_id: string; profile: string; clause?: string | null; detection_id: string; facility_class: string; measurement_name: string;
  outcome: string; threshold?: number | null; measured_value?: number | null; uncertainty?: number | null; message: string;
  image_name: string; request_id: string; annotated_url?: string | null; review: { status: string; note?: string } | null;
};
type Gap = { code: string; message: string; image_name: string; request_id: string; detection_ids: string[] };
type Compliance = {
  issues: Issue[]; gaps: Gap[]; outcome_counts: Record<string, number>; issues_by_rule: Record<string, number>;
  gap_counts: Record<string, number>; progress: { issues: number; reviewed: number; fixed: number }; statement: string;
};

export default function CompliancePage() {
  const { data, error, loading, reload } = useApi<Compliance>("/platform/compliance");
  const [open, setOpen] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function review(i: Issue, status: string) {
    setBusy(true); setErr(null);
    try {
      await apiSend(`/platform/facilities/${i.detection_id}/review`, { status, kind: "image_detection", reviewer: "web user", note: `${i.rule_id}: ${status}` });
      reload();
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }

  const p = data?.progress;
  return (
    <>
      <PageHeader title="Compliance screening" description="Measured facilities checked against the configured rule profile (ADA 2010 by default). This is screening, not certification: POSSIBLE issues are within measurement uncertainty, and nothing here declares a facility compliant." />
      {loading && <Loading />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Possible / potential issues" value={p!.issues} icon={<ShieldAlert className="size-4" />} hint="from measured facilities" />
            <Stat label="Reviewed by a person" value={`${p!.reviewed} / ${p!.issues}`} hint="latest review per facility" icon={<Check className="size-4" />} />
            <Stat label="Marked fixed" value={p!.fixed} hint="improvement tracking" icon={<Wrench className="size-4" />} />
            <Stat label="Cannot assess" value={data.outcome_counts.CANNOT_ASSESS ?? 0} hint="measurement withheld or missing" icon={<AlertTriangle className="size-4" />} />
          </div>
          <Card>
            <CardContent className="grid gap-6 p-5 md:grid-cols-3">
              <div>
                <div className="mb-2 text-sm font-medium">Review progress</div>
                <Progress value={p!.issues ? (p!.reviewed / p!.issues) * 100 : 0} />
                <div className="mt-1 text-xs text-muted-foreground">{p!.issues ? Math.round((p!.reviewed / p!.issues) * 100) : 0}% of issues reviewed</div>
              </div>
              <div>
                <div className="mb-2 text-sm font-medium">Issues by rule</div>
                {Object.entries(data.issues_by_rule).map(([k, v]) => <div key={k} className="flex justify-between text-xs"><span className="text-muted-foreground">{k.replaceAll("_", " ")}</span><span className="tabular-nums">{v}</span></div>)}
              </div>
              <div>
                <div className="mb-2 text-sm font-medium">Outcomes</div>
                {Object.entries(data.outcome_counts).map(([k, v]) => <div key={k} className="flex items-center justify-between text-xs"><StatusBadge status={k} /><span className="tabular-nums">{v}</span></div>)}
              </div>
            </CardContent>
          </Card>
          {err && <ErrorState message={err} />}
          <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
            <Card>
              <CardHeader><CardTitle className="text-base">Issues</CardTitle><CardDescription>{data.statement}</CardDescription></CardHeader>
              <CardContent className="space-y-2">
                {!data.issues.length && <Empty title="No issues">Run measurements in AI Studio to screen facilities.</Empty>}
                {data.issues.map((i, k) => (
                  <div key={k} className="rounded-lg border">
                    <button onClick={() => setOpen(open === k ? null : k)} className="flex w-full flex-wrap items-center gap-3 px-3 py-2 text-left text-sm hover:bg-muted/50">
                      <StatusBadge status={i.outcome} />
                      <span className="font-medium capitalize">{i.facility_class.replaceAll("_", " ")} · {i.measurement_name.replaceAll("_", " ")}</span>
                      <span className="text-xs text-muted-foreground">{i.profile.toUpperCase()} {i.clause}</span>
                      <span className="ml-auto text-xs text-muted-foreground">{i.image_name}</span>
                      {i.review && <StatusBadge status={i.review.status} />}
                    </button>
                    {open === k && (
                      <div className="grid gap-3 border-t p-3 text-sm md:grid-cols-[220px_1fr]">
                        {i.annotated_url && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={apiUrl(i.annotated_url)} alt="" className="w-full rounded border" />
                        )}
                        <div className="space-y-2">
                          <div className="text-xs text-muted-foreground">{i.message}</div>
                          <div className="flex gap-4 text-xs">
                            <span>Measured <b className="tabular-nums">{fmt(i.measured_value)}</b>{i.uncertainty ? ` ± ${fmt(i.uncertainty)}` : ""}</span>
                            <span>Threshold <b className="tabular-nums">{fmt(i.threshold)}</b></span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <Button size="sm" disabled={busy} onClick={() => review(i, "verified")}><Check /> Confirm issue on site</Button>
                            <Button size="sm" variant="outline" disabled={busy} onClick={() => review(i, "rejected")}>Not an issue</Button>
                            <Button size="sm" variant="outline" disabled={busy} onClick={() => review(i, "fixed")}><Wrench /> Fixed</Button>
                            <Link href={`/studio?id=${i.request_id}`}><Button size="sm" variant="ghost">Open in AI Studio</Button></Link>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
            <Card className="h-fit">
              <CardHeader><CardTitle className="text-base">Accessibility gaps</CardTitle><CardDescription>Scene-level findings from detection (e.g. stairs without a detected handrail). &ldquo;Not detected&rdquo; is not &ldquo;absent&rdquo;.</CardDescription></CardHeader>
              <CardContent className="space-y-2">
                {data.gaps.map((g, k) => (
                  <Link key={k} href={`/studio?id=${g.request_id}`} className="block rounded-md border border-amber-200 bg-amber-50/60 p-2 text-xs hover:bg-amber-50">
                    <div className="font-medium text-amber-900">{g.code.replaceAll("_", " ").toLowerCase()}</div>
                    <div className="text-amber-900/80">{g.message}</div>
                    <div className="mt-1 text-muted-foreground">{g.image_name}</div>
                  </Link>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
