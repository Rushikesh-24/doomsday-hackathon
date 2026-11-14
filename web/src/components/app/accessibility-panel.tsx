"use client";
// Overall accessibility score of a scanned map + where it is not accessible (GET /spatial/reconstructions/{id}/accessibility).
import { AlertTriangle, Info, MapPin } from "lucide-react";
import { cn } from "@/lib/utils";

export type Issue = { id: string; kind: string; severity: "high" | "medium" | "low"; title: string; detail: string; position: number[]; place?: string };
export type Accessibility = {
  score: number; grade: string;
  categories: { key: string; label: string; weight: number; score: number | null; points: number | null; reason: string }[];
  issues: Issue[]; counts: { high: number; medium: number; low: number };
  floor: { median_width_m: number | null; min_width_m: number | null; measured_fraction: number };
  note: string;
  corridor: { centre: number[][]; left: number[][]; right: number[][]; width_m: (number | null)[] };
};

export const SEVERITY = {
  high: { label: "High", color: "#e11d48", bg: "bg-rose-50", text: "text-rose-700" },
  medium: { label: "Medium", color: "#f59e0b", bg: "bg-amber-50", text: "text-amber-800" },
  low: { label: "Low", color: "#64748b", bg: "bg-slate-50", text: "text-slate-700" },
} as const;

export function gradeColor(score: number) {
  return score >= 85 ? "#059669" : score >= 70 ? "#0d9488" : score >= 55 ? "#d97706" : score >= 40 ? "#ea580c" : "#e11d48";
}

export function ScoreRing({ score, grade, size = 112 }: { score: number; grade: string; size?: number }) {
  const r = size / 2 - 9, c = 2 * Math.PI * r, col = gradeColor(score);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Accessibility score ${score} of 100, grade ${grade}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeOpacity={0.08} strokeWidth={10} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={col} strokeWidth={10} strokeLinecap="round"
        strokeDasharray={`${(c * score) / 100} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: "stroke-dasharray .8s ease" }} />
      <text x="50%" y="47%" textAnchor="middle" fontSize={size * 0.3} fontWeight={700} fill="currentColor">{score}</text>
      <text x="50%" y="66%" textAnchor="middle" fontSize={size * 0.11} fontWeight={600} fill={col}>GRADE {grade}</text>
    </svg>
  );
}

export function AccessibilityPanel({ data, selected, onIssue }: { data: Accessibility; selected?: string | null; onIssue: (i: Issue) => void }) {
  const verdict = data.score >= 85 ? "Highly accessible" : data.score >= 70 ? "Mostly accessible" : data.score >= 55 ? "Partly accessible" : "Hard to access";
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 rounded-2xl border bg-gradient-to-br from-white to-violet-50/60 p-3">
        <ScoreRing score={data.score} grade={data.grade} />
        <div className="min-w-0">
          <div className="eyebrow">Accessibility score</div>
          <div className="text-lg font-semibold leading-tight">{verdict}</div>
          <div className="mt-1 flex flex-wrap gap-1 text-[11px]">
            {(["high", "medium", "low"] as const).map((s) => data.counts[s] > 0 && (
              <span key={s} className={cn("rounded-full px-2 py-0.5 font-semibold", SEVERITY[s].bg, SEVERITY[s].text)}>{data.counts[s]} {SEVERITY[s].label.toLowerCase()}</span>
            ))}
            {!data.issues.length && <span className="rounded-full bg-emerald-50 px-2 py-0.5 font-semibold text-emerald-700">No problems found</span>}
          </div>
        </div>
      </div>

      <div className="space-y-2.5">
        {data.categories.map((c) => (
          <div key={c.key} title={c.reason}>
            <div className="flex items-baseline justify-between text-xs">
              <span className="font-medium">{c.label}</span>
              <span className="tabular-nums text-muted-foreground">{c.points == null ? "unknown" : `${c.points.toFixed(0)} / ${c.weight}`}</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full transition-all" style={{ width: `${(c.score ?? 0) * 100}%`, background: c.score == null ? "transparent" : gradeColor((c.score ?? 0) * 100) }} />
            </div>
            <div className="mt-1 text-[11px] leading-snug text-muted-foreground">{c.reason}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="eyebrow mb-2">Where it is not accessible</div>
        {!data.issues.length && <div className="rounded-xl bg-emerald-50 p-3 text-xs text-emerald-800">Nothing flagged along the walked path.</div>}
        <div className="space-y-2">
          {data.issues.map((i) => (
            <button key={i.id} onClick={() => onIssue(i)}
              className={cn("flex w-full gap-3 rounded-2xl border p-3 text-left transition hover:shadow-sm", selected === i.id ? "border-rose-300 bg-rose-50/60" : "hover:bg-muted/50")}>
              <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg text-white" style={{ background: SEVERITY[i.severity].color }}>
                <AlertTriangle className="size-3.5" />
              </span>
              <span className="min-w-0">
                <span className="block text-xs font-semibold">{i.title}</span>
                <span className="block text-[11px] leading-snug text-muted-foreground">{i.detail}</span>
                <span className="mt-1 inline-flex items-center gap-1 text-[10px] font-medium text-violet-700"><MapPin className="size-3" /> Show on map</span>
              </span>
            </button>
          ))}
        </div>
      </div>
      <p className="flex gap-1.5 text-[11px] leading-snug text-muted-foreground"><Info className="mt-0.5 size-3 shrink-0" />{data.note}</p>
    </div>
  );
}
