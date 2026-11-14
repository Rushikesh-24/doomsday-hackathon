"use client";
import { AlertTriangle, Inbox, Loader2, ShieldCheck, FlaskConical, User } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Provenance } from "@/lib/types";
import type { ReactNode } from "react";

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="font-display text-3xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1.5 max-w-3xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Stat({ label, value, hint, icon }: { label: string; value: ReactNode; hint?: string; icon?: ReactNode }) {
  return (
    <div className="surface p-5">
      <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
        {label}
        {icon && <span className="grid size-9 place-items-center rounded-xl bg-accent text-accent-foreground">{icon}</span>}
      </div>
      <div className="font-display mt-2 text-3xl font-bold tabular-nums">{value}</div>
      {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
      <div className="flex-1">
        <div className="font-medium text-destructive">Something went wrong</div>
        <div className="mt-0.5 text-muted-foreground">{message}</div>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="text-xs font-medium text-primary hover:underline">
          Retry
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed p-10 text-center">
      <Inbox className="mb-3 size-6 text-muted-foreground" />
      <div className="font-medium">{title}</div>
      {children && <div className="mt-1 max-w-md text-sm text-muted-foreground">{children}</div>}
    </div>
  );
}

const STATUS: Record<string, string> = {
  ESTIMATED: "bg-emerald-50 text-emerald-700 border-emerald-200",
  VERIFIED: "bg-sky-50 text-sky-700 border-sky-200",
  REQUIRES_CALIBRATION: "bg-amber-50 text-amber-800 border-amber-200",
  REQUIRES_MANUAL_MEASUREMENT: "bg-slate-50 text-slate-600 border-slate-200",
  INSUFFICIENT_DATA: "bg-rose-50 text-rose-700 border-rose-200",
  POTENTIAL_ISSUE: "bg-rose-50 text-rose-700 border-rose-200",
  POSSIBLE_ISSUE: "bg-amber-50 text-amber-800 border-amber-200",
  NO_ISSUE_INDICATED: "bg-emerald-50 text-emerald-700 border-emerald-200",
  MEETS_THRESHOLD_VERIFIED: "bg-sky-50 text-sky-700 border-sky-200",
  CANNOT_ASSESS: "bg-slate-50 text-slate-600 border-slate-200",
  NOT_CONFIGURED: "bg-slate-50 text-slate-500 border-slate-200",
  completed: "bg-emerald-50 text-emerald-700 border-emerald-200",
  done: "bg-emerald-50 text-emerald-700 border-emerald-200",
  running: "bg-sky-50 text-sky-700 border-sky-200",
  queued: "bg-slate-50 text-slate-600 border-slate-200",
  failed: "bg-rose-50 text-rose-700 border-rose-200",
  blocked: "bg-rose-50 text-rose-700 border-rose-200",
  completed_with_failures: "bg-amber-50 text-amber-800 border-amber-200",
  interrupted: "bg-amber-50 text-amber-800 border-amber-200",
  verified: "bg-sky-50 text-sky-700 border-sky-200",
  rejected: "bg-rose-50 text-rose-700 border-rose-200",
  needs_review: "bg-amber-50 text-amber-800 border-amber-200",
  fixed: "bg-emerald-50 text-emerald-700 border-emerald-200",
  scale_calibrated: "bg-emerald-50 text-emerald-700 border-emerald-200",
  approximate_metric: "bg-amber-50 text-amber-800 border-amber-200",
  relative: "bg-slate-50 text-slate-600 border-slate-200",
};

export function StatusBadge({ status, className }: { status?: string | null; className?: string }) {
  const s = status || "unknown";
  return (
    <Badge variant="outline" className={cn("font-mono text-[10px] uppercase tracking-wide", STATUS[s] || "", className)}>
      {s.replaceAll("_", " ")}
    </Badge>
  );
}

export function ProvenanceBadge({ p }: { p?: Provenance }) {
  if (!p) return null;
  if (p.kind === "real_sample")
    return (
      <a href={p.source} target="_blank" rel="noreferrer" title={`${p.title} — ${p.author}`}>
        <Badge variant="outline" className="gap-1 border-emerald-200 bg-emerald-50 text-emerald-700">
          <ShieldCheck className="size-3" /> Real photo · {p.license}
        </Badge>
      </a>
    );
  if (p.kind === "synthetic")
    return (
      <Badge variant="outline" className="gap-1 border-violet-200 bg-violet-50 text-violet-700">
        <FlaskConical className="size-3" /> Synthetic
      </Badge>
    );
  return (
    <Badge variant="outline" className="gap-1">
      <User className="size-3" /> Your upload
    </Badge>
  );
}

export function fmt(v: unknown, unit?: string | null, digits = 2): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return `${Number.isInteger(v) ? v : v.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
  return `${String(v)}${unit ? ` ${unit}` : ""}`;
}

export function timeAgo(iso?: string | null): string {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

export const CLASS_COLORS: Record<string, string> = {
  staircase: "#e63946", ramp: "#2a9d8f", elevator: "#457b9d", door: "#f4a261", entrance_exit: "#e9c46a",
  handrail: "#8338ec", tactile_paving: "#ffd60a", corridor: "#6495ed", walkway: "#60d394",
  accessibility_symbol: "#0077b6", signboard: "#ff006e", obstacle: "#ff5722", person: "#9ca3af", wheelchair_user: "#00b4d8",
};
