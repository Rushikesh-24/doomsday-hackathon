"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Box, Building2, ClipboardCheck, Layers, LayoutDashboard, Navigation, ScanSearch, Sparkles, Wand2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import type { ReactNode } from "react";

const NAV = [
  { group: "Workspace", items: [
    { href: "/", label: "Overview", icon: LayoutDashboard },
    { href: "/studio", label: "AI Studio", icon: Wand2 },
  ] },
  { group: "Spaces", items: [
    { href: "/explorer", label: "3D Explorer", icon: Box },
    { href: "/navigation", label: "Wayfinding", icon: Navigation },
    { href: "/buildings", label: "Buildings", icon: Layers },
    { href: "/campus", label: "Campus", icon: Building2 },
  ] },
  { group: "Insights", items: [
    { href: "/facilities", label: "Facilities", icon: ScanSearch },
    { href: "/compliance", label: "Compliance", icon: ClipboardCheck },
    { href: "/ai-lab", label: "AI Lab", icon: Sparkles },
  ] },
];
const FLAT = NAV.flatMap((g) => g.items);

type Health = { status: string; device: string; default_model: string; loaded_models: string[] };

function HealthPill() {
  const { data, error } = useApi<Health>("/vision/health");
  const ok = data && !error;
  return (
    <div className="glass flex items-center gap-2 rounded-full px-3 py-1.5 text-xs">
      <span className="relative flex size-2">
        {ok && <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-60" />}
        <span className={cn("relative inline-flex size-2 rounded-full", ok ? "bg-emerald-500" : error ? "bg-rose-500" : "bg-slate-300")} />
      </span>
      {ok ? (
        <span className="text-muted-foreground">AI engine online · <span className="font-mono font-medium text-foreground">{data.device}</span></span>
      ) : error ? <span className="text-rose-600">AI engine offline</span> : <span className="text-muted-foreground">Connecting…</span>}
    </div>
  );
}

function Logo() {
  return (
    <div className="flex items-center gap-3 px-5 pb-6 pt-6">
      <div className="relative grid size-10 place-items-center rounded-2xl bg-gradient-to-br from-violet-500 via-indigo-500 to-teal-400 shadow-lg shadow-violet-900/40">
        <Navigation className="size-5 text-white" />
      </div>
      <div>
        <div className="font-display text-base font-bold leading-tight text-white">AccessMap</div>
        <div className="text-[11px] text-white/50">Campus accessibility AI</div>
      </div>
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const isActive = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  const current = FLAT.find((n) => isActive(n.href));
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col bg-sidebar md:flex" style={{ backgroundImage: "radial-gradient(400px 300px at 0% 0%, oklch(0.4 0.15 285 / 0.45), transparent 70%)" }}>
        <Logo />
        <nav className="flex flex-col gap-5 px-3">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-white/35">{g.group}</div>
              <div className="flex flex-col gap-0.5">
                {g.items.map((n) => {
                  const active = isActive(n.href);
                  return (
                    <Link key={n.href} href={n.href}
                      className={cn("group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all",
                        active ? "bg-white/10 font-medium text-white shadow-inner" : "text-white/60 hover:bg-white/5 hover:text-white")}>
                      {active && <span className="absolute inset-y-2 left-0 w-1 rounded-full bg-gradient-to-b from-violet-400 to-teal-300" />}
                      <n.icon className={cn("size-4", active ? "text-violet-300" : "")} /> {n.label}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>
        <div className="m-3 mt-auto rounded-2xl bg-white/5 p-4 text-[11px] leading-relaxed text-white/55 ring-1 ring-white/10">
          <div className="mb-1 font-semibold text-white/80">Screening tool</div>
          Results are image-based estimates, never a compliance certification.
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-4 px-6 py-3 backdrop-blur-md">
          <nav className="flex gap-1 overflow-x-auto md:hidden">
            {FLAT.map((n) => (
              <Link key={n.href} href={n.href} className={cn("whitespace-nowrap rounded-full px-3 py-1 text-xs", isActive(n.href) ? "bg-primary text-primary-foreground" : "text-muted-foreground")}>{n.label}</Link>
            ))}
          </nav>
          <div className="hidden items-center gap-2 text-sm md:flex">
            <span className="text-muted-foreground">AccessMap</span><span className="text-muted-foreground/50">/</span>
            <span className="font-medium">{current?.label ?? ""}</span>
          </div>
          <HealthPill />
        </header>
        <main className="mx-auto w-full max-w-[1500px] flex-1 px-6 pb-10 pt-2">{children}</main>
      </div>
    </div>
  );
}
