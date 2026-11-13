"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Box, Building2, ClipboardCheck, LayoutDashboard, Navigation, ScanSearch, Sparkles, Wand2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import type { ReactNode } from "react";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/studio", label: "AI Studio", icon: Wand2 },
  { href: "/explorer", label: "3D Explorer", icon: Box },
  { href: "/facilities", label: "Facility Intelligence", icon: ScanSearch },
  { href: "/campus", label: "Campus Explorer", icon: Building2 },
  { href: "/navigation", label: "Navigation", icon: Navigation },
  { href: "/compliance", label: "Compliance", icon: ClipboardCheck },
  { href: "/ai-lab", label: "AI Lab", icon: Sparkles },
];

type Health = { status: string; device: string; default_model: string; loaded_models: string[] };

function HealthPill() {
  const { data, error } = useApi<Health>("/vision/health");
  const ok = data && !error;
  return (
    <div className="flex items-center gap-2 rounded-full border bg-card px-3 py-1 text-xs">
      <span className={cn("size-2 rounded-full", ok ? "bg-emerald-500" : error ? "bg-rose-500" : "bg-slate-300")} />
      {ok ? (
        <span className="text-muted-foreground">
          AI backend online · <span className="font-mono">{data.device}</span>
        </span>
      ) : error ? (
        <span className="text-rose-600">AI backend offline</span>
      ) : (
        <span className="text-muted-foreground">Connecting…</span>
      )}
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r bg-sidebar md:flex">
        <div className="flex items-center gap-2 px-5 py-5">
          <div className="grid size-8 place-items-center rounded-lg bg-primary text-primary-foreground">
            <Navigation className="size-4" />
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight">AccessMap</div>
            <div className="text-[11px] text-muted-foreground">Campus accessibility intelligence</div>
          </div>
        </div>
        <nav className="flex flex-col gap-0.5 px-3">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link key={n.href} href={n.href}
                className={cn("flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                  active ? "bg-accent font-medium text-accent-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground")}>
                <n.icon className="size-4" /> {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto p-4 text-[11px] leading-relaxed text-muted-foreground">
          Screening tool — results are image-based estimates and never a compliance certification.
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-4 border-b bg-background/80 px-6 py-3 backdrop-blur">
          <nav className="flex gap-1 overflow-x-auto md:hidden">
            {NAV.map((n) => (
              <Link key={n.href} href={n.href} className="rounded-md px-2 py-1 text-xs text-muted-foreground">{n.label}</Link>
            ))}
          </nav>
          <div className="hidden text-sm text-muted-foreground md:block">PS32 · Accessible campus navigation & facility intelligence</div>
          <HealthPill />
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-6 py-6">{children}</main>
      </div>
    </div>
  );
}
