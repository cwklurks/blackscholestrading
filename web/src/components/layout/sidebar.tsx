"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Calculator,
  Activity,
  Layers,
  BarChart3,
  LineChart,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/pricing", label: "Pricing", icon: Calculator },
  { href: "/volatility", label: "Volatility", icon: Activity },
  { href: "/strategies", label: "Strategies", icon: Layers },
  { href: "/backtest", label: "Backtest", icon: BarChart3 },
  { href: "/market", label: "Market", icon: LineChart },
] as const;

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-30 flex w-14 flex-col border-r border-border bg-surface lg:w-60">
      <div className="flex h-14 items-center border-b border-border px-3 lg:px-5">
        <span className="hidden text-sm font-semibold tracking-tight text-foreground lg:block">
          Black-Scholes
        </span>
        <span className="block text-sm font-bold text-foreground lg:hidden">
          BS
        </span>
      </div>

      <nav className="flex flex-1 flex-col gap-1 p-2 lg:p-3">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const isActive =
            href === "/" ? pathname === "/" : pathname.startsWith(href);

          return (
            <Link
              key={href}
              href={href}
              className={[
                "group flex items-center gap-3 rounded-md px-2.5 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
              ].join(" ")}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="hidden lg:inline">{label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
