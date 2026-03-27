"use client";

import type { PricingResponse } from "@/lib/types";
import { formatGreek } from "@/lib/format";

interface GreekDef {
  label: string;
  key: keyof Pick<PricingResponse, "delta" | "gamma" | "vega" | "theta" | "rho">;
  colorClass: string;
}

const GREEKS: GreekDef[] = [
  { label: "DELTA", key: "delta", colorClass: "text-delta" },
  { label: "GAMMA", key: "gamma", colorClass: "text-gamma" },
  { label: "THETA", key: "theta", colorClass: "text-theta" },
  { label: "VEGA", key: "vega", colorClass: "text-vega" },
  { label: "RHO", key: "rho", colorClass: "text-rho" },
];

interface GreeksRowProps {
  data: PricingResponse | null;
  className?: string;
}

export function GreeksRow({ data, className }: GreeksRowProps) {
  return (
    <div
      className={[
        "grid grid-cols-5 gap-4",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {GREEKS.map(({ label, key, colorClass }) => {
        const value = data ? data[key] : null;

        return (
          <div key={key}>
            <div className={`text-xs font-medium uppercase tracking-wider ${colorClass}`}>
              {label}
            </div>
            {value !== null && value !== undefined ? (
              <div className={`font-mono text-base font-medium tabular-nums ${colorClass}`}>
                {formatGreek(key, value)}
              </div>
            ) : (
              <div className="font-mono text-base text-muted-foreground tabular-nums">
                --
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
