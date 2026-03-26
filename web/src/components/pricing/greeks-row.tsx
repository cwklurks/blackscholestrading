"use client";

import type { PricingResponse } from "@/lib/types";

interface GreekDef {
  label: string;
  key: keyof Pick<PricingResponse, "delta" | "gamma" | "vega" | "theta" | "rho">;
  decimals: number;
}

const GREEKS: GreekDef[] = [
  { label: "Delta", key: "delta", decimals: 4 },
  { label: "Gamma", key: "gamma", decimals: 6 },
  { label: "Vega", key: "vega", decimals: 4 },
  { label: "Theta", key: "theta", decimals: 4 },
  { label: "Rho", key: "rho", decimals: 4 },
];

function signColor(value: number): string {
  if (value > 0) return "text-positive";
  if (value < 0) return "text-negative";
  return "text-muted-foreground";
}

interface GreeksRowProps {
  data: PricingResponse | null;
  className?: string;
}

export function GreeksRow({ data, className }: GreeksRowProps) {
  return (
    <div
      className={[
        "grid grid-cols-5 gap-3 rounded-lg border border-border bg-card p-3",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {GREEKS.map(({ label, key, decimals }) => {
        const value = data ? data[key] : null;

        return (
          <div key={key} className="flex flex-col items-center gap-1">
            <span className="text-xs font-medium text-muted-foreground">
              {label}
            </span>
            {value !== null && value !== undefined ? (
              <span className={`font-mono text-sm ${signColor(value)}`}>
                {value.toFixed(decimals)}
              </span>
            ) : (
              <span className="font-mono text-sm text-muted-foreground">
                --
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
