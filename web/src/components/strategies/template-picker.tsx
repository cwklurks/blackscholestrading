"use client";

import type { StrategyLeg } from "@/lib/types";

// ---------------------------------------------------------------------------
// Template definitions
// ---------------------------------------------------------------------------

interface StrategyTemplate {
  readonly name: string;
  readonly description: string;
  readonly legs: readonly StrategyLeg[];
}

const TEMPLATES: readonly StrategyTemplate[] = [
  {
    name: "Straddle",
    description: "Long call + long put at ATM",
    legs: [
      { type: "call", strike: 100, qty: 1, side: "long" },
      { type: "put", strike: 100, qty: 1, side: "long" },
    ],
  },
  {
    name: "Strangle",
    description: "Long OTM call + long OTM put",
    legs: [
      { type: "call", strike: 110, qty: 1, side: "long" },
      { type: "put", strike: 90, qty: 1, side: "long" },
    ],
  },
  {
    name: "Iron Condor",
    description: "Short strangle + long wings",
    legs: [
      { type: "call", strike: 110, qty: 1, side: "short" },
      { type: "call", strike: 120, qty: 1, side: "long" },
      { type: "put", strike: 90, qty: 1, side: "short" },
      { type: "put", strike: 80, qty: 1, side: "long" },
    ],
  },
  {
    name: "Butterfly",
    description: "Long K1 + 2x short ATM + long K3",
    legs: [
      { type: "call", strike: 90, qty: 1, side: "long" },
      { type: "call", strike: 100, qty: 2, side: "short" },
      { type: "call", strike: 110, qty: 1, side: "long" },
    ],
  },
  {
    name: "Collar",
    description: "Long put + short call (with stock)",
    legs: [
      { type: "put", strike: 95, qty: 1, side: "long" },
      { type: "call", strike: 105, qty: 1, side: "short" },
    ],
  },
] as const;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface TemplatePickerProps {
  onSelect: (legs: StrategyLeg[]) => void;
  activeName?: string;
}

export function TemplatePicker({ onSelect, activeName }: TemplatePickerProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <span className="mb-3 block text-sm font-medium text-foreground">
        Strategy Templates
      </span>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {TEMPLATES.map((tpl) => {
          const isActive = activeName === tpl.name;
          return (
            <button
              key={tpl.name}
              type="button"
              onClick={() => onSelect(tpl.legs.map((l) => ({ ...l })))}
              title={tpl.description}
              className={`rounded-md border px-3 py-2 text-left text-sm font-medium transition-colors ${
                isActive
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-input bg-background text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              }`}
            >
              <span className="block">{tpl.name}</span>
              <span
                className={`mt-0.5 block text-xs ${
                  isActive
                    ? "text-primary-foreground/70"
                    : "text-muted-foreground/70"
                }`}
              >
                {tpl.description}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
