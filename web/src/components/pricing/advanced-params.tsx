"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

export interface AdvancedParamsValues {
  q: number;
  borrowCost: number;
}

interface AdvancedParamsProps {
  values: AdvancedParamsValues;
  onChange: (values: AdvancedParamsValues) => void;
}

export function AdvancedParams({ values, onChange }: AdvancedParamsProps) {
  const [isOpen, setIsOpen] = useState(false);

  function handleChange(field: keyof AdvancedParamsValues, raw: string) {
    const parsed = parseFloat(raw);
    if (Number.isNaN(parsed)) return;
    onChange({ ...values, [field]: parsed });
  }

  return (
    <div className="rounded-lg border border-border bg-card">
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex w-full items-center justify-between px-3 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-accent/50"
      >
        <span>Advanced Parameters</span>
        {isOpen ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {isOpen && (
        <div className="space-y-3 border-t border-border px-3 pb-3 pt-3">
          <div>
            <label className="mb-1 block text-xs text-muted-foreground">
              Dividend Yield (q)
            </label>
            <input
              type="number"
              step="0.005"
              min="0"
              max="1"
              value={values.q}
              onChange={(e) => handleChange("q", e.target.value)}
              className="h-8 w-full rounded-md border border-input bg-background px-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs text-muted-foreground">
              Borrow Cost
            </label>
            <input
              type="number"
              step="0.005"
              min="0"
              max="1"
              value={values.borrowCost}
              onChange={(e) => handleChange("borrowCost", e.target.value)}
              className="h-8 w-full rounded-md border border-input bg-background px-2.5 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
            />
          </div>
        </div>
      )}
    </div>
  );
}
