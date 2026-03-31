"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { ParamInput } from "@/components/ui/param-input";

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
          <ParamInput
            label="Dividend Yield (q)"
            step={0.005}
            min={0}
            max={1}
            value={values.q}
            onChange={(v) => onChange({ ...values, q: v })}
          />
          <ParamInput
            label="Borrow Cost"
            step={0.005}
            min={0}
            max={1}
            value={values.borrowCost}
            onChange={(v) => onChange({ ...values, borrowCost: v })}
          />
        </div>
      )}
    </div>
  );
}
