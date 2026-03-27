"use client";

import { Button } from "@/components/ui/button";
import type { StrategyLeg } from "@/lib/types";
import { Trash2Icon, PlusIcon } from "lucide-react";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createDefaultLeg(): StrategyLeg {
  return { type: "call", strike: 100, qty: 1, side: "long" };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface LegBuilderProps {
  legs: StrategyLeg[];
  onChange: (legs: StrategyLeg[]) => void;
}

export function LegBuilder({ legs, onChange }: LegBuilderProps) {
  function handleAdd() {
    onChange([...legs, createDefaultLeg()]);
  }

  function handleRemove(index: number) {
    onChange(legs.filter((_, i) => i !== index));
  }

  function handleUpdate(index: number, patch: Partial<StrategyLeg>) {
    onChange(
      legs.map((leg, i) => (i === index ? { ...leg, ...patch } : leg)),
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">
          Option Legs ({legs.length})
        </span>
        <Button variant="outline" size="sm" onClick={handleAdd}>
          <PlusIcon className="size-3.5" data-icon="inline-start" />
          Add Leg
        </Button>
      </div>

      {legs.length === 0 && (
        <p className="py-4 text-center text-sm text-muted-foreground">
          Pick a template above or add legs manually
        </p>
      )}

      {legs.length > 0 && (
        <div className="space-y-2">
          {/* Column headers */}
          <div className="hidden grid-cols-[1fr_1fr_80px_80px_32px] items-center gap-2 text-xs text-muted-foreground sm:grid">
            <span>Side</span>
            <span>Type</span>
            <span>Strike</span>
            <span>Qty</span>
            <span />
          </div>

          {legs.map((leg, idx) => (
            <div
              key={idx}
              className="grid grid-cols-2 items-center gap-2 rounded-md border border-border bg-background p-2 sm:grid-cols-[1fr_1fr_80px_80px_32px] sm:p-0 sm:border-0 sm:bg-transparent"
            >
              {/* Side */}
              <select
                value={leg.side}
                onChange={(e) =>
                  handleUpdate(idx, {
                    side: e.target.value as "long" | "short",
                  })
                }
                className="h-8 rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
              >
                <option value="long">Long</option>
                <option value="short">Short</option>
              </select>

              {/* Type */}
              <select
                value={leg.type}
                onChange={(e) =>
                  handleUpdate(idx, {
                    type: e.target.value as "call" | "put",
                  })
                }
                className="h-8 rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
              >
                <option value="call">Call</option>
                <option value="put">Put</option>
              </select>

              {/* Strike */}
              <input
                type="number"
                value={leg.strike}
                step={1}
                min={0.01}
                onChange={(e) => {
                  const parsed = parseFloat(e.target.value);
                  if (!Number.isNaN(parsed)) {
                    handleUpdate(idx, { strike: parsed });
                  }
                }}
                className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
                placeholder="Strike"
              />

              {/* Qty */}
              <input
                type="number"
                value={leg.qty ?? 1}
                step={1}
                min={1}
                max={100}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value, 10);
                  if (!Number.isNaN(parsed) && parsed >= 1) {
                    handleUpdate(idx, { qty: parsed });
                  }
                }}
                className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
                placeholder="Qty"
              />

              {/* Remove */}
              <button
                type="button"
                onClick={() => handleRemove(idx)}
                title="Remove leg"
                className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              >
                <Trash2Icon className="size-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
