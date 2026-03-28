"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { TemplatePicker } from "@/components/strategies/template-picker";
import { LegBuilder } from "@/components/strategies/leg-builder";
import { PayoffChart } from "@/components/charts/payoff-chart";
import { api } from "@/lib/api";
import { ParamInput } from "@/components/ui/param-input";
import type { StrategyLeg, PayoffResponse } from "@/lib/types";

// ---------------------------------------------------------------------------
// Template name detection (for active highlight in TemplatePicker)
// ---------------------------------------------------------------------------

const TEMPLATE_SIGNATURES: Record<string, string> = {
  "long-call-100|long-put-100": "Straddle",
  "long-call-110|long-put-90": "Strangle",
  "short-call-110|long-call-120|short-put-90|long-put-80": "Iron Condor",
  "long-call-90|short-call-100-2|long-call-110": "Butterfly",
  "long-put-95|short-call-105": "Collar",
};

function legsSignature(legs: StrategyLeg[]): string {
  return legs
    .map(
      (l) =>
        `${l.side}-${l.type}-${l.strike}${(l.qty ?? 1) > 1 ? `-${l.qty}` : ""}`,
    )
    .join("|");
}

function detectTemplateName(legs: StrategyLeg[]): string | undefined {
  const sig = legsSignature(legs);
  return TEMPLATE_SIGNATURES[sig];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function StrategiesContent() {
  // Legs state
  const [legs, setLegs] = useState<StrategyLeg[]>([]);

  // Market params
  const [spot, setSpot] = useState(100);
  const [tte, setTte] = useState(0.25);
  const [rate, setRate] = useState(0.05);
  const [sigma, setSigma] = useState(0.2);

  // Spot range for chart
  const [rangeMin, setRangeMin] = useState(50);
  const [rangeMax, setRangeMax] = useState(150);

  // Results
  const [payoff, setPayoff] = useState<PayoffResponse | null>(null);

  // Loading / error
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debounce ref for auto-calculate
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Template name detection
  const activeName = detectTemplateName(legs);

  // Template selection handler
  const handleTemplateSelect = useCallback(
    (templateLegs: StrategyLeg[]) => {
      setLegs(templateLegs);
      setPayoff(null);
      setError(null);
    },
    [],
  );

  // Legs change handler
  const handleLegsChange = useCallback((newLegs: StrategyLeg[]) => {
    setLegs(newLegs);
  }, []);

  // Calculate payoff
  const handleCalculate = useCallback(async () => {
    if (legs.length === 0) {
      setError("Add at least one leg to calculate payoff");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await api.payoff({
        legs,
        spot_range: { min: rangeMin, max: rangeMax },
        S: spot,
        T: tte,
        r: rate,
        sigma,
      });
      setPayoff(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : "An error occurred";
      setError(message);
      setPayoff(null);
    } finally {
      setLoading(false);
    }
  }, [legs, rangeMin, rangeMax, spot, tte, rate, sigma]);

  // Auto-recompute payoff on parameter changes (debounced 300ms)
  useEffect(() => {
    if (legs.length === 0) return;

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      handleCalculate();
    }, 300);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [handleCalculate]);

  return (
    <div className="space-y-6">
      {/* Template Picker */}
      <TemplatePicker onSelect={handleTemplateSelect} activeName={activeName} />

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* ---------------------------------------------------------------- */}
        {/* Left rail - Leg Builder + Parameters                             */}
        {/* ---------------------------------------------------------------- */}
        <aside className="w-full shrink-0 space-y-4 border-r border-border bg-surface p-4 lg:w-80 lg:pr-6">
          {/* Leg Builder */}
          <LegBuilder legs={legs} onChange={handleLegsChange} />

          {/* Market Parameters */}
          <div className="space-y-3">
            <span className="mb-3 block text-sm font-medium text-foreground">
              Market Parameters
            </span>
            <div className="space-y-3">
              <ParamInput
                label="Spot Price (S)"
                value={spot}
                step={1}
                min={0.01}
                onChange={setSpot}
              />
              <ParamInput
                label="Time to Expiry (T)"
                value={tte}
                step={0.01}
                min={0.001}
                max={10}
                onChange={setTte}
              />
              <ParamInput
                label="Risk-free Rate (r)"
                value={rate}
                step={0.005}
                min={0}
                max={1}
                onChange={setRate}
              />
              <ParamInput
                label="Volatility (sigma)"
                value={sigma}
                step={0.01}
                min={0.001}
                max={5}
                onChange={setSigma}
              />
            </div>
          </div>

          {/* Spot Range */}
          <div className="space-y-3">
            <span className="mb-3 block text-sm font-medium text-foreground">
              Chart Range
            </span>
            <div className="grid grid-cols-2 gap-3">
              <ParamInput
                label="Min"
                value={rangeMin}
                step={5}
                min={0.01}
                onChange={setRangeMin}
              />
              <ParamInput
                label="Max"
                value={rangeMax}
                step={5}
                min={0.01}
                onChange={setRangeMax}
              />
            </div>
          </div>

          {/* Calculate button */}
          <Button
            className="w-full"
            size="lg"
            onClick={handleCalculate}
            disabled={loading || legs.length === 0}
          >
            {loading ? "Calculating..." : "Calculate Payoff"}
          </Button>

          {error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}
        </aside>

        {/* ---------------------------------------------------------------- */}
        {/* Right content - Payoff Chart + Stats                             */}
        {/* ---------------------------------------------------------------- */}
        <div className="min-w-0 flex-1 space-y-6">
          {/* Payoff chart */}
          {payoff && (
            <>
              {/* Max profit / loss labels */}
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <StatCard
                  label="Max Profit"
                  value={payoff.max_profit}
                  positive
                />
                <StatCard
                  label="Max Loss"
                  value={payoff.max_loss}
                  positive={false}
                />
                <div className="col-span-2 sm:col-span-1">
                  <div className="space-y-1">
                    <div className="text-xs text-muted-foreground">
                      Breakevens
                    </div>
                    <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-foreground">
                      {payoff.breakevens.length > 0
                        ? payoff.breakevens
                            .map((b) => `$${b.toFixed(2)}`)
                            .join(", ")
                        : "None"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Chart */}
              <div className="border-b border-border pb-6">
                <PayoffChart
                  prices={payoff.prices}
                  pnl={payoff.pnl}
                  breakevens={payoff.breakevens}
                  title="P&L at Expiration"
                  height={440}
                />
              </div>

              {/* Legs summary */}
              <div className="border-b border-border pb-4">
                <span className="mb-2 block text-sm font-medium text-foreground">
                  Strategy Summary
                </span>
                <div className="space-y-1">
                  {legs.map((leg, idx) => (
                    <div
                      key={idx}
                      className="flex items-center gap-2 text-sm text-muted-foreground"
                    >
                      <span
                        className={`inline-flex h-5 items-center rounded px-1.5 text-xs font-medium ${
                          leg.side === "long"
                            ? "bg-positive/10 text-positive"
                            : "bg-negative/10 text-negative"
                        }`}
                      >
                        {leg.side.toUpperCase()}
                      </span>
                      <span>
                        {leg.qty ?? 1}x {leg.type.toUpperCase()} @ $
                        {leg.strike}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Empty state */}
          {!payoff && !loading && (
            <div className="flex min-h-[400px] items-center justify-center">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">
                  Build a strategy — start with a call spread
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page (thin wrapper with header for standalone route)
// ---------------------------------------------------------------------------

export default function StrategiesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Strategies</h1>
      </div>
      <StrategiesContent />
    </div>
  );
}

// ---------------------------------------------------------------------------
// StatCard - max profit / loss display
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  positive,
}: {
  label: string;
  value: number | null;
  positive: boolean;
}) {
  const displayValue =
    value === null ? "Unlimited" : `$${Math.abs(value).toFixed(2)}`;
  const colorClass = positive
    ? "text-positive"
    : "text-negative";

  return (
    <div className="space-y-1">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`mt-1 font-mono text-lg font-semibold tabular-nums ${colorClass}`}>
        {displayValue}
      </div>
    </div>
  );
}
