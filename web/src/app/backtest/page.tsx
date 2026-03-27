"use client";

import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { BaseChart } from "@/components/charts/base-chart";
import { api } from "@/lib/api";
import { ParamInput } from "@/components/ui/param-input";
import type { BacktestLeg, BacktestResponse } from "@/lib/types";

// ---------------------------------------------------------------------------
// Default leg factory
// ---------------------------------------------------------------------------

function createDefaultLeg(): BacktestLeg {
  const expiry = new Date();
  expiry.setMonth(expiry.getMonth() + 1);
  const isoDate = expiry.toISOString().split("T")[0];

  return {
    type: "call",
    strike: 100,
    expiry: isoDate,
    qty: 1,
    side: "long",
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function BacktestPage() {
  // Config state
  const [ticker, setTicker] = useState("AAPL");
  const [legs, setLegs] = useState<BacktestLeg[]>([createDefaultLeg()]);
  const [rate, setRate] = useState(0.05);
  const [sigma, setSigma] = useState(0.2);

  // Results state
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Leg management
  const handleAddLeg = useCallback(() => {
    setLegs((prev) => [...prev, createDefaultLeg()]);
  }, []);

  const handleRemoveLeg = useCallback((index: number) => {
    setLegs((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleLegChange = useCallback(
    (index: number, field: keyof BacktestLeg, value: string | number) => {
      setLegs((prev) =>
        prev.map((leg, i) =>
          i === index ? { ...leg, [field]: value } : leg,
        ),
      );
    },
    [],
  );

  // Run backtest
  const handleRunBacktest = useCallback(async () => {
    if (legs.length === 0) {
      setError("Add at least one leg to run the backtest");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.backtest({
        ticker,
        legs,
        r: rate,
        sigma,
      });
      setResult(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : "An error occurred";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [ticker, legs, rate, sigma]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Backtest</h1>
      </div>

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* -------------------------------------------------------------- */}
        {/* Left rail - Config                                              */}
        {/* -------------------------------------------------------------- */}
        <aside className="w-full shrink-0 space-y-4 rounded-[var(--radius)] bg-surface p-4 lg:w-80">
          {/* Ticker */}
          <div className="space-y-1.5">
            <span className="mb-3 block text-sm font-medium text-foreground">
              Ticker
            </span>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="h-8 w-full rounded-md border border-input bg-background px-2.5 text-sm font-mono text-foreground uppercase outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
            />
          </div>

          {/* Legs */}
          <div className="space-y-3">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-foreground">Legs</span>
              <Button variant="outline" size="xs" onClick={handleAddLeg}>
                + Add Leg
              </Button>
            </div>

            {legs.length === 0 && (
              <p className="py-3 text-center text-xs text-muted-foreground">
                No legs configured. Click &quot;Add Leg&quot; to start.
              </p>
            )}

            <div className="space-y-3">
              {legs.map((leg, idx) => (
                <LegRow
                  key={idx}
                  leg={leg}
                  index={idx}
                  onChange={handleLegChange}
                  onRemove={handleRemoveLeg}
                />
              ))}
            </div>
          </div>

          {/* Parameters */}
          <div className="space-y-3">
            <span className="mb-3 block text-sm font-medium text-foreground">
              Parameters
            </span>
            <div className="space-y-3">
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

          {/* Run button */}
          <Button
            className="w-full"
            size="lg"
            onClick={handleRunBacktest}
            disabled={loading || legs.length === 0}
          >
            {loading ? "Running..." : "Run Backtest"}
          </Button>

          {error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}
        </aside>

        {/* -------------------------------------------------------------- */}
        {/* Right content - Chart + Metrics                                 */}
        {/* -------------------------------------------------------------- */}
        <div className="min-w-0 flex-1 space-y-6">
          {/* Loading skeleton */}
          {loading && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-20 animate-pulse rounded-[var(--radius)] bg-muted"
                  />
                ))}
              </div>
              <div className="h-[440px] animate-pulse rounded-[var(--radius)] bg-muted" />
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <>
              {/* Risk metrics row */}
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <MetricCard
                  label="Total P&L"
                  value={formatCurrency(result.total_pnl)}
                  colorClass={
                    result.total_pnl >= 0
                      ? "text-positive"
                      : "text-negative"
                  }
                />
                <MetricCard
                  label="Sharpe Ratio"
                  value={
                    result.sharpe_ratio !== null
                      ? result.sharpe_ratio.toFixed(2)
                      : "N/A"
                  }
                  colorClass={
                    result.sharpe_ratio !== null && result.sharpe_ratio >= 1
                      ? "text-positive"
                      : result.sharpe_ratio !== null && result.sharpe_ratio >= 0
                        ? "text-primary"
                        : "text-negative"
                  }
                />
                <MetricCard
                  label="Max Drawdown"
                  value={formatPercent(result.max_drawdown)}
                  colorClass="text-negative"
                />
                <MetricCard
                  label="Win Rate"
                  value={formatPercent(result.win_rate)}
                  colorClass={
                    result.win_rate >= 0.5
                      ? "text-positive"
                      : "text-negative"
                  }
                />
              </div>

              {/* P&L Chart */}
              <div className="border-b border-border pb-6">
                <BaseChart
                  data={[
                    {
                      x: result.pnl_series.map((p) => p.date),
                      y: result.pnl_series.map((p) => p.pnl),
                      type: "scatter",
                      mode: "lines+markers",
                      marker: { size: 4, color: "#4CAF7D" },
                      line: { color: "#4CAF7D", width: 2 },
                      name: "Cumulative P&L",
                      hovertemplate:
                        "Date: %{x}<br>P&L: $%{y:.2f}<extra></extra>",
                    },
                  ]}
                  layout={{
                    xaxis: { title: { text: "Date" }, type: "date" },
                    yaxis: { title: { text: "Cumulative P&L ($)" } },
                    showlegend: false,
                  }}
                  height={440}
                />
              </div>

              {/* Strategy summary */}
              <div className="border-b border-border pb-4">
                <span className="mb-2 block text-sm font-medium text-foreground">
                  Strategy Summary
                </span>
                <div className="space-y-1">
                  <div className="text-xs text-muted-foreground">
                    Ticker:{" "}
                    <span className="font-mono font-semibold text-foreground">
                      {ticker}
                    </span>
                  </div>
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
                      <span className="font-mono text-xs">
                        {leg.qty ?? 1}x {leg.type.toUpperCase()} @ $
                        {leg.strike} exp {leg.expiry}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Empty state */}
          {!result && !loading && (
            <div className="flex min-h-[400px] items-center justify-center">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">
                  Configure your strategy legs and click Run Backtest
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
// LegRow - individual leg configuration
// ---------------------------------------------------------------------------

interface LegRowProps {
  leg: BacktestLeg;
  index: number;
  onChange: (index: number, field: keyof BacktestLeg, value: string | number) => void;
  onRemove: (index: number) => void;
}

function LegRow({ leg, index, onChange, onRemove }: LegRowProps) {
  return (
    <div className="space-y-2 rounded-md border border-border/50 bg-background/50 p-2">
      {/* Header with remove button */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">
          Leg {index + 1}
        </span>
        <button
          type="button"
          onClick={() => onRemove(index)}
          className="rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
        >
          Remove
        </button>
      </div>

      {/* Type + Side */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1 block text-xs text-muted-foreground">
            Type
          </label>
          <select
            value={leg.type}
            onChange={(e) =>
              onChange(index, "type", e.target.value as "call" | "put")
            }
            className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
          >
            <option value="call">Call</option>
            <option value="put">Put</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-muted-foreground">
            Side
          </label>
          <select
            value={leg.side}
            onChange={(e) =>
              onChange(index, "side", e.target.value as "long" | "short")
            }
            className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
          >
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
        </div>
      </div>

      {/* Strike + Qty */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1 block text-xs text-muted-foreground">
            Strike
          </label>
          <input
            type="number"
            value={leg.strike}
            step={1}
            min={0.01}
            onChange={(e) => {
              const parsed = parseFloat(e.target.value);
              if (!Number.isNaN(parsed)) onChange(index, "strike", parsed);
            }}
            className="h-8 w-full rounded-md border border-input bg-background px-2.5 text-sm font-mono text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-muted-foreground">
            Qty
          </label>
          <input
            type="number"
            value={leg.qty ?? 1}
            step={1}
            min={1}
            onChange={(e) => {
              const parsed = parseInt(e.target.value, 10);
              if (!Number.isNaN(parsed) && parsed >= 1)
                onChange(index, "qty", parsed);
            }}
            className="h-8 w-full rounded-md border border-input bg-background px-2.5 text-sm font-mono text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
          />
        </div>
      </div>

      {/* Expiry */}
      <div>
        <label className="mb-1 block text-xs text-muted-foreground">
          Expiry
        </label>
        <input
          type="date"
          value={leg.expiry}
          onChange={(e) => onChange(index, "expiry", e.target.value)}
          className="h-8 w-full rounded-md border border-input bg-background px-2.5 text-sm font-mono text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MetricCard - risk metric display
// ---------------------------------------------------------------------------

function MetricCard({
  label,
  value,
  colorClass,
}: {
  label: string;
  value: string;
  colorClass: string;
}) {
  return (
    <div className="space-y-1">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div
        className={`mt-1 text-lg font-semibold font-mono tabular-nums ${colorClass}`}
      >
        {value}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatCurrency(value: number): string {
  const prefix = value >= 0 ? "+$" : "-$";
  return `${prefix}${Math.abs(value).toFixed(2)}`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
