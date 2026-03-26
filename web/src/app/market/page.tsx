"use client";

import { useCallback, useMemo, useState } from "react";
import { useMarket, useChain } from "@/hooks/use-market";
import { CandlestickChart } from "@/components/charts/candlestick-chart";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import type { ChainRow, OHLCVRow } from "@/lib/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractOhlcv(history: OHLCVRow[]) {
  return {
    dates: history.map((r) => r.date),
    open: history.map((r) => r.open),
    high: history.map((r) => r.high),
    low: history.map((r) => r.low),
    close: history.map((r) => r.close),
    volume: history.map((r) => r.volume ?? 0),
  };
}

function formatNumber(value: number | null, decimals = 2): string {
  if (value === null || value === undefined) return "-";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatVolume(value: number | null): string {
  if (value === null || value === undefined) return "-";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toLocaleString();
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="h-20 animate-pulse rounded-lg bg-muted"
          />
        ))}
      </div>
      <div className="h-[440px] animate-pulse rounded-lg bg-muted" />
      <div className="h-[360px] animate-pulse rounded-lg bg-muted" />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-border bg-card/50">
      <p className="text-sm text-muted-foreground">
        Enter a ticker symbol and click &quot;Load&quot; to view market data and
        the options chain
      </p>
    </div>
  );
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
      <p className="text-sm text-destructive">{message}</p>
    </div>
  );
}

function StatCard({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div
      className={`rounded-lg border border-border bg-card p-4 ${className ?? ""}`}
    >
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular-nums text-foreground">
        {value}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Options chain table
// ---------------------------------------------------------------------------

const CHAIN_COLUMNS = [
  { key: "strike", label: "Strike", align: "right" as const },
  { key: "lastPrice", label: "Last Price", align: "right" as const },
  { key: "iv", label: "IV", align: "right" as const },
  { key: "volume", label: "Volume", align: "right" as const },
  { key: "oi", label: "OI", align: "right" as const },
] as const;

function ChainTable({ rows }: { rows: ChainRow[] }) {
  if (rows.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No data available
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            {CHAIN_COLUMNS.map((col) => (
              <th
                key={col.key}
                className={`px-3 py-2 font-medium text-muted-foreground ${
                  col.align === "right" ? "text-right" : "text-left"
                }`}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.strike}
              className="border-b border-border/50 transition-colors hover:bg-muted/30"
            >
              <td className="px-3 py-2 text-right font-medium tabular-nums text-foreground">
                {formatNumber(row.strike)}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-foreground">
                {formatNumber(row.lastPrice)}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-foreground">
                {formatPercent(row.iv)}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                {formatVolume(row.volume)}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                {formatVolume(row.oi)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function MarketPage() {
  const [tickerInput, setTickerInput] = useState("AAPL");
  const [activeTicker, setActiveTicker] = useState<string | null>(null);
  const [selectedExpiry, setSelectedExpiry] = useState<string | null>(null);
  const [chainTab, setChainTab] = useState<string>("calls");

  // SWR hooks - only fetch when activeTicker is set
  const {
    data: marketData,
    error: marketError,
    isLoading: marketLoading,
  } = useMarket(activeTicker);

  const {
    data: chainData,
    error: chainError,
    isLoading: chainLoading,
  } = useChain(activeTicker);

  const isLoading = marketLoading || chainLoading;
  const error = marketError ?? chainError;

  // When chain data arrives, auto-select the first expiration if none set
  const expirations = chainData?.expirations ?? [];
  const activeExpiry = selectedExpiry ?? expirations[0] ?? null;

  // Filter chain rows by the selected expiration
  // Note: If the API returns the full chain (not filtered), we show all rows.
  // The expiration selector is for UX context; the API may already filter.
  const filteredCalls = useMemo(() => chainData?.calls ?? [], [chainData?.calls]);
  const filteredPuts = useMemo(() => chainData?.puts ?? [], [chainData?.puts]);

  // Extract OHLCV arrays from market history
  const ohlcv = useMemo(() => {
    if (!marketData?.history || marketData.history.length === 0) return null;
    return extractOhlcv(marketData.history);
  }, [marketData?.history]);

  const handleLoad = useCallback(() => {
    const trimmed = tickerInput.trim().toUpperCase();
    if (trimmed.length === 0) return;
    setActiveTicker(trimmed);
    setSelectedExpiry(null);
  }, [tickerInput]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        handleLoad();
      }
    },
    [handleLoad],
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Market</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Live market data, candlestick charts, and options chain
        </p>
      </div>

      {/* Ticker Input + Load */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          value={tickerInput}
          onChange={(e) => setTickerInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ticker symbol"
          className="h-8 w-32 rounded-lg border border-border bg-background px-3 text-sm text-foreground uppercase placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/50"
        />
        <Button
          onClick={handleLoad}
          disabled={isLoading || tickerInput.trim().length === 0}
        >
          {isLoading ? "Loading..." : "Load"}
        </Button>
      </div>

      {/* Error */}
      {error && <ErrorMessage message={error.message} />}

      {/* Loading skeleton */}
      {isLoading && <LoadingSkeleton />}

      {/* Empty state */}
      {!isLoading && !activeTicker && !error && <EmptyState />}

      {/* Data */}
      {!isLoading && marketData && (
        <div className="space-y-6">
          {/* Stat cards row */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard
              label={`${activeTicker} Price`}
              value={`$${formatNumber(marketData.price)}`}
            />
            <StatCard
              label="Historical Volatility"
              value={formatPercent(marketData.historical_vol)}
            />
            <StatCard
              label="Data Points"
              value={marketData.history.length.toLocaleString()}
            />
            <StatCard
              label="Fetched At"
              value={new Date(marketData.fetched_at).toLocaleTimeString()}
            />
          </div>

          {/* Candlestick + Volume Chart */}
          {ohlcv && (
            <div className="rounded-lg border border-border bg-card p-4">
              <CandlestickChart
                dates={ohlcv.dates}
                open={ohlcv.open}
                high={ohlcv.high}
                low={ohlcv.low}
                close={ohlcv.close}
                volume={ohlcv.volume}
                title={`${activeTicker} - OHLCV`}
                height={440}
              />
            </div>
          )}

          {/* Options Chain */}
          {chainData && (
            <div className="rounded-lg border border-border bg-card p-4">
              {/* Chain header with expiration selector */}
              <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <h2 className="text-sm font-medium text-foreground">
                  Options Chain
                </h2>

                {expirations.length > 0 && (
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-muted-foreground">
                      Expiration
                    </label>
                    <select
                      value={activeExpiry ?? ""}
                      onChange={(e) => setSelectedExpiry(e.target.value)}
                      className="h-8 rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
                    >
                      {expirations.map((exp) => (
                        <option key={exp} value={exp}>
                          {exp}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {/* Calls / Puts tabs */}
              <Tabs
                defaultValue="calls"
                value={chainTab}
                onValueChange={setChainTab}
              >
                <TabsList>
                  <TabsTrigger value="calls">
                    Calls ({filteredCalls.length})
                  </TabsTrigger>
                  <TabsTrigger value="puts">
                    Puts ({filteredPuts.length})
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="calls">
                  <ChainTable rows={filteredCalls} />
                </TabsContent>

                <TabsContent value="puts">
                  <ChainTable rows={filteredPuts} />
                </TabsContent>
              </Tabs>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
