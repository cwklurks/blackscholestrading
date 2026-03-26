"use client";

import { useState, useCallback, useMemo } from "react";
import { useVolSurface } from "@/hooks/use-volatility";
import { SurfaceChart } from "@/components/charts/surface-chart";
import { BaseChart } from "@/components/charts/base-chart";
import { Button } from "@/components/ui/button";
import type { VolSurfacePoint } from "@/lib/types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SMILE_COLORS = [
  "hsl(210, 100%, 60%)",
  "hsl(150, 80%, 50%)",
  "hsl(45, 100%, 55%)",
  "hsl(0, 85%, 60%)",
  "hsl(280, 80%, 65%)",
  "hsl(180, 70%, 50%)",
  "hsl(30, 95%, 55%)",
  "hsl(330, 80%, 60%)",
  "hsl(120, 60%, 50%)",
  "hsl(60, 90%, 50%)",
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface SurfaceMatrix {
  strikes: number[];
  expiries: string[];
  ivMatrix: number[][];
}

function buildSurfaceMatrix(points: VolSurfacePoint[]): SurfaceMatrix {
  const expirySet = new Set<string>();
  const strikeSet = new Set<number>();

  for (const pt of points) {
    expirySet.add(pt.expiry);
    strikeSet.add(pt.strike);
  }

  const expiries = Array.from(expirySet).sort();
  const strikes = Array.from(strikeSet).sort((a, b) => a - b);

  // Build a lookup map: "expiry|strike" -> iv
  const lookup = new Map<string, number | null>();
  for (const pt of points) {
    lookup.set(`${pt.expiry}|${pt.strike}`, pt.iv);
  }

  // ivMatrix[row=expiry][col=strike]
  const ivMatrix = expiries.map((exp) =>
    strikes.map((k) => {
      const iv = lookup.get(`${exp}|${k}`);
      return iv ?? 0;
    }),
  );

  return { strikes, expiries, ivMatrix };
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function CoverageBadge({ coverage }: { coverage: number }) {
  const pct = Math.round(coverage * 100);
  const isSparse = coverage < 0.4;

  return (
    <div className="flex items-center gap-3">
      <span
        className={`inline-flex items-center rounded-md px-2.5 py-1 text-sm font-medium ${
          isSparse
            ? "bg-destructive/10 text-destructive"
            : "bg-primary/10 text-primary"
        }`}
      >
        {pct}% coverage
      </span>
      {isSparse && (
        <span className="text-sm text-destructive">
          Sparse data - surface may be unreliable
        </span>
      )}
    </div>
  );
}

interface SmileChartProps {
  smileData: Record<string, Array<Record<string, unknown>>>;
}

function SmileChart({ smileData }: SmileChartProps) {
  const traces = useMemo(() => {
    const entries = Object.entries(smileData);
    return entries.map(([expiry, points], idx) => {
      const strikes = points.map((p) => Number(p.strike ?? p.Strike ?? 0));
      const ivs = points.map((p) => Number(p.iv ?? p.IV ?? 0));

      return {
        type: "scatter" as const,
        mode: "lines+markers" as const,
        name: expiry,
        x: strikes,
        y: ivs,
        line: {
          color: SMILE_COLORS[idx % SMILE_COLORS.length],
          width: 2,
        },
        marker: {
          size: 4,
          color: SMILE_COLORS[idx % SMILE_COLORS.length],
        },
      };
    });
  }, [smileData]);

  if (traces.length === 0) {
    return null;
  }

  return (
    <BaseChart
      data={traces}
      layout={{
        title: { text: "Volatility Smile by Expiry", font: { size: 14 } },
        xaxis: {
          title: { text: "Strike" },
          gridcolor: "hsl(0 0% 18%)",
        },
        yaxis: {
          title: { text: "IV" },
          gridcolor: "hsl(0 0% 18%)",
          tickformat: ".0%",
        },
        showlegend: true,
        legend: {
          font: { color: "hsl(0 0% 70%)", size: 11 },
          bgcolor: "transparent",
        },
      }}
      height={400}
    />
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-[480px] animate-pulse rounded-lg bg-muted" />
      <div className="h-[400px] animate-pulse rounded-lg bg-muted" />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-64 items-center justify-center rounded-lg border border-border bg-surface">
      <p className="text-text-secondary">
        Enter a ticker and click Load Surface
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

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function VolatilityPage() {
  const [ticker, setTicker] = useState("AAPL");
  const { trigger, data, error, isMutating } = useVolSurface();

  const handleLoad = useCallback(() => {
    const trimmed = ticker.trim().toUpperCase();
    if (trimmed.length === 0) return;
    trigger({ ticker: trimmed });
  }, [ticker, trigger]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        handleLoad();
      }
    },
    [handleLoad],
  );

  const surfaceMatrix = useMemo(() => {
    if (!data?.surface || data.surface.length === 0) return null;
    return buildSurfaceMatrix(data.surface);
  }, [data?.surface]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Volatility</h1>
        <p className="mt-1 text-text-secondary">
          Implied volatility surface, smile curves, and term structure
        </p>
      </div>

      {/* Ticker Input + Fetch */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ticker symbol"
          className="h-8 w-32 rounded-lg border border-border bg-background px-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/50"
        />
        <Button onClick={handleLoad} disabled={isMutating || ticker.trim().length === 0}>
          {isMutating ? "Loading..." : "Load Surface"}
        </Button>
      </div>

      {/* States */}
      {error && <ErrorMessage message={error.message} />}

      {isMutating && <LoadingSkeleton />}

      {!isMutating && !data && !error && <EmptyState />}

      {!isMutating && data && (
        <div className="space-y-6">
          {/* Coverage Badge */}
          <CoverageBadge coverage={data.coverage} />

          {/* 3D Volatility Surface */}
          {surfaceMatrix && (
            <div className="rounded-lg border border-border bg-surface p-4">
              <SurfaceChart
                strikes={surfaceMatrix.strikes}
                expiries={surfaceMatrix.expiries}
                ivMatrix={surfaceMatrix.ivMatrix}
                height={480}
              />
            </div>
          )}

          {/* Smile Curves */}
          {data.smile_data && Object.keys(data.smile_data).length > 0 && (
            <div className="rounded-lg border border-border bg-surface p-4">
              <SmileChart smileData={data.smile_data} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
