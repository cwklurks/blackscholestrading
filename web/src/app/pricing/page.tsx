"use client";

import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { SensitivityChart } from "@/components/charts/sensitivity-chart";
import { HeatmapChart } from "@/components/charts/heatmap-chart";
import { BaseChart, GREEK_COLORS } from "@/components/charts/base-chart";
import { formatPrice, formatGreek } from "@/lib/format";
import { AdvancedParams, type AdvancedParamsValues } from "@/components/pricing/advanced-params";
import {
  ModelParamPanel,
  getDefaultModelParams,
  MC_MODELS,
} from "@/components/pricing/model-param-panel";
import { api } from "@/lib/api";
import { ParamInput } from "@/components/ui/param-input";
import type {
  PricingResponse,
  HeatmapResponse,
  MonteCarloResponse,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// Available models
// ---------------------------------------------------------------------------

const MODELS = [
  "Black-Scholes",
  "Binomial",
  "Heston MC",
  "GARCH MC",
  "Bates Jump-Diffusion",
] as const;

type Model = (typeof MODELS)[number];

// ---------------------------------------------------------------------------
// Sensitivity data shape
// ---------------------------------------------------------------------------

interface SensitivityData {
  spots: number[];
  delta: number[];
  gamma: number[];
  vega: number[];
  theta: number[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PricingPage() {
  // Basic params
  const [spot, setSpot] = useState(100);
  const [strike, setStrike] = useState(100);
  const [tte, setTte] = useState(0.25);
  const [rate, setRate] = useState(0.05);
  const [sigma, setSigma] = useState(0.2);
  const [optionType, setOptionType] = useState<"call" | "put">("call");
  const [model, setModel] = useState<Model>("Black-Scholes");

  // Advanced params
  const [advanced, setAdvanced] = useState<AdvancedParamsValues>({
    q: 0,
    borrowCost: 0,
  });

  // Model-specific params
  const [modelParams, setModelParams] = useState<Record<string, number>>(() =>
    getDefaultModelParams("Black-Scholes"),
  );

  // Results
  const [priceResult, setPriceResult] = useState<PricingResponse | null>(null);
  const [sensitivity, setSensitivity] = useState<SensitivityData | null>(null);
  const [heatmap, setHeatmap] = useState<HeatmapResponse | null>(null);
  const [mcResult, setMcResult] = useState<MonteCarloResponse | null>(null);

  // Loading / error
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Active sensitivity tab
  const [activeGreek, setActiveGreek] = useState(0);

  // Handle model change - reset model params
  function handleModelChange(newModel: string) {
    setModel(newModel as Model);
    setModelParams(getDefaultModelParams(newModel));
  }

  // Build the common request body for api.price()
  const buildPriceRequest = useCallback(
    (overrideSpot?: number) => ({
      model,
      S: overrideSpot ?? spot,
      K: strike,
      T: tte,
      r: rate,
      sigma,
      q: advanced.q || undefined,
      borrow_cost: advanced.borrowCost || undefined,
      option_type: optionType,
      model_params: Object.keys(modelParams).length > 0 ? modelParams : undefined,
    }),
    [model, spot, strike, tte, rate, sigma, advanced, optionType, modelParams],
  );

  // Main pricing action
  async function handlePrice() {
    setLoading(true);
    setError(null);

    try {
      // 1. Price the single option
      const result = await api.price(buildPriceRequest());
      setPriceResult(result);

      // 2. Generate sensitivity data (~20 spot points)
      const spotMin = spot * 0.7;
      const spotMax = spot * 1.3;
      const numPoints = 20;
      const step = (spotMax - spotMin) / (numPoints - 1);
      const spots = Array.from({ length: numPoints }, (_, i) => spotMin + i * step);

      const sensitivityResults = await Promise.all(
        spots.map((s) => api.price(buildPriceRequest(s))),
      );

      setSensitivity({
        spots,
        delta: sensitivityResults.map((r) => r.delta),
        gamma: sensitivityResults.map((r) => r.gamma),
        vega: sensitivityResults.map((r) => r.vega),
        theta: sensitivityResults.map((r) => r.theta),
      });

      // 3. Heatmap
      const heatmapResult = await api.heatmap({
        K: strike,
        T: tte,
        r: rate,
        q: advanced.q || undefined,
        borrow_cost: advanced.borrowCost || undefined,
        spot_range: { min: spot * 0.7, max: spot * 1.3, steps: 15 },
        vol_range: { min: Math.max(0.05, sigma * 0.5), max: sigma * 2, steps: 15 },
      });
      setHeatmap(heatmapResult);

      // 4. MC histogram (only for MC models)
      if (MC_MODELS.has(model)) {
        const mcPaths = modelParams.mc_paths ?? 10000;
        const mcRes = await api.monteCarlo({
          S: spot,
          K: strike,
          T: tte,
          r: rate,
          sigma,
          paths: mcPaths,
          option_type: optionType,
          q: advanced.q || undefined,
          borrow_cost: advanced.borrowCost || undefined,
        });
        setMcResult(mcRes);
      } else {
        setMcResult(null);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "An error occurred";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const greekTabs = ["Delta", "Gamma", "Vega", "Theta"] as const;
  const greekKeys = ["delta", "gamma", "vega", "theta"] as const;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Pricing</h1>
      </div>

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* ---------------------------------------------------------------- */}
        {/* Left rail - Parameters                                           */}
        {/* ---------------------------------------------------------------- */}
        <aside className="w-full shrink-0 space-y-4 border-r border-border bg-surface p-4 lg:w-72 lg:pr-6">
          {/* Model selector */}
          <div className="space-y-1.5">
            <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
              Model
            </label>
            <select
              value={model}
              onChange={(e) => handleModelChange(e.target.value)}
              className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none transition-colors focus:border-ring focus:ring-1 focus:ring-ring"
            >
              {MODELS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Basic params */}
          <div className="space-y-3">
            <span className="mb-3 block text-sm font-medium text-foreground">
              Basic Parameters
            </span>

            <div className="space-y-3">
              <ParamInput label="Spot (S)" value={spot} step={1} min={0.01} onChange={setSpot} />
              <ParamInput label="Strike (K)" value={strike} step={1} min={0.01} onChange={setStrike} />
              <ParamInput label="Time to Expiry (T)" value={tte} step={0.01} min={0.001} max={10} onChange={setTte} />
              <ParamInput label="Risk-free Rate (r)" value={rate} step={0.005} min={0} max={1} onChange={setRate} />
              <ParamInput label="Volatility (sigma)" value={sigma} step={0.01} min={0.001} max={5} onChange={setSigma} />

              <div>
                <label className="mb-1 block text-xs text-muted-foreground">
                  Option Type
                </label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setOptionType("call")}
                    className={`flex-1 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                      optionType === "call"
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input bg-background text-muted-foreground hover:bg-accent/50"
                    }`}
                  >
                    Call
                  </button>
                  <button
                    type="button"
                    onClick={() => setOptionType("put")}
                    className={`flex-1 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                      optionType === "put"
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input bg-background text-muted-foreground hover:bg-accent/50"
                    }`}
                  >
                    Put
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Advanced params */}
          <AdvancedParams values={advanced} onChange={setAdvanced} />

          {/* Model-specific params */}
          <ModelParamPanel
            model={model}
            values={modelParams}
            onChange={setModelParams}
          />

          {/* Price button */}
          <Button
            className="w-full"
            size="lg"
            onClick={handlePrice}
            disabled={loading}
          >
            {loading ? "Pricing..." : "Price Option"}
          </Button>

          {error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}
        </aside>

        {/* ---------------------------------------------------------------- */}
        {/* Right content - Results                                          */}
        {/* ---------------------------------------------------------------- */}
        <div className="min-w-0 flex-1 space-y-6">
          {/* Price result card */}
          {priceResult && (
            <div className="border-b border-border pb-4">
              <div className="mb-3 text-sm font-medium text-muted-foreground">
                {priceResult.model} - {optionType.toUpperCase()}
              </div>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
                <ResultStat label="Price" value={priceResult.price} format={formatPrice} />
                <ResultStat label="Delta" value={priceResult.delta} format={(v) => formatGreek("delta", v)} />
                <ResultStat label="Gamma" value={priceResult.gamma} format={(v) => formatGreek("gamma", v)} />
                <ResultStat label="Theta" value={priceResult.theta} format={(v) => formatGreek("theta", v)} />
                <ResultStat label="Vega" value={priceResult.vega} format={(v) => formatGreek("vega", v)} />
                <ResultStat label="Rho" value={priceResult.rho} format={(v) => formatGreek("rho", v)} />
              </div>
            </div>
          )}

          {/* Sensitivity charts with tabs */}
          {sensitivity && (
            <div className="border-b border-border pb-6">
              <Tabs defaultValue={0} value={activeGreek} onValueChange={setActiveGreek}>
                <TabsList>
                  {greekTabs.map((name, idx) => (
                    <TabsTrigger key={name} value={idx}>
                      {name}
                    </TabsTrigger>
                  ))}
                </TabsList>

                {greekTabs.map((name, idx) => {
                  const key = greekKeys[idx];
                  return (
                    <TabsContent key={name} value={idx}>
                      <SensitivityChart
                        x={sensitivity.spots}
                        y={sensitivity[key]}
                        title={`${name} vs Spot`}
                        xLabel="Spot Price"
                        yLabel={name}
                        color={GREEK_COLORS[key]}
                        height={340}
                      />
                    </TabsContent>
                  );
                })}
              </Tabs>
            </div>
          )}

          {/* Heatmap */}
          {heatmap && (
            <div className="border-b border-border pb-6">
              <HeatmapChart
                spotValues={heatmap.spot_values}
                volValues={heatmap.vol_values}
                prices={
                  optionType === "call"
                    ? heatmap.call_prices
                    : heatmap.put_prices
                }
                title={`${optionType === "call" ? "Call" : "Put"} Price Heatmap (Spot x Vol)`}
                height={400}
              />
            </div>
          )}

          {/* MC Histogram - only for MC models */}
          {mcResult && MC_MODELS.has(model) && (
            <div className="border-b border-border pb-6">
              <div className="mb-3 text-sm font-medium text-foreground">
                Monte Carlo Distribution
              </div>

              {/* Stats row */}
              <div className="mb-4 grid grid-cols-3 gap-4">
                <ResultStat label="MC Price" value={mcResult.price} />
                <ResultStat label="Std Error" value={mcResult.std_error} />
                <div>
                  <div className="text-xs text-muted-foreground">
                    95% CI
                  </div>
                  <div className="text-lg font-semibold tabular-nums text-foreground">
                    [{mcResult.confidence_interval[0]?.toFixed(4)},{" "}
                    {mcResult.confidence_interval[1]?.toFixed(4)}]
                  </div>
                </div>
              </div>

              {/* Histogram */}
              <BaseChart
                data={[
                  {
                    type: "histogram",
                    x: mcResult.terminal_prices,
                    nbinsx: 60,
                    marker: { color: `${GREEK_COLORS.delta}80`, line: { color: GREEK_COLORS.delta, width: 1 } },
                    name: "Terminal Prices",
                  },
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                ] as any}
                layout={{
                  title: { text: "Terminal Price Distribution", font: { size: 14 } },
                  xaxis: { title: { text: "Terminal Price" } },
                  yaxis: { title: { text: "Frequency" } },
                  bargap: 0.02,
                  shapes: [
                    {
                      type: "line",
                      x0: strike,
                      x1: strike,
                      y0: 0,
                      y1: 1,
                      yref: "paper",
                      line: { color: GREEK_COLORS.theta, width: 2, dash: "dash" },
                    },
                  ],
                  annotations: [
                    {
                      x: strike,
                      y: 1,
                      yref: "paper",
                      text: `K=${strike}`,
                      showarrow: false,
                      font: { color: GREEK_COLORS.theta, size: 11 },
                      yanchor: "bottom",
                    },
                  ],
                }}
                height={340}
              />
            </div>
          )}

          {/* Empty state */}
          {!priceResult && !loading && (
            <div className="flex min-h-[400px] items-center justify-center">
              <div className="text-center">
                <p className="font-mono text-3xl font-bold tabular-nums text-muted-foreground/30">
                  $0.00
                </p>
                <p className="mt-3 text-sm text-muted-foreground">
                  Configure parameters and press Enter or click Price Option
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
// ResultStat - small stat display
// ---------------------------------------------------------------------------

function ResultStat({ label, value, format }: { label: string; value: number; format?: (v: number) => string }) {
  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="font-mono text-lg font-medium tabular-nums text-foreground">
        {format ? format(value) : value.toFixed(4)}
      </div>
    </div>
  );
}
