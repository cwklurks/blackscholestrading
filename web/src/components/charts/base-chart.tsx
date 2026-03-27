"use client";

import dynamic from "next/dynamic";
import type { Config, Data, Layout } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const DEFAULT_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: {
    family: "var(--font-geist-sans), Geist Sans, system-ui, sans-serif",
    color: "#8A8580",
  },
  margin: { l: 48, r: 16, t: 32, b: 40 },
  xaxis: {
    gridcolor: "#2A2725",
    zerolinecolor: "#2A2725",
    tickfont: { family: "var(--font-geist-mono), Geist Mono, monospace" },
  },
  yaxis: {
    gridcolor: "#2A2725",
    zerolinecolor: "#2A2725",
    tickfont: { family: "var(--font-geist-mono), Geist Mono, monospace" },
  },
};

const DEFAULT_CONFIG: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};

/** Semantic Greek colors from DESIGN.md — use in chart traces */
export const GREEK_COLORS: Record<string, string> = {
  delta: "#5B8DEF",
  gamma: "#45B899",
  vega: "#D4A017",
  theta: "#E05252",
  rho: "#9B8EC4",
};

function deepMergeAxis(
  base: Record<string, unknown> | undefined,
  override: Record<string, unknown> | undefined
): Record<string, unknown> | undefined {
  if (!override) return base;
  if (!base) return override;
  return { ...base, ...override };
}

function mergeLayouts(
  base: Partial<Layout>,
  override: Partial<Layout>
): Partial<Layout> {
  const merged = { ...base, ...override };

  // Deep merge axis objects so grid colors aren't lost
  if (base.xaxis || override.xaxis) {
    merged.xaxis = deepMergeAxis(
      base.xaxis as Record<string, unknown> | undefined,
      override.xaxis as Record<string, unknown> | undefined
    ) as Partial<Layout>["xaxis"];
  }
  if (base.yaxis || override.yaxis) {
    merged.yaxis = deepMergeAxis(
      base.yaxis as Record<string, unknown> | undefined,
      override.yaxis as Record<string, unknown> | undefined
    ) as Partial<Layout>["yaxis"];
  }

  return merged;
}

export interface BaseChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  height?: number;
}

export function BaseChart({
  data,
  layout = {},
  config = {},
  className,
  height = 320,
}: BaseChartProps) {
  const mergedLayout = mergeLayouts(DEFAULT_LAYOUT, {
    ...layout,
    height,
  });

  const mergedConfig: Partial<Config> = {
    ...DEFAULT_CONFIG,
    ...config,
  };

  return (
    <Plot
      data={data}
      layout={mergedLayout}
      config={mergedConfig}
      className={className}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
