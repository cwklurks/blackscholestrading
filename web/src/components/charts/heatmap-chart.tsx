"use client";

import { BaseChart } from "./base-chart";

export interface HeatmapChartProps {
  spotValues: number[];
  volValues: number[];
  prices: number[][];
  title?: string;
  className?: string;
  height?: number;
}

export function HeatmapChart({
  spotValues,
  volValues,
  prices,
  title = "Price Grid",
  className,
  height = 400,
}: HeatmapChartProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    {
      type: "heatmap" as const,
      x: spotValues,
      y: volValues,
      z: prices,
      colorscale: "Viridis",
      colorbar: {
        title: { text: "Price", side: "right" },
        tickfont: { color: "hsl(0 0% 70%)" },
      },
    },
  ];

  return (
    <BaseChart
      data={data}
      layout={{
        title: { text: title, font: { size: 14 } },
        xaxis: { title: { text: "Spot Price" } },
        yaxis: { title: { text: "Volatility" } },
      }}
      className={className}
      height={height}
    />
  );
}
