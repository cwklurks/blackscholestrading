"use client";

import { BaseChart } from "./base-chart";

const POSITIVE = "hsl(142 70% 45%)";
const NEGATIVE = "hsl(0 72% 51%)";
const ACCENT = "#3b82f6";

export interface PayoffChartProps {
  prices: number[];
  pnl: number[];
  breakevens?: number[];
  title?: string;
  className?: string;
  height?: number;
}

export function PayoffChart({
  prices,
  pnl,
  breakevens = [],
  title = "P&L at Expiration",
  className,
  height,
}: PayoffChartProps) {
  const positiveY = pnl.map((v) => (v >= 0 ? v : null));
  const negativeY = pnl.map((v) => (v < 0 ? v : null));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    {
      type: "scatter" as const,
      mode: "lines" as const,
      x: prices,
      y: positiveY,
      line: { color: POSITIVE, width: 2 },
      fill: "tozeroy" as const,
      fillcolor: `${POSITIVE}1a`,
      name: "Profit",
      connectgaps: false,
    },
    {
      type: "scatter" as const,
      mode: "lines" as const,
      x: prices,
      y: negativeY,
      line: { color: NEGATIVE, width: 2 },
      fill: "tozeroy" as const,
      fillcolor: `${NEGATIVE}1a`,
      name: "Loss",
      connectgaps: false,
    },
  ];

  if (breakevens.length > 0) {
    const breakevenPnl = breakevens.map(() => 0);
    data.push({
      type: "scatter" as const,
      mode: "markers+text" as const,
      x: breakevens,
      y: breakevenPnl,
      marker: {
        symbol: "diamond",
        size: 10,
        color: ACCENT,
      },
      text: breakevens.map((b) => `$${b.toFixed(2)}`),
      textposition: "top center",
      textfont: { color: ACCENT, size: 11 },
      name: "Breakeven",
      showlegend: true,
    });
  }

  return (
    <BaseChart
      data={data}
      layout={{
        title: { text: title, font: { size: 14 } },
        xaxis: { title: { text: "Underlying Price" } },
        yaxis: { title: { text: "Profit / Loss" } },
        shapes: [
          {
            type: "line",
            x0: prices[0],
            x1: prices[prices.length - 1],
            y0: 0,
            y1: 0,
            line: { color: "hsl(0 0% 30%)", width: 1, dash: "dot" },
          },
        ],
        showlegend: true,
        legend: {
          orientation: "h",
          y: -0.2,
          x: 0.5,
          xanchor: "center",
        },
      }}
      className={className}
      height={height}
    />
  );
}
