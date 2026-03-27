"use client";

import { BaseChart } from "./base-chart";

const ACCENT = "#3b82f6";

export interface SensitivityChartProps {
  x: number[];
  y: number[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
  color?: string;
  className?: string;
  height?: number;
}

export function SensitivityChart({
  x,
  y,
  title,
  xLabel,
  yLabel,
  color = ACCENT,
  className,
  height,
}: SensitivityChartProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    {
      type: "scatter" as const,
      mode: "lines" as const,
      x,
      y,
      line: { color, width: 2 },
      fill: "tozeroy" as const,
      fillcolor: `${color}1a`,
    },
  ];

  return (
    <BaseChart
      data={data}
      layout={{
        title: title ? { text: title, font: { size: 14 } } : undefined,
        xaxis: { title: xLabel ? { text: xLabel } : undefined },
        yaxis: { title: yLabel ? { text: yLabel } : undefined },
      }}
      className={className}
      height={height}
    />
  );
}
