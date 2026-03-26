"use client";

import { BaseChart } from "./base-chart";

export interface SurfaceChartProps {
  strikes: number[];
  expiries: string[];
  ivMatrix: number[][];
  title?: string;
  className?: string;
  height?: number;
}

export function SurfaceChart({
  strikes,
  expiries,
  ivMatrix,
  title = "Implied Volatility Surface",
  className,
  height = 480,
}: SurfaceChartProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    {
      type: "surface" as const,
      x: strikes,
      y: expiries,
      z: ivMatrix,
      colorscale: "Portland",
      colorbar: {
        title: { text: "IV", side: "right" },
        tickfont: { color: "hsl(0 0% 70%)" },
        tickformat: ".0%",
      },
      contours: {
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: "hsl(0 0% 50%)",
          project: { z: false },
        },
      },
    },
  ];

  return (
    <BaseChart
      data={data}
      layout={{
        title: { text: title, font: { size: 14 } },
        scene: {
          xaxis: {
            title: { text: "Strike" },
            gridcolor: "hsl(0 0% 18%)",
          },
          yaxis: {
            title: { text: "Expiry" },
            gridcolor: "hsl(0 0% 18%)",
          },
          zaxis: {
            title: { text: "IV" },
            gridcolor: "hsl(0 0% 18%)",
            tickformat: ".0%",
          },
          bgcolor: "transparent",
          camera: {
            eye: { x: 1.5, y: -1.5, z: 1.2 },
          },
        },
      }}
      className={className}
      height={height}
    />
  );
}
