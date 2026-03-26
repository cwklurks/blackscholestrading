"use client";

import { BaseChart } from "./base-chart";

const POSITIVE = "hsl(142 70% 45%)";
const NEGATIVE = "hsl(0 72% 51%)";

export interface CandlestickChartProps {
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume?: number[];
  title?: string;
  className?: string;
  height?: number;
}

export function CandlestickChart({
  dates,
  open,
  high,
  low,
  close,
  volume,
  title = "Price",
  className,
  height = 400,
}: CandlestickChartProps) {
  const showVolume = volume && volume.length > 0;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    {
      type: "candlestick" as const,
      x: dates,
      open,
      high,
      low,
      close,
      increasing: { line: { color: POSITIVE }, fillcolor: POSITIVE },
      decreasing: { line: { color: NEGATIVE }, fillcolor: NEGATIVE },
      name: "OHLC",
      xaxis: "x",
      yaxis: showVolume ? "y2" : "y",
    },
  ];

  if (showVolume) {
    const volumeColors = close.map((c, i) =>
      c >= open[i] ? POSITIVE : NEGATIVE
    );

    data.push({
      type: "bar" as const,
      x: dates,
      y: volume,
      marker: {
        color: volumeColors,
        opacity: 0.4,
      },
      name: "Volume",
      xaxis: "x",
      yaxis: "y",
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const layout: Record<string, any> = {
    title: { text: title, font: { size: 14 } },
    xaxis: {
      rangeslider: { visible: false },
      type: "date" as const,
    },
    showlegend: false,
  };

  if (showVolume) {
    layout.yaxis = {
      title: { text: "Volume" },
      domain: [0, 0.25],
      gridcolor: "hsl(0 0% 18%)",
      zerolinecolor: "hsl(0 0% 18%)",
    };
    layout.yaxis2 = {
      title: { text: "Price" },
      domain: [0.3, 1],
      gridcolor: "hsl(0 0% 18%)",
      zerolinecolor: "hsl(0 0% 18%)",
    };
    layout.margin = { l: 48, r: 16, t: 32, b: 40 };
  } else {
    layout.yaxis = {
      title: { text: "Price" },
    };
  }

  return (
    <BaseChart
      data={data}
      layout={layout}
      className={className}
      height={height}
    />
  );
}
