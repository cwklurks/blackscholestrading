"use client";

import {
  Calculator,
  Activity,
  Layers,
  BarChart3,
  LineChart,
  TrendingUp,
} from "lucide-react";

export interface Tool {
  id: string;
  label: string;
  icon: typeof Calculator;
}

export const TOOLS: Tool[] = [
  { id: "pricing", label: "Pricing", icon: Calculator },
  { id: "sensitivity", label: "Sensitivity", icon: TrendingUp },
  { id: "volatility", label: "Vol Surface", icon: Activity },
  { id: "strategies", label: "Strategies", icon: Layers },
  { id: "backtest", label: "Backtest", icon: BarChart3 },
  { id: "market", label: "Market", icon: LineChart },
];

interface ToolBarProps {
  activeToolId: string;
  onToolChange: (id: string) => void;
}

export function ToolBar({ activeToolId, onToolChange }: ToolBarProps) {
  return (
    <div className="flex items-center border-b border-border">
      {/* Brand mark */}
      <div className="mr-6 flex items-baseline gap-2 py-3">
        <span className="font-mono text-lg font-bold tracking-tight text-primary">
          BST
        </span>
        <span className="hidden text-xs text-muted-foreground sm:inline">
          Black-Scholes Trader
        </span>
      </div>

      {/* Tool tabs */}
      <div className="flex items-center gap-0.5">
        {TOOLS.map(({ id, label, icon: Icon }) => {
          const isActive = activeToolId === id;

          return (
            <button
              key={id}
              onClick={() => onToolChange(id)}
              className={[
                "flex items-center gap-2 border-b-2 px-4 py-3 text-[13px] font-medium transition-colors duration-100",
                isActive
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              ].join(" ")}
            >
              <Icon className="h-4 w-4" />
              <span>{label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
