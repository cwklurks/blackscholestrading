"use client";

import {
  Calculator,
  Activity,
  Layers,
  BarChart3,
  LineChart,
  TrendingUp,
} from "lucide-react";
import type { ReactNode } from "react";

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
  children?: ReactNode;
}

export function ToolBar({ activeToolId, onToolChange }: ToolBarProps) {
  return (
    <div className="flex items-center gap-1 border-b border-border px-1 pb-0">
      {TOOLS.map(({ id, label, icon: Icon }) => {
        const isActive = activeToolId === id;

        return (
          <button
            key={id}
            onClick={() => onToolChange(id)}
            className={[
              "flex items-center gap-2 border-b-2 px-3 py-2.5 text-sm font-medium transition-colors duration-100",
              isActive
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
            ].join(" ")}
          >
            <Icon className="h-3.5 w-3.5" />
            <span>{label}</span>
          </button>
        );
      })}
    </div>
  );
}
