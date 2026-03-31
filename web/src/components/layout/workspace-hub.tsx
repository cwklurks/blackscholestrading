"use client";

import { useState, lazy, Suspense, useMemo } from "react";
import { ToolBar } from "./tool-bar";
import { WorkspaceProvider } from "@/contexts/workspace-context";
import { CommandPalette } from "@/components/command-palette";

const PricingPanel = lazy(() => import("@/app/pricing/panel"));
const VolatilityPanel = lazy(() => import("@/app/volatility/panel"));
const StrategiesPanel = lazy(() => import("@/app/strategies/panel"));
const BacktestPanel = lazy(() => import("@/app/backtest/panel"));
const MarketPanel = lazy(() => import("@/app/market/panel"));

/** The Workspace panel is the simplified pricer from the home page */
const WorkspacePanel = lazy(() =>
  import("@/components/pricing/workspace").then((m) => ({
    default: m.Workspace,
  }))
);

function PanelSkeleton() {
  return (
    <div className="space-y-4 p-2">
      <div className="h-8 w-48 animate-pulse rounded bg-muted" />
      <div className="h-64 w-full animate-pulse rounded bg-muted" />
    </div>
  );
}

const PANELS: Record<string, React.LazyExoticComponent<React.ComponentType>> = {
  pricing: WorkspacePanel,
  sensitivity: PricingPanel,
  volatility: VolatilityPanel,
  strategies: StrategiesPanel,
  backtest: BacktestPanel,
  market: MarketPanel,
};

export function WorkspaceHub() {
  const [activeTool, setActiveTool] = useState("pricing");
  const ctxValue = useMemo(
    () => ({ activeTool, setActiveTool }),
    [activeTool],
  );

  const ActivePanel = PANELS[activeTool];

  return (
    <WorkspaceProvider value={ctxValue}>
      <CommandPalette />
      <div className="flex h-full flex-col">
        <ToolBar activeToolId={activeTool} onToolChange={setActiveTool} />
        <div className="flex-1 pt-6">
          <Suspense fallback={<PanelSkeleton />}>
            {ActivePanel && <ActivePanel />}
          </Suspense>
        </div>
      </div>
    </WorkspaceProvider>
  );
}
