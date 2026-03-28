"use client";

import { useEffect, useState, useCallback } from "react";
import { Command } from "cmdk";
import {
  Calculator,
  Activity,
  Layers,
  BarChart3,
  LineChart,
  TrendingUp,
  Search,
} from "lucide-react";
import { useWorkspace } from "@/contexts/workspace-context";
import { useTickerDispatch } from "@/contexts/ticker-context";
import { api } from "@/lib/api";
import { TOOLS } from "@/components/layout/tool-bar";

const TOOL_ICONS: Record<string, typeof Calculator> = {
  pricing: Calculator,
  sensitivity: TrendingUp,
  volatility: Activity,
  strategies: Layers,
  backtest: BarChart3,
  market: LineChart,
};

const POPULAR_TICKERS = ["AAPL", "SPY", "TSLA", "NVDA", "MSFT", "AMZN", "QQQ", "META"];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const { setActiveTool } = useWorkspace();
  const dispatch = useTickerDispatch();

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  const handleToolSwitch = useCallback(
    (toolId: string) => {
      setActiveTool(toolId);
      setOpen(false);
    },
    [setActiveTool],
  );

  const handleTickerLoad = useCallback(
    async (ticker: string) => {
      setLoading(true);
      try {
        const data = await api.market(ticker);
        dispatch({
          type: "SET_MARKET_DATA",
          ticker,
          spot: data.price,
          historicalVol: data.historical_vol,
        });
        setActiveTool("pricing");
      } catch {
        // Silently fail - user can try again
      } finally {
        setLoading(false);
        setOpen(false);
      }
    },
    [dispatch, setActiveTool],
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />

      {/* Palette */}
      <div className="absolute left-1/2 top-[20%] w-full max-w-lg -translate-x-1/2">
        <Command
          className="rounded-lg border border-border bg-surface shadow-2xl"
          loop
        >
          <div className="flex items-center gap-2 border-b border-border px-4">
            <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
            <Command.Input
              placeholder={loading ? "Loading..." : "Search tools, tickers..."}
              className="h-12 w-full bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
              autoFocus
            />
          </div>

          <Command.List className="max-h-80 overflow-y-auto p-2">
            <Command.Empty className="px-4 py-6 text-center text-sm text-muted-foreground">
              No results. Try a ticker symbol like AAPL.
            </Command.Empty>

            {/* Tool switching */}
            <Command.Group
              heading={
                <span className="px-2 text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
                  Tools
                </span>
              }
            >
              {TOOLS.map((tool) => {
                const Icon = TOOL_ICONS[tool.id] ?? Calculator;
                return (
                  <Command.Item
                    key={tool.id}
                    value={`go to ${tool.label}`}
                    onSelect={() => handleToolSwitch(tool.id)}
                    className="flex cursor-pointer items-center gap-3 rounded-md px-3 py-2.5 text-sm text-foreground data-[selected=true]:bg-accent/10 data-[selected=true]:text-foreground"
                  >
                    <Icon className="h-4 w-4 text-muted-foreground" />
                    <span>Go to {tool.label}</span>
                  </Command.Item>
                );
              })}
            </Command.Group>

            {/* Quick ticker load */}
            <Command.Group
              heading={
                <span className="px-2 text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
                  Load Ticker
                </span>
              }
            >
              {POPULAR_TICKERS.map((ticker) => (
                <Command.Item
                  key={ticker}
                  value={`load ${ticker}`}
                  onSelect={() => handleTickerLoad(ticker)}
                  className="flex cursor-pointer items-center gap-3 rounded-md px-3 py-2.5 text-sm text-foreground data-[selected=true]:bg-accent/10 data-[selected=true]:text-foreground"
                >
                  <span className="font-mono text-xs text-primary">{ticker}</span>
                  <span className="text-muted-foreground">Load market data</span>
                </Command.Item>
              ))}
            </Command.Group>
          </Command.List>

          {/* Footer hint */}
          <div className="border-t border-border px-4 py-2">
            <div className="flex items-center gap-4 text-[11px] text-muted-foreground">
              <span>
                <kbd className="rounded border border-border bg-background px-1.5 py-0.5 font-mono text-[10px]">
                  ↑↓
                </kbd>{" "}
                navigate
              </span>
              <span>
                <kbd className="rounded border border-border bg-background px-1.5 py-0.5 font-mono text-[10px]">
                  ↵
                </kbd>{" "}
                select
              </span>
              <span>
                <kbd className="rounded border border-border bg-background px-1.5 py-0.5 font-mono text-[10px]">
                  esc
                </kbd>{" "}
                close
              </span>
            </div>
          </div>
        </Command>
      </div>
    </div>
  );
}
