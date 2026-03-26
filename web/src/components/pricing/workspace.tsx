"use client";

import { useState, useCallback } from "react";
import { api, ApiError } from "@/lib/api";
import type { PricingResponse, PayoffResponse } from "@/lib/types";
import { ParamRail, DEFAULT_FORM_VALUES, type ParamRailFormValues } from "./param-rail";
import { GreeksRow } from "./greeks-row";
import { PayoffChart } from "@/components/charts/payoff-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function Workspace() {
  const [form, setForm] = useState<ParamRailFormValues>(DEFAULT_FORM_VALUES);
  const [pricingResult, setPricingResult] = useState<PricingResponse | null>(null);
  const [payoffResult, setPayoffResult] = useState<PayoffResponse | null>(null);
  const [isPricing, setIsPricing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePrice = useCallback(async () => {
    setIsPricing(true);
    setError(null);

    try {
      const [priceRes, payoffRes] = await Promise.all([
        api.price({
          model: form.model,
          S: form.S,
          K: form.K,
          T: form.T,
          r: form.r,
          sigma: form.sigma,
          option_type: form.optionType,
          q: form.advanced.q || undefined,
          borrow_cost: form.advanced.borrowCost || undefined,
          model_params:
            Object.keys(form.modelParams).length > 0
              ? form.modelParams
              : undefined,
        }),
        api.payoff({
          legs: [
            {
              type: form.optionType,
              strike: form.K,
              qty: 1,
              side: "long",
            },
          ],
          spot_range: {
            min: form.S * 0.7,
            max: form.S * 1.3,
          },
          S: form.S,
          T: form.T,
          r: form.r,
          sigma: form.sigma,
        }),
      ]);

      setPricingResult(priceRes);
      setPayoffResult(payoffRes);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Failed to price option. Check that the API is running.";
      setError(message);
      setPricingResult(null);
      setPayoffResult(null);
    } finally {
      setIsPricing(false);
    }
  }, [form]);

  return (
    <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
      {/* Left rail - parameter inputs */}
      <aside className="rounded-lg border border-border bg-card p-4">
        <ParamRail
          values={form}
          onChange={setForm}
          onPrice={handlePrice}
          isPricing={isPricing}
        />
      </aside>

      {/* Right canvas - results */}
      <div className="flex flex-col gap-4">
        {/* Error display */}
        {error && (
          <div className="rounded-lg border border-negative/30 bg-negative/10 px-4 py-3 text-sm text-negative">
            {error}
          </div>
        )}

        {/* Price display */}
        {pricingResult && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-baseline justify-between">
                <span>{pricingResult.model}</span>
                <span className="font-mono text-2xl font-bold tabular-nums text-foreground">
                  ${pricingResult.price.toFixed(4)}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <GreeksRow data={pricingResult} />
            </CardContent>
          </Card>
        )}

        {/* Loading skeleton */}
        {isPricing && !pricingResult && (
          <Card>
            <CardContent className="space-y-3 py-6">
              <div className="h-6 w-1/3 animate-pulse rounded bg-muted" />
              <div className="grid grid-cols-5 gap-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="flex flex-col items-center gap-1">
                    <div className="h-3 w-12 animate-pulse rounded bg-muted" />
                    <div className="h-4 w-16 animate-pulse rounded bg-muted" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Payoff chart */}
        {payoffResult && (
          <Card>
            <CardContent>
              <PayoffChart
                prices={payoffResult.prices}
                pnl={payoffResult.pnl}
                breakevens={payoffResult.breakevens}
                title={`${form.optionType === "call" ? "Call" : "Put"} P&L at Expiration`}
                height={380}
              />
            </CardContent>
          </Card>
        )}

        {/* Payoff loading skeleton */}
        {isPricing && !payoffResult && (
          <Card>
            <CardContent className="py-6">
              <div className="h-[380px] w-full animate-pulse rounded bg-muted" />
            </CardContent>
          </Card>
        )}

        {/* Empty state */}
        {!pricingResult && !isPricing && !error && (
          <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-border py-20 text-center text-muted-foreground">
            <div>
              <p className="text-sm font-medium">No pricing results yet</p>
              <p className="mt-1 text-xs">
                Configure parameters and click &quot;Price&quot; to get started.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
