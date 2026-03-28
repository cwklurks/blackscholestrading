"use client";

import { useState, useCallback } from "react";
import { api, ApiError } from "@/lib/api";
import type { PricingResponse, PayoffResponse } from "@/lib/types";
import { ParamRail, DEFAULT_FORM_VALUES, type ParamRailFormValues } from "./param-rail";
import { GreeksRow } from "./greeks-row";
import { PayoffChart } from "@/components/charts/payoff-chart";
import { formatPrice } from "@/lib/format";

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
            min: form.S * 0.5,
            max: form.S * 1.5,
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
    <div className="grid gap-8 lg:grid-cols-[300px_1fr]">
      {/* Left rail - parameter inputs */}
      <aside className="border-r border-border bg-surface p-4 lg:pr-6">
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
          <div className="rounded-[var(--radius)] border border-negative/30 bg-negative/10 px-4 py-3 text-sm text-negative">
            {error}
          </div>
        )}

        {/* Price display */}
        {pricingResult && (
          <div className="border-b border-border pb-4">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted-foreground">{pricingResult.model}</span>
              <div className="font-mono text-5xl font-bold tabular-nums tracking-tight">
                {formatPrice(pricingResult.price)}
              </div>
            </div>
            <GreeksRow data={pricingResult} className="mt-4" />
          </div>
        )}

        {/* Loading skeleton */}
        {isPricing && !pricingResult && (
          <div className="border-b border-border pb-4">
            <div className="h-10 w-1/3 animate-pulse rounded bg-muted" />
            <div className="mt-4 grid grid-cols-5 gap-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="space-y-1">
                  <div className="h-3 w-12 animate-pulse rounded bg-muted" />
                  <div className="h-5 w-16 animate-pulse rounded bg-muted" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Payoff chart */}
        {payoffResult && (
          <PayoffChart
            prices={payoffResult.prices}
            pnl={payoffResult.pnl}
            breakevens={payoffResult.breakevens}
            title={`${form.optionType === "call" ? "Call" : "Put"} P&L at Expiration`}
            height={380}
          />
        )}

        {/* Payoff loading skeleton */}
        {isPricing && !payoffResult && (
          <div className="h-[380px] w-full animate-pulse rounded bg-muted" />
        )}

        {/* Empty state */}
        {!pricingResult && !isPricing && !error && (
          <div className="flex min-h-[400px] items-center justify-center">
            <div className="text-center">
              <p className="font-mono text-5xl font-bold tabular-nums tracking-tight text-muted-foreground/20">
                $0.00
              </p>
              <p className="mt-4 text-sm text-muted-foreground">
                Price an AAPL call to get started
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
