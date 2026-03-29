"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { api, ApiError } from "@/lib/api";
import type { PricingResponse, PayoffResponse } from "@/lib/types";
import { ParamRail, DEFAULT_FORM_VALUES, type ParamRailFormValues } from "./param-rail";
import { MC_MODELS } from "@/components/pricing/model-param-panel";
import { GreeksRow } from "./greeks-row";
import { PayoffChart } from "@/components/charts/payoff-chart";
import { formatPrice } from "@/lib/format";

const AUTO_COMPUTE_DEBOUNCE_MS = 300;

export function Workspace() {
  const [form, setForm] = useState<ParamRailFormValues>(DEFAULT_FORM_VALUES);
  const [pricingResult, setPricingResult] = useState<PricingResponse | null>(null);
  const [payoffResult, setPayoffResult] = useState<PayoffResponse | null>(null);
  const [isPricing, setIsPricing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const isAutoCompute = !MC_MODELS.has(form.model);

  const computePrice = useCallback(
    async (signal: AbortSignal) => {
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

        // Discard stale results if a newer request has been fired
        if (signal.aborted) return;

        setPricingResult(priceRes);
        setPayoffResult(payoffRes);
      } catch (err) {
        if (signal.aborted) return;

        const message =
          err instanceof ApiError
            ? err.message
            : "Failed to price option. Check that the API is running.";
        setError(message);
        setPricingResult(null);
        setPayoffResult(null);
      } finally {
        if (!signal.aborted) {
          setIsPricing(false);
        }
      }
    },
    [form],
  );

  const handlePrice = useCallback(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    computePrice(controller.signal);
  }, [computePrice]);

  // Auto-compute for analytical models (BS, Binomial) with debounce
  useEffect(() => {
    if (MC_MODELS.has(form.model)) return;

    // Skip if core params are invalid
    if (form.S <= 0 || form.K <= 0 || form.T <= 0 || form.sigma <= 0) return;

    const timer = setTimeout(() => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      computePrice(controller.signal);
    }, AUTO_COMPUTE_DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [
    form.model,
    form.S,
    form.K,
    form.T,
    form.r,
    form.sigma,
    form.optionType,
    form.modelParams,
    form.advanced.q,
    form.advanced.borrowCost,
    computePrice,
  ]);

  return (
    <div className="grid gap-8 lg:grid-cols-[300px_1fr]">
      {/* Left rail - parameter inputs */}
      <aside className="border-r border-border bg-surface p-4 lg:pr-6">
        <ParamRail
          values={form}
          onChange={setForm}
          onPrice={handlePrice}
          isPricing={isPricing}
          isAutoCompute={isAutoCompute}
        />
      </aside>

      {/* Right canvas - results */}
      <div className="flex flex-col gap-4">
        {/* Auto-compute indicator */}
        {isAutoCompute && isPricing && (
          <p className="text-xs text-muted-foreground">computing...</p>
        )}

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
