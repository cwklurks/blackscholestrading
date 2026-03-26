"use client";

import { useCallback } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ModelParamPanel,
  getDefaultModelParams,
} from "@/components/pricing/model-param-panel";
import {
  AdvancedParams,
  type AdvancedParamsValues,
} from "@/components/pricing/advanced-params";

const MODELS = [
  "Black-Scholes",
  "Binomial",
  "Heston MC",
  "GARCH MC",
  "Bates Jump-Diffusion",
] as const;

export interface ParamRailFormValues {
  ticker: string;
  model: string;
  S: number;
  K: number;
  T: number;
  r: number;
  sigma: number;
  optionType: "call" | "put";
  modelParams: Record<string, number>;
  advanced: AdvancedParamsValues;
}

export const DEFAULT_FORM_VALUES: ParamRailFormValues = {
  ticker: "AAPL",
  model: "Black-Scholes",
  S: 100,
  K: 100,
  T: 0.5,
  r: 0.05,
  sigma: 0.2,
  optionType: "call",
  modelParams: {},
  advanced: { q: 0, borrowCost: 0 },
};

interface ParamRailProps {
  values: ParamRailFormValues;
  onChange: (values: ParamRailFormValues) => void;
  onPrice: () => void;
  isPricing: boolean;
}

export function ParamRail({
  values,
  onChange,
  onPrice,
  isPricing,
}: ParamRailProps) {
  const updateField = useCallback(
    <K extends keyof ParamRailFormValues>(
      field: K,
      value: ParamRailFormValues[K],
    ) => {
      onChange({ ...values, [field]: value });
    },
    [values, onChange],
  );

  const handleNumberChange = useCallback(
    (field: keyof Pick<ParamRailFormValues, "S" | "K" | "T" | "r" | "sigma">, raw: string) => {
      const parsed = parseFloat(raw);
      if (!Number.isNaN(parsed)) {
        updateField(field, parsed);
      }
    },
    [updateField],
  );

  const handleModelChange = useCallback(
    (model: string) => {
      onChange({
        ...values,
        model,
        modelParams: getDefaultModelParams(model),
      });
    },
    [values, onChange],
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Ticker */}
      <div className="space-y-1.5">
        <Label htmlFor="ticker">Ticker</Label>
        <Input
          id="ticker"
          value={values.ticker}
          onChange={(e) => updateField("ticker", e.target.value.toUpperCase())}
          placeholder="AAPL"
        />
      </div>

      {/* Model selector */}
      <div className="space-y-1.5">
        <Label>Model</Label>
        <Select
          value={values.model}
          onValueChange={(val) => handleModelChange(val as string)}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODELS.map((m) => (
              <SelectItem key={m} value={m}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Core parameters */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="spot">Spot (S)</Label>
          <Input
            id="spot"
            type="number"
            step="1"
            min="0"
            value={values.S}
            onChange={(e) => handleNumberChange("S", e.target.value)}
          />
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="strike">Strike (K)</Label>
          <Input
            id="strike"
            type="number"
            step="1"
            min="0"
            value={values.K}
            onChange={(e) => handleNumberChange("K", e.target.value)}
          />
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="expiry">Time (T)</Label>
          <Input
            id="expiry"
            type="number"
            step="0.05"
            min="0.01"
            value={values.T}
            onChange={(e) => handleNumberChange("T", e.target.value)}
          />
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="rate">Rate (r)</Label>
          <Input
            id="rate"
            type="number"
            step="0.005"
            min="0"
            value={values.r}
            onChange={(e) => handleNumberChange("r", e.target.value)}
          />
        </div>

        <div className="col-span-2 space-y-1.5">
          <Label htmlFor="sigma">Volatility (sigma)</Label>
          <Input
            id="sigma"
            type="number"
            step="0.01"
            min="0.01"
            value={values.sigma}
            onChange={(e) => handleNumberChange("sigma", e.target.value)}
          />
        </div>
      </div>

      {/* Call / Put toggle */}
      <div className="space-y-1.5">
        <Label>Option Type</Label>
        <div className="grid grid-cols-2 gap-2">
          <Button
            type="button"
            variant={values.optionType === "call" ? "default" : "outline"}
            onClick={() => updateField("optionType", "call")}
          >
            Call
          </Button>
          <Button
            type="button"
            variant={values.optionType === "put" ? "default" : "outline"}
            onClick={() => updateField("optionType", "put")}
          >
            Put
          </Button>
        </div>
      </div>

      {/* Model-specific params */}
      <ModelParamPanel
        model={values.model}
        values={values.modelParams}
        onChange={(params) => updateField("modelParams", params)}
      />

      {/* Advanced params */}
      <AdvancedParams
        values={values.advanced}
        onChange={(adv) => updateField("advanced", adv)}
      />

      {/* Price button */}
      <Button
        type="button"
        size="lg"
        onClick={onPrice}
        disabled={isPricing}
        className="w-full"
      >
        {isPricing ? "Pricing..." : "Price"}
      </Button>
    </div>
  );
}
