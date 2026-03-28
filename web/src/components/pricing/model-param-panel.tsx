"use client";

import { ParamInput } from "@/components/ui/param-input";

/** Model-specific parameter definitions. */
interface ParamDef {
  key: string;
  label: string;
  defaultValue: number;
  step: number;
  min?: number;
  max?: number;
}

const MODEL_PARAMS: Record<string, ParamDef[]> = {
  "Heston MC": [
    { key: "heston_kappa", label: "Kappa (mean reversion)", defaultValue: 2.0, step: 0.1, min: 0 },
    { key: "heston_theta", label: "Theta (long-run var)", defaultValue: 0.04, step: 0.01, min: 0 },
    { key: "heston_rho", label: "Rho (correlation)", defaultValue: -0.7, step: 0.05, min: -1, max: 1 },
    { key: "heston_vol_of_vol", label: "Vol of Vol", defaultValue: 0.3, step: 0.05, min: 0 },
    { key: "mc_paths", label: "MC Paths", defaultValue: 10000, step: 1000, min: 100, max: 50000 },
    { key: "mc_steps", label: "MC Steps", defaultValue: 252, step: 10, min: 10, max: 1000 },
  ],
  "GARCH MC": [
    { key: "garch_alpha0", label: "Alpha0 (omega)", defaultValue: 0.00001, step: 0.000005, min: 0 },
    { key: "garch_alpha1", label: "Alpha1 (ARCH)", defaultValue: 0.1, step: 0.01, min: 0, max: 1 },
    { key: "garch_beta1", label: "Beta1 (GARCH)", defaultValue: 0.85, step: 0.01, min: 0, max: 1 },
    { key: "mc_paths", label: "MC Paths", defaultValue: 10000, step: 1000, min: 100, max: 50000 },
  ],
  "Bates Jump-Diffusion": [
    { key: "jump_lambda", label: "Jump Intensity", defaultValue: 0.1, step: 0.01, min: 0 },
    { key: "jump_mu", label: "Jump Mean", defaultValue: -0.05, step: 0.01 },
    { key: "jump_delta", label: "Jump Vol", defaultValue: 0.1, step: 0.01, min: 0 },
    { key: "mc_paths", label: "MC Paths", defaultValue: 10000, step: 1000, min: 100, max: 50000 },
    { key: "mc_steps", label: "MC Steps", defaultValue: 252, step: 10, min: 10, max: 1000 },
  ],
  "Binomial (American)": [
    { key: "binomial_steps", label: "Binomial Steps", defaultValue: 100, step: 10, min: 10, max: 1000 },
  ],
};

interface ModelParamPanelProps {
  model: string;
  values: Record<string, number>;
  onChange: (values: Record<string, number>) => void;
}

/**
 * Renders model-specific parameter inputs based on the selected model.
 * Returns null for Black-Scholes or any model without extra params.
 */
export function ModelParamPanel({ model, values, onChange }: ModelParamPanelProps) {
  const paramDefs = MODEL_PARAMS[model];
  if (!paramDefs || paramDefs.length === 0) return null;

  return (
    <div className="rounded-lg border border-border bg-card">
      <div className="border-b border-border px-3 py-2.5">
        <span className="text-sm font-medium text-foreground">
          {model} Parameters
        </span>
      </div>

      <div className="space-y-3 px-3 pb-3 pt-3">
        {paramDefs.map((def) => (
          <ParamInput
            key={def.key}
            label={def.label}
            step={def.step}
            min={def.min}
            max={def.max}
            value={values[def.key] ?? def.defaultValue}
            onChange={(v) => onChange({ ...values, [def.key]: v })}
          />
        ))}
      </div>
    </div>
  );
}

/** Returns the default model_params for a given model name. */
export function getDefaultModelParams(model: string): Record<string, number> {
  const paramDefs = MODEL_PARAMS[model];
  if (!paramDefs) return {};
  const defaults: Record<string, number> = {};
  for (const def of paramDefs) {
    defaults[def.key] = def.defaultValue;
  }
  return defaults;
}

/** Model names that use Monte Carlo and should not auto-reprice. */
export const MC_MODELS = new Set(["Heston MC", "GARCH MC", "Bates Jump-Diffusion"]);
