type GreekName = "delta" | "gamma" | "vega" | "theta" | "rho";

const GREEK_CONFIG: Record<GreekName, { decimals: number; suffix?: string }> = {
  delta: { decimals: 3 },
  gamma: { decimals: 3 },
  theta: { decimals: 3, suffix: "/d" },
  vega: { decimals: 2 },
  rho: { decimals: 2 },
};

export function formatPrice(value: number): string {
  if (value > 0 && value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function formatGreek(name: string, value: number): string {
  const config = GREEK_CONFIG[name as GreekName];
  if (!config) return value.toFixed(4);
  const formatted = value.toFixed(config.decimals);
  return config.suffix ? `${formatted}${config.suffix}` : formatted;
}

export function formatPnl(value: number): string {
  if (value === 0) return "$0.00";
  const abs = Math.abs(value).toFixed(2);
  return value > 0 ? `+$${abs}` : `-$${abs}`;
}
