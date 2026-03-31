/**
 * Typed fetch wrapper for the Black-Scholes pricing API.
 *
 * All methods throw `ApiError` on non-2xx responses so callers get
 * both the HTTP status and a parsed error message.
 */
import type {
  PricingRequest,
  PricingResponse,
  HeatmapRequest,
  HeatmapResponse,
  MonteCarloRequest,
  MonteCarloResponse,
  VolSurfaceRequest,
  VolSurfaceResponse,
  MarketResponse,
  ChainResponse,
  PayoffRequest,
  PayoffResponse,
  BacktestRequest,
  BacktestResponse,
  HealthResponse,
  ModelInfo,
} from "@/lib/types"

// ---------------------------------------------------------------------------
// Base URL
// ---------------------------------------------------------------------------

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? ""

// ---------------------------------------------------------------------------
// Error class
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
    this.name = "ApiError"
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${BASE_URL}${path}`
  const res = await fetch(url, init)

  if (!res.ok) {
    let message = `API error ${res.status}`
    try {
      const body = await res.json()
      if (body.detail) {
        message =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail)
      }
    } catch {
      // body wasn't JSON - keep the generic message
    }
    throw new ApiError(message, res.status)
  }

  return (await res.json()) as T
}

function post<TReq, TRes>(path: string) {
  return (body: TReq): Promise<TRes> =>
    request<TRes>(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
}

function get<T>(path: string): Promise<T> {
  return request<T>(path)
}

// ---------------------------------------------------------------------------
// Public API object
// ---------------------------------------------------------------------------

export const api = {
  /** POST /api/price - Price a single option. */
  price: post<PricingRequest, PricingResponse>("/api/price"),

  /** POST /api/heatmap - Generate a spot x vol heatmap. */
  heatmap: post<HeatmapRequest, HeatmapResponse>("/api/heatmap"),

  /** POST /api/monte-carlo - Run Monte Carlo simulation. */
  monteCarlo: post<MonteCarloRequest, MonteCarloResponse>("/api/monte-carlo"),

  /** POST /api/volatility-surface - Compute volatility surface. */
  volSurface: post<VolSurfaceRequest, VolSurfaceResponse>(
    "/api/volatility-surface",
  ),

  /** POST /api/strategy/payoff - Compute multi-leg payoff diagram. */
  payoff: post<PayoffRequest, PayoffResponse>("/api/strategy/payoff"),

  /** POST /api/backtest - Run a historical backtest. */
  backtest: post<BacktestRequest, BacktestResponse>("/api/backtest"),

  /** GET /api/market/:ticker - Fetch market data for a ticker. */
  market: (ticker: string): Promise<MarketResponse> =>
    get<MarketResponse>(`/api/market/${encodeURIComponent(ticker)}`),

  /** GET /api/chain/:ticker - Fetch options chain for a ticker. */
  chain: (ticker: string): Promise<ChainResponse> =>
    get<ChainResponse>(`/api/chain/${encodeURIComponent(ticker)}`),

  /** GET /api/health - Health check. */
  health: (): Promise<HealthResponse> => get<HealthResponse>("/api/health"),

  /** GET /api/models - List available pricing models. */
  models: (): Promise<ModelInfo[]> => get<ModelInfo[]>("/api/models"),
} as const
