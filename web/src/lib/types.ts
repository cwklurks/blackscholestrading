/**
 * TypeScript types mirroring the FastAPI Pydantic schemas.
 *
 * Organised by domain: pricing, market, volatility, strategy/backtest.
 */

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

export interface RangeSpec {
  min: number
  max: number
  steps?: number // default 20, 2-100
}

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

export interface PricingRequest {
  model?: string // default "Black-Scholes"
  S: number
  K: number
  T: number
  r: number
  sigma: number
  q?: number
  borrow_cost?: number
  option_type?: "call" | "put"
  model_params?: Record<string, number>
}

export interface PricingResponse {
  model: string
  price: number
  delta: number
  gamma: number
  theta: number
  vega: number
  rho: number
}

export interface HeatmapRequest {
  K: number
  T: number
  r: number
  q?: number
  borrow_cost?: number
  spot_range: RangeSpec
  vol_range: RangeSpec
}

export interface HeatmapResponse {
  spot_values: number[]
  vol_values: number[]
  call_prices: number[][]
  put_prices: number[][]
}

export interface MonteCarloRequest {
  S: number
  K: number
  T: number
  r: number
  sigma: number
  paths?: number // default 10000, 100-50000
  option_type?: "call" | "put"
  q?: number
  borrow_cost?: number
}

export interface MonteCarloResponse {
  price: number
  std_error: number
  terminal_prices: number[]
  confidence_interval: number[]
}

// ---------------------------------------------------------------------------
// Volatility Surface
// ---------------------------------------------------------------------------

export interface VolSurfaceRequest {
  ticker: string
  strikes?: number[]
  expirations?: string[]
}

export interface VolSurfacePoint {
  strike: number
  expiry: string
  iv: number | null
}

export interface VolSurfaceResponse {
  surface: VolSurfacePoint[]
  smile_data: Record<string, Array<Record<string, unknown>>>
  coverage: number
}

// ---------------------------------------------------------------------------
// Market Data
// ---------------------------------------------------------------------------

export interface OHLCVRow {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number | null
}

export interface MarketResponse {
  price: number
  history: OHLCVRow[]
  historical_vol: number
  fetched_at: string
}

export interface ChainRow {
  strike: number
  lastPrice: number
  iv: number | null
  volume: number | null
  oi: number | null
  expiration: string | null
}

export interface ChainResponse {
  calls: ChainRow[]
  puts: ChainRow[]
  expirations: string[]
}

// ---------------------------------------------------------------------------
// Strategy / Payoff
// ---------------------------------------------------------------------------

export interface StrategyLeg {
  type: "call" | "put"
  strike: number
  qty?: number // default 1
  side: "long" | "short"
  entry_price?: number | null
}

export interface SpotRange {
  min: number
  max: number
}

export interface PayoffRequest {
  legs: StrategyLeg[]
  spot_range: SpotRange
  S: number
  T?: number
  r?: number
  sigma?: number
}

export interface PayoffResponse {
  prices: number[]
  pnl: number[]
  breakevens: number[]
  max_profit: number | null
  max_loss: number | null
}

// ---------------------------------------------------------------------------
// Backtest
// ---------------------------------------------------------------------------

export interface BacktestLeg {
  type: "call" | "put"
  strike: number
  expiry: string // ISO date string
  qty?: number
  side: "long" | "short"
}

export interface BacktestRequest {
  ticker: string
  legs: BacktestLeg[]
  r?: number
  sigma?: number
}

export interface PnLPoint {
  date: string
  pnl: number
}

export interface BacktestResponse {
  pnl_series: PnLPoint[]
  total_pnl: number
  max_drawdown: number
  sharpe_ratio: number | null
  win_rate: number
}

// ---------------------------------------------------------------------------
// Health / Models
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string
  models: string[]
}

export interface ModelParam {
  type: string
  default: number
  description: string
}

export interface ModelInfo {
  name: string
  description: string
  params: Record<string, ModelParam>
}
