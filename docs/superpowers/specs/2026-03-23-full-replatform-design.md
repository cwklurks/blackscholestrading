# Full Replatform: Black-Scholes Trader

**Date**: 2026-03-23
**Status**: Approved
**Approach**: B - Full Replatform (FastAPI + Next.js)

## Problem Statement

The Black-Scholes Trader is a portfolio project to demonstrate financial engineering depth and software engineering quality. The current state has:

- **Broken build**: `models/black_scholes.py` imports from `models.utils` which doesn't exist (should reference `utils.constants`). All 3 test files fail to collect.
- **Massive code duplication**: `analytics.py` (621 lines) duplicates everything in the `models/` package. `app.py` imports from both.
- **Monolithic UI**: 431-line `app.py` with all Streamlit logic in one file.
- **Missing features**: No volatility surface, no multi-leg strategies, no risk metrics.
- **Misleading README**: Claims "GPU-accelerated" but there's no GPU code.
- **Stale refactoring**: Uncommitted changes from an incomplete refactoring (412 lines added, 306 removed).
- **Orphaned data_service.py**: Root-level `data_service.py` wraps the `data/` providers with caching but sits outside any package structure.

## Target Audience

SWE-leaning roles (fintech, general software engineering) with doors open to quant-adjacent positions. Prioritize: clean architecture, test coverage, full-stack ability, modern tooling. Preserve: mathematical rigor, model correctness, performance story (Numba).

## Architecture

### Monorepo Structure

```
blackscholestrader/
├── api/                          # Python FastAPI backend
│   ├── app/
│   │   ├── main.py               # FastAPI entrypoint + CORS
│   │   ├── routers/
│   │   │   ├── pricing.py        # /api/price, /api/greeks, /api/heatmap
│   │   │   ├── market.py         # /api/market/{ticker}, /api/chain/{ticker}
│   │   │   └── backtest.py       # /api/backtest, /api/strategy/payoff
│   │   ├── schemas/              # Pydantic request/response models
│   │   │   ├── pricing.py
│   │   │   ├── market.py
│   │   │   └── backtest.py
│   │   └── services/             # Thin wrappers around models/
│   │       ├── pricing_service.py
│   │       ├── market_service.py
│   │       └── backtest_service.py
│   ├── models/                   # Existing pricing models (cleaned up)
│   │   ├── __init__.py
│   │   ├── black_scholes.py      # Analytical BS model + vectorized
│   │   ├── numerical.py          # Binomial American option
│   │   └── simulation.py         # Heston, GARCH, Bates MC engines
│   ├── utils/                    # Shared utilities (moved from root utils/)
│   │   ├── constants.py          # EPS_TIME, EPS_VOL, clamp_inputs
│   │   └── numba_compat.py       # Numba JIT fallback layer
│   ├── data/                     # Data providers (Yahoo, Mock)
│   ├── strategies/               # Backtest framework + Strategy ABC
│   ├── tests/
│   │   ├── test_models.py        # Existing: parity, convergence, validation
│   │   ├── test_api.py           # NEW: endpoint tests
│   │   └── test_integration.py   # NEW: end-to-end pricing flow
│   ├── requirements.txt
│   └── pyproject.toml
│
├── web/                          # Next.js frontend
│   ├── src/
│   │   ├── app/                  # App Router pages
│   │   │   ├── layout.tsx        # Root layout with sidebar nav
│   │   │   ├── page.tsx          # Dashboard (hero pricing + quick Greeks)
│   │   │   ├── pricing/page.tsx  # Full pricing with model selector
│   │   │   ├── volatility/page.tsx  # Vol surface + smile + term structure
│   │   │   ├── strategies/page.tsx  # Multi-leg strategy builder
│   │   │   ├── backtest/page.tsx    # Backtesting with risk metrics
│   │   │   └── market/page.tsx      # Candlesticks + chain + HV
│   │   ├── components/
│   │   │   ├── ui/               # shadcn/ui primitives
│   │   │   ├── charts/           # Plotly wrappers (heatmap, surface, candlestick)
│   │   │   ├── pricing/          # PricingCard, GreeksTable, ModelSelector
│   │   │   └── layout/           # AppShell, Sidebar, Nav
│   │   ├── lib/
│   │   │   ├── api.ts            # Typed API client (fetch wrappers)
│   │   │   └── types.ts          # TypeScript types matching Pydantic schemas
│   │   └── hooks/
│   │       ├── use-pricing.ts    # SWR hook for pricing data
│   │       └── use-market.ts     # SWR hook for market data
│   ├── package.json
│   └── tsconfig.json
│
├── docker-compose.yml            # Local dev: api + web
├── README.md
└── .github/workflows/ci.yml     # Parallel: api-tests + web-tests + integration
```

### Data Flow

```
User Browser
    |
    v
Next.js (SSR shell + Client Components)
    |  POST /api/price { model, S, K, T, r, sigma, ... }
    v
FastAPI Backend (CORS enabled for localhost:3000)
    |  pricing_service.price_option(request)
    v
Models Layer (BlackScholesModel, binomial, Heston MC, GARCH MC, Bates MC)
    |  Numba JIT acceleration on Monte Carlo cores
    v
JSON Response -> SWR cache -> React renders charts/tables
```

### Python Package Root

`api/` is the Python package root (`PYTHONPATH=api/`). All imports within the API use package-relative paths:
- `from models.black_scholes import BlackScholesModel`
- `from utils.constants import clamp_inputs`
- `from data.yahoo_provider import YahooProvider`

The `pyproject.toml` in `api/` will set `[tool.pytest.ini_options] pythonpath = ["."]` and `[tool.mypy] mypy_path = "."` to make this explicit.

After the move, update all `from models.utils import ...` to `from utils.constants import ...` in `models/black_scholes.py`, `models/simulation.py`, and `strategies/backtest.py`. This consolidates the constants into one canonical location (`utils/constants.py`) and eliminates the duplicate that caused the current broken build.

### Field Name Convention

The API uses short quant-standard field names (`q` for dividend yield, `sigma` for volatility). The Pydantic schemas translate to the internal Python model names:

```python
class PricingRequest(BaseModel):
    q: float = Field(0.0, description="Dividend yield")

    def to_model_kwargs(self) -> dict:
        return {"dividend_yield": self.q, ...}
```

This translation happens in the schema layer, not in the service or model layer. The internal `BlackScholesModel` constructor signature (`dividend_yield=`) remains unchanged.

## API Contract

### Pricing Endpoints

**POST /api/price**
- Request: `{ model: string, S: float, K: float, T: float, r: float, sigma: float, q: float, borrow_cost: float, option_type: "call"|"put", model_params: {...} }`
- Response: `{ model: string, price: float, delta: float, gamma: float, theta: float, vega: float, rho: float }`
- Computes price + all Greeks for a single option using the selected model. Response echoes back the model name for frontend display.

**POST /api/heatmap**
- Request: `{ K: float, T: float, r: float, q: float, borrow_cost: float, spot_range: {min, max, steps}, vol_range: {min, max, steps} }`
- Response: `{ spot_values: float[], vol_values: float[], call_prices: float[][], put_prices: float[][] }`
- Vectorized BS pricing across a Spot x Vol grid. Uses `bs_call_price_vectorized`.

**POST /api/monte-carlo**
- Request: `{ S, K, T, r, sigma, paths: int, option_type, q, borrow_cost }`
- Response: `{ price: float, std_error: float, terminal_prices: float[], confidence_interval: [float, float] }`
- Returns terminal price distribution for histogram visualization.

### Volatility Surface (NEW)

**POST /api/volatility-surface**
- Request: `{ ticker: string, strikes: float[], expirations: string[] }`
- Response: `{ surface: [{ strike, expiry, iv }...], smile_data: { [expiry]: [{ strike, iv }...] } }`
- Computes implied volatilities across a strike/expiry grid using the existing `implied_volatility()` function and live options chain data.
- **NaN handling**: yfinance often returns NaN IVs for illiquid options. The service will: (1) attempt to compute IV from market price using `implied_volatility()`, (2) if the market price is also NaN/zero, mark that cell as `null` in the response, (3) the frontend interpolates missing cells or shows gaps in the surface. The response includes a `coverage` field (0.0-1.0) indicating what fraction of the grid has valid data.

### Market Data Endpoints

**GET /api/market/{ticker}**
- Response: `{ price: float, history: [{ date, open, high, low, close, volume }...], historical_vol: float, fetched_at: string }`
- Uses existing Yahoo Finance provider + caching layer.

**GET /api/chain/{ticker}?expiry=YYYY-MM-DD**
- Response: `{ calls: [{ strike, lastPrice, iv, volume, oi }...], puts: [...], expirations: string[] }`

### Strategy & Backtest Endpoints

**POST /api/strategy/payoff**
- Request: `{ legs: [{ type: "call"|"put", strike, qty, side: "long"|"short", entry_price? }...], spot_range: {min, max} }`
- Response: `{ prices: float[], pnl: float[], breakevens: float[], max_profit: float|null, max_loss: float|null }`
- Multi-leg payoff diagram calculation. NEW feature - replaces the single-position portfolio tab.

**POST /api/backtest**
- Request: `{ ticker_or_prices, legs: [{ type, strike, expiry, qty, side }...], r, sigma }`
- Response: `{ pnl_series: [{ date, pnl }...], total_pnl, max_drawdown, sharpe_ratio, win_rate }`
- Enhanced backtest with risk metrics (Sharpe, drawdown, win rate). NEW metrics.
- **Multi-leg architecture**: The existing `Strategy` ABC and `SingleOptionStrategy` class in `strategies/backtest.py` handle single-leg mark-to-market repricing. For multi-leg, we add a new `MultiLegStrategy(Strategy)` that: (1) maintains a list of `SingleOptionStrategy` legs, (2) handles per-leg expiration (a leg's P&L freezes at intrinsic value when T=0), (3) aggregates P&L across all legs. This extends the existing ABC without breaking backward compatibility.

### Meta Endpoints

**GET /api/health** -> `{ status: "ok", models: string[] }`
**GET /api/models** -> `[{ name, description, params: { [key]: { type, default, description } } }]`

Example response for `GET /api/models` (Heston entry):
```json
{
  "name": "Heston MC",
  "description": "Heston stochastic volatility model via Monte Carlo simulation",
  "params": {
    "heston_kappa": { "type": "float", "default": 1.5, "description": "Mean reversion speed" },
    "heston_theta": { "type": "float", "default": 0.04, "description": "Long-run variance" },
    "heston_rho":   { "type": "float", "default": -0.7, "description": "Price-vol correlation" },
    "heston_vol_of_vol": { "type": "float", "default": 0.3, "description": "Volatility of volatility" },
    "mc_paths":     { "type": "int", "default": 4000, "description": "Number of simulation paths" },
    "mc_steps":     { "type": "int", "default": 120, "description": "Number of time steps" }
  }
}
```

## Frontend Design

### Tech Stack
- Next.js 15 with App Router
- shadcn/ui for all UI primitives
- Plotly.js for financial charts
- Tailwind CSS with dark mode (zinc/neutral tokens)
- Geist Sans for UI text, Geist Mono for prices/Greeks/numbers
- SWR for data fetching with stale-while-revalidate caching

### Pages

1. **Dashboard** (`/`): Hero pricing cards (Call/Put with price + top Greeks), quick model selector, key charts at a glance
2. **Pricing** (`/pricing`): Full model configuration, all 5 Greeks displayed, sensitivity charts (Delta/Gamma/Vega vs Spot), heatmap, Monte Carlo terminal price distribution histogram (when MC model is selected)
3. **Volatility** (`/volatility`): 3D vol surface (Plotly surface plot), 2D smile curves per expiry, ATM term structure. Requires live data.
4. **Strategies** (`/strategies`): Multi-leg builder with add/remove legs, real-time payoff diagram, breakeven annotations, pre-built templates (straddle, strangle, iron condor, butterfly, collar)
5. **Backtest** (`/backtest`): Strategy selection, time period, cumulative P&L chart, risk metrics (Sharpe, drawdown, win rate), comparison vs buy-and-hold
6. **Market** (`/market`): Candlestick chart, options chain table, historical volatility chart, IV/HV comparison

### Design Principles
- Dark mode by default (financial dashboard aesthetic)
- Numbers in monospace font (Geist Mono)
- Green = calls/profit, Red = puts/loss (consistent color coding)
- Dense but readable information layout
- Skeleton loaders for every async operation
- Responsive: desktop primary, tablet secondary

## Backend Cleanup

### Bug Fixes (Immediate)
1. Fix broken imports: update `models/black_scholes.py` and `models/simulation.py` to import from `utils.constants` instead of the nonexistent `models.utils`. The constants already exist in `utils/constants.py` - no new file needed, just fix the import paths.
2. Delete `analytics.py` - fully duplicated by `models/` package
3. Delete `app.py` and `ui_components.py` - Streamlit code replaced by Next.js
4. Fix README to remove "GPU-accelerated" claim

### File Migration

| Source | Destination | Action |
|--------|------------|--------|
| `models/` | `api/models/` | Move, fix imports to use `utils.constants` |
| `strategies/` | `api/strategies/` | Move as-is |
| `data/` | `api/data/` | Move as-is |
| `data_service.py` | `api/app/services/market_service.py` | Move + rename |
| `tests/` | `api/tests/` | Move + add API tests |
| `analytics.py` | (deleted) | Duplicated by models/ |
| `app.py` | (deleted) | Replaced by Next.js |
| `ui_components.py` | (deleted) | Replaced by React |
| `utils/` | `api/utils/` | Move as-is |

### Testing Strategy
- Existing tests: put-call parity, convergence, reproducibility, input validation
- New API tests: endpoint schema validation, error handling
- New integration tests: full pricing flow through FastAPI test client
- Frontend tests: component rendering with Vitest + React Testing Library
- Target: 80%+ coverage on both api/ and web/

## CI/CD & Deployment

### GitHub Actions Pipeline
```
api-tests (parallel):     ruff check + mypy + pytest --cov
web-tests (parallel):     eslint + tsc + vitest + next build
integration (sequential): docker-compose up -d, wait for health checks, run cross-service tests, docker-compose down
```

The integration job uses Docker Compose to start both services. The API health endpoint (`GET /api/health`) is polled until ready (max 30s). Integration tests use `httpx` to hit the real API from the web test runner. This avoids fragile `uvicorn &` background process management in CI.

### Deployment
- Frontend: Vercel (automatic deploys from main)
- API: Railway or Render (free tier, Python support). **Known limitation**: free tiers spin down after inactivity; cold starts take 10-30s. Mitigation: the frontend shows a "Connecting to pricing engine..." skeleton state on first load. For live demos, keep a browser tab open or use a paid tier ($5/mo).
- Local: `docker-compose up` starts both services

### Environment
- `NEXT_PUBLIC_API_URL`: Frontend -> API URL
- `CORS_ORIGINS`: API allowed origins
- No secrets required for core pricing (pure math)
- Yahoo Finance works without API key on free tier

## What's NOT In Scope
- WebSocket/real-time streaming prices (polling via SWR is sufficient)
- User authentication/accounts
- Persistent portfolio storage (in-memory/session only)
- Mobile-optimized layout (desktop + tablet only)
- GPU/CUDA acceleration (Numba CPU JIT is the performance story)
- 3D Greeks surfaces (Delta/Gamma/Vega as 3D surface plots over Spot x Vol) - could be a follow-up. Note: 2D sensitivity curves (Greeks vs Spot) ARE in scope on the Pricing page.
