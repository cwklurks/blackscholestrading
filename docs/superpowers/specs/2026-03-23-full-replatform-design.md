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
- shadcn/ui for UI primitives (not cards — workspace panes, tables, inputs, selects)
- Plotly.js for financial charts
- Tailwind CSS with dark mode (custom CSS variable system, NOT raw zinc classes)
- Geist Sans for UI text, Geist Mono for prices/Greeks/numbers
- SWR for data fetching with stale-while-revalidate caching

### Visual Thesis

This is a professional pricing workstation, not a SaaS dashboard. The visual anchor is the **live payoff/sensitivity curve** that dominates the workspace canvas and updates as parameters change. Data IS the brand. No decorative cards, no card grids, no hero sections.

### Design Tokens

**Colors (CSS variables in `:root`)**:
- `--bg`: `#09090b` (zinc-950) — page background
- `--surface`: `#18181b` (zinc-900) — pane/panel background
- `--border`: `#27272a` (zinc-800) — borders and dividers
- `--text-primary`: `#fafafa` (zinc-50) — primary text
- `--text-secondary`: `#a1a1aa` (zinc-400) — labels, captions
- `--text-muted`: `#52525b` (zinc-600) — disabled, placeholder
- `--positive`: `#34d399` (emerald-400) — calls, profit, positive Greeks
- `--negative`: `#f87171` (red-400) — puts, loss, negative values
- `--accent`: `#3b82f6` (blue-500) — interactive elements, selected states
- `--heatmap-scale`: red (#ef4444) → transparent black → green (#22c55e), centered on ATM
- `--vol-surface-scale`: Plotly `Plasma` colorscale

**Typography**:
- Price display: `text-3xl font-semibold font-mono tabular-nums` (Geist Mono 30px/600)
- Greek values: `text-sm font-mono tabular-nums` (Geist Mono 14px/400)
- Greek labels: `text-xs text-[--text-secondary] uppercase tracking-wider` (Geist Sans 12px)
- Section headings: `text-lg font-medium` (Geist Sans 18px/500)
- Body / labels: `text-sm` (Geist Sans 14px/400)
- Nav items: `text-sm font-medium` (Geist Sans 14px/500)

**Layout**:
- Max content width: 1440px
- Sidebar: 240px (desktop), 48px icon rail (<1024px)
- Pane padding: `p-5` (20px)
- Section gap: `gap-4` (16px)
- Border radius: `rounded-lg` (8px) — not rounded-xl, not rounded-2xl
- Chart heights: primary 360px, secondary 240px, full-page 480px

**Charts (Plotly config)**:
- Template: `plotly_dark`
- Paper/plot bgcolor: `rgba(0,0,0,0)` (transparent, uses CSS --surface)
- Axis text: `--text-secondary` (#a1a1aa)
- Font: `'Geist Sans', sans-serif`, 12px
- Margins: `{ l: 48, r: 16, t: 32, b: 40 }`
- Sensitivity charts: tabbed single chart (Delta | Gamma | Vega selector), not three separate charts
- MC histogram: 50 bins, density curve overlay, vertical lines for strike + expected value
- Candlestick: includes volume bars below, default range 6M

### Pages

**Nav order**: Dashboard > Pricing > Volatility > Strategies > Backtest > Market

All pages **pre-populate with defaults**: AAPL, ATM strike, 30 DTE, historical vol, risk-free rate 5%. The app shows results on first load without any user input.

1. **Dashboard** (`/`): Pricing workspace layout — left rail has ticker input, model selector, base params (S, K, T, r, sigma). Right canvas shows: (top) price + Call/Put toggle with all 5 Greeks in a row, (middle) live payoff curve as the visual anchor, (bottom) vol surface preview (240px, AAPL default, click to navigate to /volatility). No cards. One job: "price an option fast."

2. **Pricing** (`/pricing`): Full sensitivity analysis workspace. Left rail: all params including collapsible "Advanced Parameters" section for model-specific params (Heston: kappa, theta, rho, vol_of_vol; GARCH: alpha0, alpha1, beta1). "Price Option" button triggers computation (no auto-reprice for MC models). Right canvas: tabbed sensitivity chart (Delta | Gamma | Vega | Theta vs Spot), heatmap below, MC histogram (when MC model selected). One job: "inspect sensitivities."

3. **Volatility** (`/volatility`): Upper 60% = 3D vol surface (480px, Plasma colorscale, rotation enabled, scroll-zoom disabled). Lower 40% = two columns: smile curves (left, all expiries as colored lines) and ATM term structure (right). Coverage badge: "Coverage: 73%" next to ticker. Warning below 40%: "Sparse data — surface may be unreliable." One job: "explore the surface."

4. **Strategies** (`/strategies`): Template picker as primary entry point (straddle, strangle, iron condor, butterfly, collar) with payoff diagram thumbnail for each. Below: custom leg builder (secondary). Right canvas: live payoff diagram with breakeven annotations, max profit/loss labels. One job: "build a payoff."

5. **Backtest** (`/backtest`): Strategy config (from /strategies or manual), date range presets (1M, 3M, 6M, 1Y, 3Y) + custom date picker. Canvas: cumulative P&L line chart, risk metrics row (Sharpe, Max Drawdown, Win Rate in mono). One job: "evaluate performance."

6. **Market** (`/market`): Candlestick + volume chart (6M default), options chain table (scrolls horizontally on tablet), HV chart with IV overlay. One job: "see the market."

### Interaction States

| Feature | Loading | Empty | Error | Success |
|---------|---------|-------|-------|---------|
| Dashboard pricing | Skeleton workspace + "Waking pricing engine... ~15s" badge | Pre-populated (never empty) | Inline error below ticker: "No data for XYZ" | Price + Greeks fill in, curve animates |
| Sensitivity charts | Shimmer skeleton at chart height | Pre-populated | "Could not compute — check parameters" banner | Chart renders with fade-in |
| Heatmap | Shimmer grid | Pre-populated | Same banner | Grid fills in |
| Monte Carlo | "Running simulation..." + elapsed time counter | Requires button click | "Simulation failed — reduce paths" | Histogram + metrics appear |
| Vol surface | Shimmer 3D region | "Enter a ticker to generate surface" | "Sparse data" warning OR "Network error — could not reach data provider" (distinct from bad ticker) | Surface renders with rotation |
| Strategy payoff | Instant (client-side math) | Template picker shown, no legs yet — "Pick a strategy or build custom" | N/A (client-side) | Payoff curve + breakevens |
| Backtest | "Running backtest..." + elapsed time | "Configure a strategy to backtest" | "No price data available for this period" | P&L chart + metrics |
| Options chain | Shimmer table rows | "Enter a ticker" | "No chain data — ticker may not have listed options" | Table fills in |

**Error presentation pattern**: Inline error below the triggering input for bad data (wrong ticker, invalid params). Top banner for infrastructure errors (network down, API timeout). Never toast — toasts disappear and financial data errors need to persist.

**SWR behavior**: Show stale data during revalidation (stale-while-revalidate is correct for financial data — better to show slightly old prices than flash a skeleton).

### Responsive Behavior

- **>1024px (desktop)**: Full sidebar (240px) + workspace canvas
- **768-1024px (tablet)**: Sidebar collapses to 48px icon rail (labels on hover, toggle to expand). Charts reflow to 100% width. Options chain table scrolls horizontally.
- **<768px (mobile)**: Not a primary target. Icon rail persists. Single-column layout. Charts stack vertically.

### Accessibility

- ARIA labels on all data tables (`role="table"`, column headers)
- ARIA labels on chart regions (`role="img"`, `aria-label="Delta sensitivity chart"`)
- Screen reader text for Greek symbols: "Delta: 0.5432" not just "Δ 0.5432"
- Keyboard nav: Tab through sidebar → inputs → action buttons → charts
- Minimum touch target: 44px on all interactive elements
- Color contrast: WCAG AA minimum (4.5:1 for text against --bg and --surface)
- Motion: respect `prefers-reduced-motion` — disable chart animations, keep functionality

### Greeks Display Format

- Delta, Rho: 4 decimal places
- Gamma: 6 decimal places (very small values)
- Vega, Theta: 4 decimal places
- Always show sign (+0.5432, -0.0234)
- Label format: "Δ Delta" with full name for screen readers
- Color: negative values in `--negative`, positive in `--text-primary` (NOT --positive — Greeks are not directional bets)
- Units shown in tooltip: "Theta: daily decay per $1 of option premium"

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

## Engineering Review Amendments

The following decisions were made during /plan-eng-review on 2026-03-23:

1. **Extract `models/engine.py`** — Orchestration functions (`option_metrics`, `price_with_model`, `numerical_greeks`, `implied_volatility`, `calculate_historical_volatility`, `iv_hv_stats`, `atm_iv_from_chain`, `risk_reversal_and_fly`) move from `analytics.py` to `models/engine.py`. Service layer calls engine. Engine is testable independently.

2. **CORS from environment variable** — `CORS_ORIGINS` env var, defaulting to `http://localhost:3000` for dev. Production URL configured per-environment.

3. **Phased migration** — Not a big-bang restructure. Four phases:
   - Phase 1: Fix broken imports, create `models/engine.py`, fix deprecated `datetime.utcnow()`, add engine tests
   - Phase 2: Add FastAPI layer (`api/`) on top of existing `models/`, `data/`, `strategies/`
   - Phase 3: Build Next.js frontend (`web/`)
   - Phase 4: Delete Streamlit code (`app.py`, `ui_components.py`, `analytics.py`), update CI, add Docker Compose

4. **Fix deprecated `datetime.utcnow()`** — Replace with `datetime.now(datetime.UTC)` in `data_service.py`.

5. **Tests mandatory per phase** — Each phase includes its own test suite. No phase merges without passing tests. 38 new tests needed across engine, API, frontend, and E2E.

6. **MC timeout + input limits** — Cap `mc_paths` at 50,000 and `mc_steps` at 500 in Pydantic schemas. 30-second request timeout in FastAPI middleware.

7. **Critical gap: yfinance error disambiguation** — Network errors vs bad ticker should return distinct error messages to the frontend.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 5 issues, 1 critical gap |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | CLEAR | score: 4/10 → 8/10, 7 decisions |

- **UNRESOLVED:** 0
- **VERDICT:** ENG + DESIGN CLEARED — ready to implement. Run `/design-consultation` before Phase 3 (frontend) to create DESIGN.md.
