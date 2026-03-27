# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1.0] - 2026-03-27

### Added
- Warm amber design system per DESIGN.md: #D4A017 accent, warm gray neutrals, 0.25rem radius
- Semantic Greek color tokens (delta=blue, gamma=teal, vega=amber, theta=red, rho=purple) in CSS and charts
- Shared ParamInput component with font-mono tabular-nums styling
- Number formatting utilities: formatPrice (adaptive precision for deep-OTM), formatGreek (per-Greek decimals + theta /d suffix), formatPnl
- Guided empty states on all pages replacing generic dashed boxes
- BST brand mark in sidebar with Geist Mono amber typography
- ARIA landmarks on sidebar and main content area
- Ticker chip suggestions on market page empty state

### Changed
- Replaced zero-chroma oklch palette with warm hex values (light + dark mode)
- Chart theme derived from app tokens instead of Plotly's plotly_dark template
- Dashboard renamed to Workspace throughout
- Parameter rails styled as flush surface panels instead of bordered cards
- Result areas use border-bottom separators instead of card wrappers
- Active nav state uses amber tint (bg-primary/10) instead of generic accent
- All hardcoded emerald-500/red-500 replaced with text-positive/text-negative tokens
- Greek values use context-appropriate precision (delta 3dp, vega 2dp, theta with /d)
- Page headings use font-semibold instead of font-bold

### Fixed
- ResultStat Greek precision now consistent with GreeksRow (was .toFixed(4) everywhere)
- formatPrice shows 4 decimals for deep-OTM options below $0.01
- Backtest empty state no longer references nonexistent template picker

## [1.0.0.0] - 2026-03-27

### Added
- FastAPI REST API with 10 endpoints: /price, /heatmap, /monte-carlo, /volatility-surface, /market/{ticker}, /chain/{ticker}, /strategy/payoff, /backtest, /health, /models
- Pydantic schema validation with MC param clamping, ticker sanitization, and range bounds
- Next.js 16 frontend with 7 pages: Dashboard, Pricing, Volatility, Strategies, Backtest, Market
- Plotly chart components: sensitivity, payoff, heatmap, 3D vol surface, candlestick
- Typed API client with SWR hooks for all endpoints
- Collapsible sidebar navigation with responsive icon rail
- Multi-leg strategy builder with 5 preset templates (straddle, strangle, iron condor, butterfly, collar)
- Historical backtesting with Sharpe ratio, max drawdown, and win rate metrics
- Docker Compose for local development (api on :8000, web on :3000)
- GitHub Actions CI with parallel api-tests and web-tests jobs
- 89 Python tests (engine + API endpoints) and 32 frontend tests (API client, components)

### Changed
- Extracted pricing engine from analytics.py into models/engine.py
- Moved all Python modules into api/ package (monorepo structure)
- Replaced deprecated datetime.utcnow() with datetime.now(UTC)
- Fixed broken models.utils imports to use utils.constants

### Removed
- Legacy Streamlit UI (app.py, ui_components.py, analytics.py)
