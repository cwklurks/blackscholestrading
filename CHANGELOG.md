# Changelog

All notable changes to this project will be documented in this file.

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
