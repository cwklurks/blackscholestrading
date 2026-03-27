# Black-Scholes Trader Full Replatform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replatform the Black-Scholes Trader from a broken Streamlit monolith to a FastAPI backend + Next.js frontend, producing a portfolio-quality full-stack options pricing workstation.

**Architecture:** Python pricing engine (models/, strategies/, data/) served via FastAPI REST API, consumed by a Next.js 15 frontend with shadcn/ui, Plotly charts, and SWR caching. Monorepo with api/ and web/ directories.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, NumPy/SciPy/Numba, pytest | Next.js 15, TypeScript, shadcn/ui, Tailwind CSS, Plotly.js, SWR, Vitest

**Spec:** `docs/superpowers/specs/2026-03-23-full-replatform-design.md`

---

## Phase 1: Fix & Stabilize Python Engine

Phase 1 fixes the broken build, extracts the engine layer from analytics.py, fixes deprecations, and adds comprehensive engine tests. After this phase, `pytest` passes and the models layer is clean.

### Task 1: Fix Broken Imports

The build is broken because `models/black_scholes.py` and `models/simulation.py` import from `models.utils` which doesn't exist. The constants live in `utils/constants.py`.

**Files:**
- Modify: `models/black_scholes.py` (line 5)
- Modify: `models/simulation.py` (line 7)
- Modify: `strategies/backtest.py` (line 22)

- [ ] **Step 1: Fix `models/black_scholes.py` import**

Change line 5 from:
```python
from models.utils import clamp_inputs, EPS_TIME, EPS_VOL
```
to:
```python
from utils.constants import clamp_inputs, EPS_TIME, EPS_VOL
```

- [ ] **Step 2: Fix `models/simulation.py` import**

Change line 7 from:
```python
from models.utils import clamp_inputs
```
to:
```python
from utils.constants import clamp_inputs
```

- [ ] **Step 3: Verify `strategies/backtest.py` import**

Confirm line 22 already uses:
```python
from utils.constants import EPS_TIME, clamp_inputs
```
If it uses `models.utils`, fix it the same way.

- [ ] **Step 4: Run tests to verify fix**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All existing tests PASS (test_analytics, test_advanced_models, test_convergence)

- [ ] **Step 5: Commit**

```bash
git add models/black_scholes.py models/simulation.py strategies/backtest.py
git commit -m "fix: resolve broken models.utils imports to use utils.constants"
```

---

### Task 2: Extract models/engine.py

The orchestration functions (`option_metrics`, `price_with_model`, `numerical_greeks`, `implied_volatility`, etc.) currently exist ONLY in `analytics.py`. They are the glue between raw pricing models and any consumer (API, Streamlit, tests). Extract them into `models/engine.py` — a standalone module testable without FastAPI.

**Files:**
- Create: `models/engine.py`
- Modify: `models/__init__.py` (add engine exports)

- [ ] **Step 1: Create `models/engine.py`**

Extract these functions from `analytics.py` into `models/engine.py`. Each function's signature and logic stays identical — we're moving, not rewriting. The functions to extract:

```python
"""
Pricing engine — orchestration layer between raw models and consumers.

Routes pricing requests to the correct model, computes Greeks analytically
(BS) or numerically (bump-and-revalue for MC models), and provides IV
solving, historical volatility, and chain analytics.

    Consumer (API / CLI / test)
        |
        v
    option_metrics() ── dispatches to ──> price_with_model()
        |                                      |
        v                                      v
    BS: analytical Greeks              Non-BS: numerical_greeks()
        |                                      |
        v                                      v
    { price, delta, gamma, ... }       { price, delta, gamma, ... }
"""
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy import exp, log, sqrt
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from models.black_scholes import BlackScholesModel, bs_call_price_vectorized
from models.numerical import binomial_american_option
from models.simulation import (
    heston_mc_price,
    garch_mc_price,
    bates_jump_diffusion_mc_price,
    monte_carlo_option_price,
)
from utils.constants import EPS_TIME, EPS_VOL, clamp_inputs


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    borrow_cost: float = 0.0,
    option_type: str = "call",
) -> float:
    """Solve for implied volatility using bounded minimization."""
    T, _ = clamp_inputs(T, EPS_VOL)

    def objective(sigma: float) -> float:
        sigma = max(sigma, EPS_VOL)
        model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
        if option_type == "call":
            return abs(model.call_price() - option_price)
        return abs(model.put_price() - option_price)

    result = minimize_scalar(objective, bounds=(0.001, 5), method="bounded")
    return float(result.x)


def calculate_historical_volatility(prices: pd.Series, periods: int = 252) -> float:
    """Annualized historical volatility from log returns."""
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        return np.nan
    return float(returns.std() * np.sqrt(periods))


def price_with_model(
    model_name: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float,
    option_type: str,
    model_params: dict,
) -> float:
    """Route a pricing request to the correct model."""
    if model_name == "Black-Scholes":
        model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
        return model.call_price() if option_type == "call" else model.put_price()
    if model_name == "Binomial (American)":
        return binomial_american_option(
            S, K, T, r, q, sigma, borrow_cost,
            steps=model_params.get("binomial_steps", 150),
            option_type=option_type,
        )
    if model_name == "Heston MC":
        return heston_mc_price(
            S, K, T, r, q, sigma, borrow_cost, option_type,
            kappa=model_params.get("heston_kappa", 1.5),
            theta=model_params.get("heston_theta", 0.04),
            v0=model_params.get("heston_v0", sigma ** 2),
            rho=model_params.get("heston_rho", -0.7),
            vol_of_vol=model_params.get("heston_vol_of_vol", 0.3),
            paths=model_params.get("mc_paths", 4000),
            steps=model_params.get("mc_steps", 120),
        )
    if model_name == "GARCH MC":
        return garch_mc_price(
            S, K, T, r, q, sigma, borrow_cost, option_type,
            alpha0=model_params.get("garch_alpha0", 2e-6),
            alpha1=model_params.get("garch_alpha1", 0.08),
            beta1=model_params.get("garch_beta1", 0.9),
            paths=model_params.get("mc_paths", 4000),
        )
    if model_name == "Bates Jump-Diffusion":
        return bates_jump_diffusion_mc_price(
            S, K, T, r, q, sigma, borrow_cost, option_type,
            lambda_jump=model_params.get("jump_lambda", 0.1),
            mu_jump=model_params.get("jump_mu", -0.05),
            delta_jump=model_params.get("jump_delta", 0.2),
            paths=model_params.get("mc_paths", 4000),
            steps=model_params.get("mc_steps", 120),
        )
    # Fallback to BS
    model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
    return model.call_price() if option_type == "call" else model.put_price()


def numerical_greeks(
    model_name: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float,
    option_type: str,
    model_params: dict,
) -> dict:
    """Compute Greeks via bump-and-revalue for non-BS models."""
    base_price = price_with_model(model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params)
    ds = max(0.01, S * 0.01)
    dv = max(0.0001, sigma * 0.05)
    dt_step = min(1 / 365, T * 0.5) if T > 0 else 1 / 365
    dr = 0.0001

    price_up = price_with_model(model_name, S + ds, K, T, r, q, sigma, borrow_cost, option_type, model_params)
    price_dn = price_with_model(model_name, S - ds, K, T, r, q, sigma, borrow_cost, option_type, model_params)
    delta = (price_up - price_dn) / (2 * ds)
    gamma = (price_up - 2 * base_price + price_dn) / (ds ** 2)

    price_vol_up = price_with_model(model_name, S, K, T, r, q, sigma + dv, borrow_cost, option_type, model_params)
    price_vol_dn = price_with_model(model_name, S, K, T, r, q, sigma - dv, borrow_cost, option_type, model_params)
    vega = (price_vol_up - price_vol_dn) / (2 * dv) / 100

    T_forward = max(T - dt_step, EPS_TIME)
    price_time = price_with_model(model_name, S, K, T_forward, r, q, sigma, borrow_cost, option_type, model_params)
    theta = (price_time - base_price) / dt_step / -365

    price_r_up = price_with_model(model_name, S, K, T, r + dr, q, sigma, borrow_cost, option_type, model_params)
    price_r_dn = price_with_model(model_name, S, K, T, r - dr, q, sigma, borrow_cost, option_type, model_params)
    rho = (price_r_up - price_r_dn) / (2 * dr) / 100

    return {"price": base_price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def option_metrics(
    model_name: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float,
    model_params: dict,
    option_type: str = "call",
) -> dict:
    """Compute price + all Greeks for any model. BS uses analytical; others use numerical."""
    if model_name == "Black-Scholes":
        bs = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
        return {
            "price": bs.call_price() if option_type == "call" else bs.put_price(),
            "delta": bs.delta_call() if option_type == "call" else bs.delta_put(),
            "gamma": bs.gamma(),
            "vega": bs.vega(),
            "theta": bs.theta_call() if option_type == "call" else bs.theta_put(),
            "rho": bs.rho_call() if option_type == "call" else bs.rho_put(),
        }
    return numerical_greeks(model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params)


def iv_hv_stats(implied_vol: float, hv_series: pd.Series) -> Optional[dict]:
    """Compute IV vs HV statistics."""
    hv_series = hv_series.dropna()
    if hv_series.empty or implied_vol is None:
        return None
    hv_current = hv_series.iloc[-1]
    percentile = float((hv_series < implied_vol).mean())
    rank = float(pd.concat([hv_series, pd.Series([implied_vol])]).rank(pct=True).iloc[-1])
    edge = float(implied_vol - hv_current)
    return {"hv_current": hv_current, "iv_hv_percentile": percentile, "iv_hv_rank": rank, "edge": edge}


def atm_iv_from_chain(options_df: pd.DataFrame, expiry: str, spot: float) -> Optional[float]:
    """Find ATM implied vol from an options chain."""
    subset = options_df[options_df["expiration"] == expiry]
    if subset.empty:
        return None
    idx = (subset["strike"] - spot).abs().idxmin()
    if pd.isna(idx):
        return None
    return float(subset.loc[idx, "impliedVolatility"])


def risk_reversal_and_fly(
    options_df: pd.DataFrame,
    spot: float,
    expiry: str,
    call_moneyness: float = 1.1,
    put_moneyness: float = 0.9,
) -> Optional[dict]:
    """Compute risk reversal and butterfly from options chain."""
    subset = options_df[options_df["expiration"] == expiry]
    if subset.empty or spot <= 0:
        return None
    atm_iv = atm_iv_from_chain(options_df, expiry, spot)

    def _nearest_iv(df: pd.DataFrame, target: float) -> Optional[float]:
        if df.empty:
            return None
        idx = (df["strike"] - target).abs().idxmin()
        if pd.isna(idx):
            return None
        return float(df.loc[idx, "impliedVolatility"])

    calls = subset[subset["type"].str.lower() == "call"]
    puts = subset[subset["type"].str.lower() == "put"]
    rr_call_iv = _nearest_iv(calls, spot * call_moneyness)
    rr_put_iv = _nearest_iv(puts, spot * put_moneyness)
    if atm_iv is None or rr_call_iv is None or rr_put_iv is None:
        return None
    rr = rr_call_iv - rr_put_iv
    fly = 0.5 * (rr_call_iv + rr_put_iv) - atm_iv
    return {"risk_reversal": rr, "butterfly": fly, "atm_iv": atm_iv}
```

- [ ] **Step 2: Update `models/__init__.py`**

Add engine exports:
```python
from models.engine import (
    option_metrics,
    price_with_model,
    numerical_greeks,
    implied_volatility,
    calculate_historical_volatility,
    iv_hv_stats,
    atm_iv_from_chain,
    risk_reversal_and_fly,
)
```

Add to `__all__`:
```python
"option_metrics",
"price_with_model",
"numerical_greeks",
"implied_volatility",
"calculate_historical_volatility",
"iv_hv_stats",
"atm_iv_from_chain",
"risk_reversal_and_fly",
```

- [ ] **Step 3: Update test imports**

In `tests/test_analytics.py`, change:
```python
from analytics import (
    backtest_option_strategy,
    iv_hv_stats,
)
```
to:
```python
from models.engine import iv_hv_stats
from strategies.backtest import backtest_option_strategy
```

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/engine.py models/__init__.py tests/test_analytics.py
git commit -m "refactor: extract engine.py from analytics.py for orchestration layer"
```

---

### Task 3: Fix Deprecated datetime.utcnow()

**Files:**
- Modify: `data_service.py` (3 occurrences)

- [ ] **Step 1: Update import and replace all occurrences**

In `data_service.py`, update the existing import:
```python
from datetime import datetime, timedelta, UTC
```

Replace all `datetime.utcnow()` with `datetime.now(UTC)` (4 occurrences: lines 50, 69, 92, 136).

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS, no DeprecationWarning

- [ ] **Step 3: Commit**

```bash
git add data_service.py
git commit -m "fix: replace deprecated datetime.utcnow() with datetime.now(UTC)"
```

---

### Task 4: Add Engine Tests

Write comprehensive tests for the newly extracted `models/engine.py`. These cover the 10 gaps identified in the eng review test diagram.

**Files:**
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write engine test file**

```python
"""Tests for models/engine.py — orchestration layer."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.engine import (
    option_metrics,
    price_with_model,
    numerical_greeks,
    implied_volatility,
    calculate_historical_volatility,
    iv_hv_stats,
    atm_iv_from_chain,
    risk_reversal_and_fly,
)


class TestPriceWithModel:
    """Test model routing for all 5 models."""

    @pytest.mark.parametrize("model_name", [
        "Black-Scholes",
        "Binomial (American)",
        "Heston MC",
        "GARCH MC",
        "Bates Jump-Diffusion",
    ])
    def test_routes_to_correct_model(self, model_name):
        price = price_with_model(
            model_name, S=100, K=100, T=0.5, r=0.05, q=0.0,
            sigma=0.2, borrow_cost=0.0, option_type="call",
            model_params={"mc_paths": 1000, "mc_steps": 50},
        )
        assert math.isfinite(price)
        assert price > 0

    def test_unknown_model_falls_back_to_bs(self):
        price = price_with_model(
            "NonexistentModel", S=100, K=100, T=0.5, r=0.05, q=0.0,
            sigma=0.2, borrow_cost=0.0, option_type="call",
            model_params={},
        )
        assert math.isfinite(price)


class TestOptionMetrics:
    """Test the unified metrics dispatcher."""

    def test_bs_returns_analytical_greeks(self):
        metrics = option_metrics(
            "Black-Scholes", S=100, K=100, T=0.5, r=0.05, q=0.0,
            sigma=0.2, borrow_cost=0.0, model_params={}, option_type="call",
        )
        assert all(k in metrics for k in ["price", "delta", "gamma", "vega", "theta", "rho"])
        assert 0 < metrics["delta"] < 1  # Call delta is between 0 and 1
        assert metrics["gamma"] > 0

    def test_non_bs_returns_numerical_greeks(self):
        metrics = option_metrics(
            "Heston MC", S=100, K=100, T=0.5, r=0.05, q=0.0,
            sigma=0.2, borrow_cost=0.0,
            model_params={"mc_paths": 1000, "mc_steps": 50},
            option_type="call",
        )
        assert all(k in metrics for k in ["price", "delta", "gamma", "vega", "theta", "rho"])
        assert math.isfinite(metrics["delta"])

    def test_put_option_negative_delta(self):
        metrics = option_metrics(
            "Black-Scholes", S=100, K=100, T=0.5, r=0.05, q=0.0,
            sigma=0.2, borrow_cost=0.0, model_params={}, option_type="put",
        )
        assert metrics["delta"] < 0


class TestNumericalGreeks:
    """Test bump-and-revalue edge cases."""

    def test_near_zero_time(self):
        metrics = numerical_greeks(
            "Heston MC", S=100, K=100, T=0.001, r=0.05, q=0.0,
            sigma=0.2, borrow_cost=0.0, option_type="call",
            model_params={"mc_paths": 500, "mc_steps": 10},
        )
        assert all(math.isfinite(v) for v in metrics.values())

    def test_near_zero_sigma(self):
        metrics = numerical_greeks(
            "Binomial (American)", S=100, K=100, T=0.5, r=0.05, q=0.0,
            sigma=0.01, borrow_cost=0.0, option_type="call",
            model_params={"binomial_steps": 50},
        )
        assert all(math.isfinite(v) for v in metrics.values())


class TestImpliedVolatility:
    """Test IV solver."""

    def test_recovers_known_vol(self):
        """If we price at sigma=0.3, IV solver should recover ~0.3."""
        from models.black_scholes import BlackScholesModel
        bs = BlackScholesModel(100, 100, 0.5, 0.05, 0.3)
        target_price = bs.call_price()
        recovered_iv = implied_volatility(target_price, 100, 100, 0.5, 0.05, option_type="call")
        assert abs(recovered_iv - 0.3) < 0.01

    def test_put_iv(self):
        from models.black_scholes import BlackScholesModel
        bs = BlackScholesModel(100, 100, 0.5, 0.05, 0.25)
        target_price = bs.put_price()
        recovered_iv = implied_volatility(target_price, 100, 100, 0.5, 0.05, option_type="put")
        assert abs(recovered_iv - 0.25) < 0.01

    def test_zero_price_returns_low_vol(self):
        iv = implied_volatility(0.0, 100, 100, 0.5, 0.05)
        assert math.isfinite(iv)
        assert iv < 0.05  # Near zero


class TestHistoricalVolatility:
    """Test HV calculation."""

    def test_returns_finite_for_valid_series(self):
        prices = pd.Series(np.linspace(90, 110, 60))
        hv = calculate_historical_volatility(prices)
        assert math.isfinite(hv)
        assert hv > 0

    def test_returns_nan_for_empty_series(self):
        hv = calculate_historical_volatility(pd.Series(dtype=float))
        assert math.isnan(hv)

    def test_single_price_returns_nan(self):
        hv = calculate_historical_volatility(pd.Series([100.0]))
        assert math.isnan(hv)


class TestChainAnalytics:
    """Test IV/HV stats and chain analytics."""

    def _make_chain(self):
        """Create a mock options chain DataFrame."""
        return pd.DataFrame({
            "strike": [90, 95, 100, 105, 110, 90, 95, 100, 105, 110],
            "type": ["Call"] * 5 + ["Put"] * 5,
            "expiration": ["2024-06-21"] * 10,
            "impliedVolatility": [0.30, 0.25, 0.22, 0.25, 0.30, 0.32, 0.27, 0.22, 0.27, 0.32],
            "lastPrice": [12, 8, 5, 3, 1.5, 1, 2, 4, 7, 11],
        })

    def test_atm_iv_finds_nearest_strike(self):
        chain = self._make_chain()
        iv = atm_iv_from_chain(chain, "2024-06-21", spot=101)
        assert iv is not None
        assert abs(iv - 0.22) < 0.01  # ATM is strike 100

    def test_atm_iv_empty_chain(self):
        chain = self._make_chain()
        iv = atm_iv_from_chain(chain, "2025-01-01", spot=100)  # Wrong expiry
        assert iv is None

    def test_risk_reversal_and_fly(self):
        chain = self._make_chain()
        result = risk_reversal_and_fly(chain, spot=100, expiry="2024-06-21")
        assert result is not None
        assert "risk_reversal" in result
        assert "butterfly" in result
        assert math.isfinite(result["risk_reversal"])

    def test_risk_reversal_empty_chain(self):
        chain = self._make_chain()
        result = risk_reversal_and_fly(chain, spot=100, expiry="2025-01-01")
        assert result is None

    def test_iv_hv_stats_valid(self):
        hv_series = pd.Series([0.15, 0.18, 0.20, 0.22, 0.19, 0.21])
        result = iv_hv_stats(0.25, hv_series)
        assert result is not None
        assert math.isfinite(result["edge"])
        assert result["edge"] > 0  # IV > HV

    def test_iv_hv_stats_empty_series(self):
        result = iv_hv_stats(0.25, pd.Series(dtype=float))
        assert result is None
```

- [ ] **Step 2: Run new tests**

Run: `.venv/bin/python -m pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 3: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine.py
git commit -m "test: add comprehensive tests for models/engine.py orchestration layer"
```

---

## Phase 2: FastAPI Backend

Phase 2 adds the REST API layer on top of the existing models. After this phase, `uvicorn api.app.main:app` serves all pricing/market/strategy endpoints.

### Task 5: Scaffold API Directory + FastAPI App

**Files:**
- Create: `api/` directory structure
- Create: `api/app/__init__.py`
- Create: `api/app/main.py` (FastAPI entrypoint)
- Create: `api/requirements.txt`
- Create: `api/pyproject.toml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p api/app/routers api/app/schemas api/app/services
touch api/__init__.py api/app/__init__.py api/app/routers/__init__.py
touch api/app/schemas/__init__.py api/app/services/__init__.py
```

- [ ] **Step 2: Symlink or copy existing code into api/**

For the phased migration, create symlinks so both the old and new import paths work:
```bash
ln -s ../../models api/models
ln -s ../../utils api/utils
ln -s ../../data api/data
ln -s ../../strategies api/strategies
ln -s ../../data_service.py api/data_service.py
```

(These will be replaced with actual moves in Phase 4.)

- [ ] **Step 3: Create `api/app/main.py`**

```python
"""FastAPI entrypoint for the Black-Scholes Pricing Engine."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.app.routers import pricing, market, backtest

app = FastAPI(
    title="Black-Scholes Pricing Engine",
    description="Production-grade options pricing with Heston, GARCH, and Bates models",
    version="1.0.0",
)

# CORS — configurable via environment variable
cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pricing.router, prefix="/api")
app.include_router(market.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models": [
            "Black-Scholes",
            "Binomial (American)",
            "Heston MC",
            "GARCH MC",
            "Bates Jump-Diffusion",
        ],
    }


MODEL_REGISTRY = [
    {
        "name": "Black-Scholes",
        "description": "Closed-form analytical solution (European options)",
        "params": {},
    },
    {
        "name": "Binomial (American)",
        "description": "CRR binomial tree for American options",
        "params": {
            "binomial_steps": {"type": "int", "default": 150, "description": "Number of tree steps"},
        },
    },
    {
        "name": "Heston MC",
        "description": "Heston stochastic volatility model via Monte Carlo",
        "params": {
            "heston_kappa": {"type": "float", "default": 1.5, "description": "Mean reversion speed"},
            "heston_theta": {"type": "float", "default": 0.04, "description": "Long-run variance"},
            "heston_rho": {"type": "float", "default": -0.7, "description": "Price-vol correlation"},
            "heston_vol_of_vol": {"type": "float", "default": 0.3, "description": "Volatility of volatility"},
            "mc_paths": {"type": "int", "default": 4000, "description": "Simulation paths (max 50000)"},
            "mc_steps": {"type": "int", "default": 120, "description": "Time steps (max 500)"},
        },
    },
    {
        "name": "GARCH MC",
        "description": "GARCH(1,1) volatility model via Monte Carlo",
        "params": {
            "garch_alpha0": {"type": "float", "default": 2e-6, "description": "GARCH constant"},
            "garch_alpha1": {"type": "float", "default": 0.08, "description": "ARCH coefficient"},
            "garch_beta1": {"type": "float", "default": 0.9, "description": "GARCH coefficient"},
            "mc_paths": {"type": "int", "default": 4000, "description": "Simulation paths (max 50000)"},
        },
    },
    {
        "name": "Bates Jump-Diffusion",
        "description": "Jump-diffusion model for tail risk pricing",
        "params": {
            "jump_lambda": {"type": "float", "default": 0.1, "description": "Jump intensity"},
            "jump_mu": {"type": "float", "default": -0.05, "description": "Mean jump size"},
            "jump_delta": {"type": "float", "default": 0.2, "description": "Jump volatility"},
            "mc_paths": {"type": "int", "default": 4000, "description": "Simulation paths (max 50000)"},
            "mc_steps": {"type": "int", "default": 120, "description": "Time steps (max 500)"},
        },
    },
]


@app.get("/api/models")
async def list_models():
    return MODEL_REGISTRY
```

- [ ] **Step 3b: Add 30-second request timeout middleware**

Add to `api/app/main.py` before the router includes:

```python
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=30.0)
        except asyncio.TimeoutError:
            return JSONResponse({"detail": "Request timeout (30s limit)"}, status_code=408)


app.add_middleware(TimeoutMiddleware)
```

- [ ] **Step 4: Create `api/requirements.txt`**

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
numpy
scipy
pandas
numba
yfinance
```

- [ ] **Step 5: Create `api/pyproject.toml`**

```toml
[project]
name = "blackscholes-api"
version = "1.0.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[tool.mypy]
mypy_path = "."
ignore_missing_imports = true

[tool.ruff]
target-version = "py310"
```

- [ ] **Step 6: Test that the app starts**

Run: `cd api && pip install fastapi uvicorn && python -m uvicorn app.main:app --port 8000` (verify no import errors, then Ctrl+C)

- [ ] **Step 7: Commit**

```bash
git add api/
git commit -m "feat: scaffold FastAPI app structure with CORS and health endpoint"
```

---

### Task 6: Pydantic Schemas

Define the request/response models for all API endpoints. The schema layer handles the `q` → `dividend_yield` translation.

**Files:**
- Create: `api/app/schemas/pricing.py`
- Create: `api/app/schemas/market.py`
- Create: `api/app/schemas/backtest.py`

- [ ] **Step 1: Create pricing schemas**

```python
"""Pydantic schemas for pricing endpoints."""
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class RangeSpec(BaseModel):
    min: float
    max: float
    steps: int = Field(ge=2, le=100, default=20)


class PricingRequest(BaseModel):
    model: str = Field(default="Black-Scholes", description="Pricing model name")
    S: float = Field(gt=0, description="Spot price")
    K: float = Field(gt=0, description="Strike price")
    T: float = Field(ge=0, description="Time to expiry in years")
    r: float = Field(description="Risk-free rate")
    sigma: float = Field(ge=0, le=5.0, description="Volatility")
    q: float = Field(default=0.0, description="Dividend yield")
    borrow_cost: float = Field(default=0.0, description="Borrow cost")
    option_type: str = Field(default="call", pattern="^(call|put)$")
    model_params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_mc_params(self):
        """Enforce mc_paths <= 50000 and mc_steps <= 500 (eng review #6)."""
        if "mc_paths" in self.model_params:
            self.model_params["mc_paths"] = min(self.model_params["mc_paths"], 50000)
        if "mc_steps" in self.model_params:
            self.model_params["mc_steps"] = min(self.model_params["mc_steps"], 500)
        return self

    def to_engine_kwargs(self) -> dict:
        return {
            "S": self.S, "K": self.K, "T": self.T, "r": self.r,
            "q": self.q, "sigma": self.sigma, "borrow_cost": self.borrow_cost,
            "model_params": self.model_params, "option_type": self.option_type,
        }


class PricingResponse(BaseModel):
    model: str
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class HeatmapRequest(BaseModel):
    K: float = Field(gt=0)
    T: float = Field(ge=0)
    r: float
    q: float = 0.0
    borrow_cost: float = 0.0
    spot_range: RangeSpec
    vol_range: RangeSpec


class HeatmapResponse(BaseModel):
    spot_values: list[float]
    vol_values: list[float]
    call_prices: list[list[float]]
    put_prices: list[list[float]]


class MonteCarloRequest(BaseModel):
    S: float = Field(gt=0)
    K: float = Field(gt=0)
    T: float = Field(ge=0)
    r: float
    sigma: float = Field(ge=0)
    paths: int = Field(ge=100, le=50000, default=10000)
    option_type: str = Field(default="call", pattern="^(call|put)$")
    q: float = 0.0
    borrow_cost: float = 0.0


class MonteCarloResponse(BaseModel):
    price: float
    std_error: float
    terminal_prices: list[float]
    confidence_interval: list[float]


class VolSurfaceRequest(BaseModel):
    ticker: str
    strikes: Optional[list[float]] = None
    expirations: Optional[list[str]] = None


class VolSurfacePoint(BaseModel):
    strike: float
    expiry: str
    iv: Optional[float]


class VolSurfaceResponse(BaseModel):
    surface: list[VolSurfacePoint]
    smile_data: dict[str, list[dict]]
    coverage: float
```

- [ ] **Step 2: Create market schemas**

```python
"""Pydantic schemas for market data endpoints."""
from typing import Optional
from pydantic import BaseModel


class OHLCVRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class MarketResponse(BaseModel):
    price: float
    history: list[OHLCVRow]
    historical_vol: float
    fetched_at: str


class ChainRow(BaseModel):
    strike: float
    lastPrice: float
    iv: Optional[float] = None
    volume: Optional[float] = None
    oi: Optional[float] = None


class ChainResponse(BaseModel):
    calls: list[ChainRow]
    puts: list[ChainRow]
    expirations: list[str]
```

- [ ] **Step 3: Create backtest schemas**

```python
"""Pydantic schemas for strategy and backtest endpoints."""
from typing import Optional
from pydantic import BaseModel, Field


class StrategyLeg(BaseModel):
    type: str = Field(pattern="^(call|put)$")
    strike: float = Field(gt=0)
    qty: int = Field(ge=1, default=1)
    side: str = Field(pattern="^(long|short)$")
    entry_price: Optional[float] = None


class PayoffRequest(BaseModel):
    legs: list[StrategyLeg] = Field(min_length=1)
    spot_range: dict  # {min, max}
    S: float = Field(gt=0, description="Current spot for entry pricing")
    T: float = Field(ge=0, default=0.0833, description="Time to expiry for entry pricing")
    r: float = Field(default=0.05)
    sigma: float = Field(ge=0, default=0.2)


class PayoffResponse(BaseModel):
    prices: list[float]
    pnl: list[float]
    breakevens: list[float]
    max_profit: Optional[float]
    max_loss: Optional[float]


class BacktestLeg(BaseModel):
    type: str = Field(pattern="^(call|put)$")
    strike: float = Field(gt=0)
    expiry: str
    qty: int = Field(ge=1, default=1)
    side: str = Field(pattern="^(long|short)$")


class BacktestRequest(BaseModel):
    ticker: str
    legs: list[BacktestLeg] = Field(min_length=1)
    r: float = Field(default=0.05)
    sigma: float = Field(ge=0, default=0.2)


class PnLPoint(BaseModel):
    date: str
    pnl: float


class BacktestResponse(BaseModel):
    pnl_series: list[PnLPoint]
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    win_rate: float
```

- [ ] **Step 4: Commit**

```bash
git add api/app/schemas/
git commit -m "feat: add Pydantic schemas for all API endpoints"
```

---

### Task 7: Pricing Router + Service

**Files:**
- Create: `api/app/routers/pricing.py`
- Create: `api/app/services/pricing_service.py`

- [ ] **Step 1: Create pricing service**

The service is a thin wrapper — it calls `models/engine.py` and returns dicts that map to Pydantic responses.

```python
"""Pricing service — thin wrapper around models/engine.py."""
import numpy as np

from models.engine import option_metrics, implied_volatility
from models.black_scholes import bs_call_price_vectorized


def compute_price(model_name: str, **kwargs) -> dict:
    """Compute option price + Greeks."""
    metrics = option_metrics(model_name=model_name, **kwargs)
    return {"model": model_name, **metrics}


def compute_heatmap(K, T, r, q, borrow_cost, spot_range, vol_range) -> dict:
    """Compute BS call/put prices across a Spot x Vol grid."""
    spot_values = np.linspace(spot_range["min"], spot_range["max"], spot_range.get("steps", 20))
    vol_values = np.linspace(vol_range["min"], vol_range["max"], vol_range.get("steps", 20))
    spot_grid, vol_grid = np.meshgrid(spot_values, vol_values)

    call_prices = bs_call_price_vectorized(spot_grid, K, T, r, vol_grid, q, borrow_cost)
    # Put via put-call parity: P = C - S*e^(-qT) + K*e^(-rT)
    decay = np.exp(-(q + borrow_cost) * T)
    put_prices = call_prices - spot_grid * decay + K * np.exp(-r * T)

    return {
        "spot_values": spot_values.tolist(),
        "vol_values": vol_values.tolist(),
        "call_prices": call_prices.tolist(),
        "put_prices": put_prices.tolist(),
    }
```

- [ ] **Step 2: Create pricing router**

```python
"""Pricing API endpoints."""
from fastapi import APIRouter

from api.app.schemas.pricing import (
    PricingRequest, PricingResponse,
    HeatmapRequest, HeatmapResponse,
    MonteCarloRequest, MonteCarloResponse,
)
from api.app.services.pricing_service import compute_price, compute_heatmap
from models.simulation import monte_carlo_option_price

router = APIRouter(tags=["pricing"])


@router.post("/price", response_model=PricingResponse)
async def price_option(request: PricingRequest):
    result = compute_price(model_name=request.model, **request.to_engine_kwargs())
    return result


@router.post("/heatmap", response_model=HeatmapResponse)
async def compute_heatmap_endpoint(request: HeatmapRequest):
    result = compute_heatmap(
        K=request.K, T=request.T, r=request.r, q=request.q,
        borrow_cost=request.borrow_cost,
        spot_range=request.spot_range.model_dump(),
        vol_range=request.vol_range.model_dump(),
    )
    return result


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def monte_carlo(request: MonteCarloRequest):
    price, se, terminal = monte_carlo_option_price(
        S=request.S, K=request.K, T=request.T, r=request.r,
        sigma=request.sigma, num_simulations=request.paths,
        option_type=request.option_type, q=request.q,
        borrow_cost=request.borrow_cost,
    )
    ci = [price - 1.96 * se, price + 1.96 * se]
    return {
        "price": price,
        "std_error": se,
        "terminal_prices": terminal.tolist(),
        "confidence_interval": ci,
    }
```

- [ ] **Step 3: Commit**

```bash
git add api/app/routers/pricing.py api/app/services/pricing_service.py
git commit -m "feat: add pricing router with /price, /heatmap, /monte-carlo endpoints"
```

---

### Task 7b: Volatility Surface Router + Service

Implements `POST /api/volatility-surface` — computes IV grid across strikes/expiries with NaN handling and coverage field.

**Files:**
- Create: `api/app/services/volatility_service.py`
- Modify: `api/app/routers/pricing.py` (add vol surface route)

- [ ] **Step 1: Create volatility service**

```python
"""Volatility surface service — IV computation across strike/expiry grid."""
from typing import Optional
from models.engine import implied_volatility
from data_service import fetch_options_chain


def compute_volatility_surface(ticker: str, strikes: Optional[list], expirations: Optional[list]) -> dict:
    """Compute IV surface from live options chain data."""
    chain_df, available_exps, _ = fetch_options_chain(ticker)
    if chain_df is None or chain_df.empty:
        return {"surface": [], "smile_data": {}, "coverage": 0.0}

    exps = expirations or (available_exps[:5] if available_exps else [])

    surface = []
    smile_data = {}
    total_cells = 0
    valid_cells = 0

    for exp in exps:
        subset = chain_df[chain_df["expiration"] == exp]
        if subset.empty:
            continue
        exp_strikes = strikes or sorted(subset["strike"].unique().tolist())
        smile_points = []

        for strike in exp_strikes:
            total_cells += 1
            row = subset[(subset["strike"] == strike) & (subset["type"] == "Call")]
            iv = None
            if not row.empty:
                market_price = row.iloc[0].get("lastPrice", 0)
                if market_price and market_price > 0:
                    try:
                        # Get spot from chain or use nearest strike as proxy
                        spot = chain_df["strike"].median()  # Rough proxy
                        T = 30 / 365  # Approximate
                        iv = implied_volatility(market_price, spot, strike, T, 0.05)
                        valid_cells += 1
                    except Exception:
                        iv = None

            surface.append({"strike": strike, "expiry": exp, "iv": iv})
            smile_points.append({"strike": strike, "iv": iv})

        smile_data[exp] = smile_points

    coverage = valid_cells / total_cells if total_cells > 0 else 0.0
    return {"surface": surface, "smile_data": smile_data, "coverage": round(coverage, 3)}
```

- [ ] **Step 2: Add route to pricing router**

In `api/app/routers/pricing.py`, add:
```python
from api.app.schemas.pricing import VolSurfaceRequest, VolSurfaceResponse
from api.app.services.volatility_service import compute_volatility_surface

@router.post("/volatility-surface", response_model=VolSurfaceResponse)
async def volatility_surface(request: VolSurfaceRequest):
    result = compute_volatility_surface(request.ticker, request.strikes, request.expirations)
    return result
```

- [ ] **Step 3: Commit**

```bash
git add api/app/services/volatility_service.py api/app/routers/pricing.py
git commit -m "feat: add POST /api/volatility-surface endpoint with NaN handling"
```

---

### Task 8: Market Router + Service

**Files:**
- Create: `api/app/routers/market.py`
- Create: `api/app/services/market_service.py`

- [ ] **Step 1: Create market service**

Wraps the existing `data_service.py` functions with error disambiguation (yfinance network error vs bad ticker).

- [ ] **Step 2: Create market router**

Implements `GET /api/market/{ticker}` and `GET /api/chain/{ticker}`.

- [ ] **Step 3: Commit**

---

### Task 9: Strategy/Backtest Router + Service

**Files:**
- Create: `api/app/routers/backtest.py`
- Create: `api/app/services/backtest_service.py`
- Create: `strategies/multi_leg.py` (MultiLegStrategy)

- [ ] **Step 1: Create MultiLegStrategy**

Extends the existing Strategy ABC. Maintains a list of SingleOptionStrategy legs, handles per-leg expiration (P&L freezes at intrinsic when T=0), aggregates P&L.

- [ ] **Step 2: Create payoff computation service**

Implements multi-leg payoff diagram calculation (breakevens, max profit/loss).

- [ ] **Step 3: Create backtest router**

Implements `POST /api/strategy/payoff` and `POST /api/backtest` with Sharpe, drawdown, win rate metrics.

- [ ] **Step 4: Commit**

---

### Task 10: API Tests

**Files:**
- Create: `api/tests/test_api.py`

- [ ] **Step 1: Write API endpoint tests using FastAPI TestClient**

Test each endpoint: valid request → correct schema, invalid model → 422, negative strike → 422, MC path limits → enforced.

- [ ] **Step 2: Run all API tests**

Run: `cd api && python -m pytest tests/ -v`

- [ ] **Step 3: Commit**

---

## Phase 3: Next.js Frontend

Phase 3 builds the entire frontend. After this phase, `npm run dev` in web/ serves the pricing workstation that consumes the FastAPI backend.

### Task 11: Scaffold Next.js + shadcn/ui + Design Tokens

**Files:**
- Create: `web/` directory (via `npx create-next-app@latest`)
- Configure: shadcn/ui, Tailwind, Geist fonts, CSS variables, Plotly

- [ ] **Step 1: Create Next.js app**

```bash
npx create-next-app@latest web --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
```

- [ ] **Step 2: Install dependencies**

```bash
cd web && npm install swr plotly.js react-plotly.js @types/react-plotly.js
npx shadcn@latest init
```

- [ ] **Step 3: Set up CSS variables in `globals.css`**

Add the design token system from the spec (--bg, --surface, --border, --text-primary, --text-secondary, --positive, --negative, --accent).

- [ ] **Step 4: Configure Geist fonts in layout.tsx**

- [ ] **Step 5: Commit**

---

### Task 12: Layout Shell + Sidebar Navigation

**Files:**
- Create: `web/src/components/layout/sidebar.tsx`
- Create: `web/src/components/layout/app-shell.tsx`
- Modify: `web/src/app/layout.tsx`

- [ ] **Step 1: Build sidebar with nav order**: Dashboard > Pricing > Volatility > Strategies > Backtest > Market
- [ ] **Step 2: Implement icon rail collapse at <1024px**
- [ ] **Step 3: Commit**

---

### Task 13: API Client + TypeScript Types

**Files:**
- Create: `web/src/lib/api.ts`
- Create: `web/src/lib/types.ts`
- Create: `web/src/hooks/use-pricing.ts`
- Create: `web/src/hooks/use-market.ts`
- Create: `web/src/hooks/use-volatility.ts`

- [ ] **Step 1: Define TypeScript types matching Pydantic schemas**
- [ ] **Step 2: Create typed fetch wrappers with error handling**
- [ ] **Step 3: Create SWR hooks (stale-while-revalidate behavior)**
- [ ] **Step 4: Commit**

---

### Task 14: Plotly Chart Wrapper

**Files:**
- Create: `web/src/components/charts/base-chart.tsx`
- Create: `web/src/components/charts/sensitivity-chart.tsx`
- Create: `web/src/components/charts/payoff-chart.tsx`
- Create: `web/src/components/charts/heatmap-chart.tsx`
- Create: `web/src/components/charts/surface-chart.tsx`
- Create: `web/src/components/charts/candlestick-chart.tsx`

- [ ] **Step 1: Create base chart component with Plotly config from spec**

Apply: `plotly_dark` template, transparent bgcolor, `--text-secondary` axis text, Geist Sans font, margins `{l:48, r:16, t:32, b:40}`.

- [ ] **Step 2: Create specialized chart components**
- [ ] **Step 3: Commit**

---

### Task 15: Dashboard Page (/)

The visual anchor page. Workspace layout: left rail (inputs), right canvas (price, Greeks, live payoff curve, vol surface preview).

**Files:**
- Modify: `web/src/app/page.tsx`
- Create: `web/src/components/pricing/workspace.tsx`
- Create: `web/src/components/pricing/greeks-row.tsx`
- Create: `web/src/components/pricing/param-rail.tsx`

- [ ] **Step 1: Build param rail (ticker, model selector, S, K, T, r, sigma)**
- [ ] **Step 2: Build Greeks row with proper formatting (spec: delta/rho 4dp, gamma 6dp, sign coloring)**
- [ ] **Step 3: Build live payoff curve (visual anchor)**
- [ ] **Step 4: Add vol surface preview (240px, links to /volatility)**
- [ ] **Step 5: Pre-populate with AAPL defaults**
- [ ] **Step 6: Add cold-start skeleton + "Waking pricing engine..." badge**
- [ ] **Step 7: Commit**

---

### Task 16: Pricing Page (/pricing)

Full sensitivity analysis workspace with advanced params and tabbed charts.

**Files:**
- Create: `web/src/app/pricing/page.tsx`
- Create: `web/src/components/pricing/advanced-params.tsx`
- Create: `web/src/components/pricing/model-param-panel.tsx`

- [ ] **Step 1: Build full param rail with collapsible "Advanced Parameters" section**
- [ ] **Step 2: Add "Price Option" button (manual trigger, no auto-reprice for MC)**
- [ ] **Step 3: Build tabbed sensitivity chart (Delta | Gamma | Vega | Theta)**
- [ ] **Step 4: Add heatmap below**
- [ ] **Step 5: Add MC histogram (conditional on MC model selection)**
- [ ] **Step 6: Commit**

---

### Task 17: Volatility Page (/volatility)

3D vol surface + smile curves + term structure.

**Files:**
- Create: `web/src/app/volatility/page.tsx`

- [ ] **Step 1: Build 3D surface (480px, Plasma, rotation on, scroll-zoom off)**
- [ ] **Step 2: Build smile curves (all expiries as colored lines)**
- [ ] **Step 3: Build ATM term structure chart**
- [ ] **Step 4: Add coverage badge + sparse data warning (<40%)**
- [ ] **Step 5: Commit**

---

### Task 18: Strategies Page (/strategies)

Template picker + custom leg builder + live payoff diagram.

**Files:**
- Create: `web/src/app/strategies/page.tsx`
- Create: `web/src/components/strategies/template-picker.tsx`
- Create: `web/src/components/strategies/leg-builder.tsx`

- [ ] **Step 1: Build template picker as primary entry (straddle, strangle, iron condor, butterfly, collar)**
- [ ] **Step 2: Build custom leg builder (add/remove legs)**
- [ ] **Step 3: Build live payoff diagram with breakeven annotations, max profit/loss labels**
- [ ] **Step 4: Commit**

---

### Task 19: Backtest Page (/backtest)

**Files:**
- Create: `web/src/app/backtest/page.tsx`

- [ ] **Step 1: Build strategy config + date range presets (1M, 3M, 6M, 1Y, 3Y)**
- [ ] **Step 2: Build P&L line chart**
- [ ] **Step 3: Build risk metrics row (Sharpe, Max Drawdown, Win Rate in mono)**
- [ ] **Step 4: Commit**

---

### Task 20: Market Page (/market)

**Files:**
- Create: `web/src/app/market/page.tsx`

- [ ] **Step 1: Build candlestick + volume chart (6M default)**
- [ ] **Step 2: Build options chain table (horizontal scroll on tablet)**
- [ ] **Step 3: Build HV chart with IV overlay**
- [ ] **Step 4: Commit**

---

### Task 21: Frontend Tests

**Files:**
- Create: `web/src/__tests__/api-client.test.ts`
- Create: `web/src/__tests__/components/greeks-row.test.tsx`
- Create: `web/src/__tests__/components/template-picker.test.tsx`

- [ ] **Step 1: Test API client (mock responses, error handling)**
- [ ] **Step 2: Test GreeksRow component (renders 5 Greeks, correct precision, sign coloring)**
- [ ] **Step 3: Test template picker (selecting straddle populates 2 legs)**
- [ ] **Step 4: Add accessibility audit step**

Install `vitest-axe` (or `@axe-core/react`). Add a test that renders the Dashboard page and GreeksRow component and asserts no a11y violations (ARIA labels, contrast, touch targets).

- [ ] **Step 5: Commit**

---

## Phase 4: Cleanup & Deploy

Phase 4 removes legacy code, sets up Docker Compose for local dev, updates CI/CD, and polishes the README.

### Task 22: Delete Legacy Streamlit Code

**Files:**
- Delete: `app.py`
- Delete: `ui_components.py`
- Delete: `analytics.py`

- [ ] **Step 1: Verify no remaining imports from deleted files**

```bash
grep -r "from analytics import\|import analytics\|from ui_components\|from app import" --include="*.py" .
```
Expected: no matches (or only in test files that were already updated)

- [ ] **Step 2: Delete files**
- [ ] **Step 3: Run all tests to confirm nothing breaks**
- [ ] **Step 4: Commit**

---

### Task 23: Move Files into api/ (Replace Symlinks)

Replace the symlinks from Task 5 with actual file moves.

- [ ] **Step 1: Move `models/`, `utils/`, `data/`, `strategies/`, `data_service.py`, `tests/` into `api/`**
- [ ] **Step 2: Update all imports to use `api/` as package root**
- [ ] **Step 3: Update `api/pyproject.toml` pythonpath**
- [ ] **Step 4: Run all tests from `api/`**
- [ ] **Step 5: Commit**

---

### Task 24: Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Create: `api/Dockerfile`
- Create: `web/Dockerfile` (dev mode)

- [ ] **Step 1: Create API Dockerfile (Python 3.12 + uvicorn)**
- [ ] **Step 2: Create docker-compose.yml (api on :8000, web on :3000)**
- [ ] **Step 3: Test `docker-compose up`**
- [ ] **Step 4: Commit**

---

### Task 25: CI/CD Pipeline

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add parallel `api-tests` and `web-tests` jobs**
- [ ] **Step 2: Add sequential `integration` job with Docker Compose**
- [ ] **Step 3: Commit**

---

### Task 26: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README with new architecture, tech stack, screenshots**
- [ ] **Step 2: Remove "GPU-accelerated" claim**
- [ ] **Step 3: Add Quick Start for both `docker-compose up` and manual setup**
- [ ] **Step 4: Commit**

---

## Summary

| Phase | Tasks | Key Deliverable |
|-------|-------|----------------|
| 1. Fix & Stabilize | 1-4 | Passing tests, clean engine.py |
| 2. FastAPI Backend | 5-10 | All API endpoints serving JSON |
| 3. Next.js Frontend | 11-21 | Pricing workstation UI |
| 4. Cleanup & Deploy | 22-26 | Monorepo, Docker, CI/CD, README |
