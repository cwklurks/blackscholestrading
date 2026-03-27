"""Comprehensive tests for models/engine.py orchestration layer."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.engine import (  # noqa: E402
    atm_iv_from_chain,
    calculate_historical_volatility,
    implied_volatility,
    iv_hv_stats,
    numerical_greeks,
    option_metrics,
    price_with_model,
    risk_reversal_and_fly,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# Standard option parameters used across many tests
S, K, T, R, Q, SIGMA, BORROW = 100.0, 100.0, 0.5, 0.05, 0.0, 0.25, 0.0

# Low path/step counts for MC models to keep tests fast
MC_PARAMS = {"mc_paths": 1000, "mc_steps": 50}


def _make_chain() -> pd.DataFrame:
    """Build a mock options DataFrame with 10 rows (5 calls, 5 puts)."""
    strikes = [90, 95, 100, 105, 110]
    expiry = "2024-06-21"
    rows = []
    # IVs have a slight smile: higher at wings, lower ATM
    ivs = [0.28, 0.24, 0.22, 0.24, 0.27]
    for strike, iv in zip(strikes, ivs):
        rows.append(
            {
                "strike": float(strike),
                "expiration": expiry,
                "type": "call",
                "impliedVolatility": iv,
            }
        )
        rows.append(
            {
                "strike": float(strike),
                "expiration": expiry,
                "type": "put",
                "impliedVolatility": iv + 0.01,  # Puts slightly higher IV
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestPriceWithModel - routing tests
# ---------------------------------------------------------------------------


class TestPriceWithModel:
    """Verify price_with_model routes to the correct pricing model."""

    @pytest.mark.parametrize(
        "model_name,extra_params",
        [
            ("Black-Scholes", {}),
            ("Binomial (American)", {"binomial_steps": 50}),
            ("Heston MC", {**MC_PARAMS, "heston_kappa": 1.5, "heston_theta": 0.04, "heston_rho": -0.7, "heston_vol_of_vol": 0.3}),
            ("GARCH MC", {**MC_PARAMS, "garch_alpha0": 2e-6, "garch_alpha1": 0.08, "garch_beta1": 0.9}),
            ("Bates Jump-Diffusion", {**MC_PARAMS, "jump_lambda": 0.1, "jump_mu": -0.05, "jump_delta": 0.2}),
        ],
        ids=["Black-Scholes", "Binomial (American)", "Heston MC", "GARCH MC", "Bates Jump-Diffusion"],
    )
    def test_routes_to_correct_model(self, model_name, extra_params):
        price = price_with_model(
            model_name, S, K, T, R, Q, SIGMA, BORROW, "call", extra_params
        )
        assert math.isfinite(price), f"{model_name} returned non-finite price: {price}"
        assert price > 0, f"{model_name} returned non-positive price: {price}"

    def test_unknown_model_falls_back_to_bs(self):
        price = price_with_model(
            "NonexistentModel", S, K, T, R, Q, SIGMA, BORROW, "call", {}
        )
        assert math.isfinite(price), "Fallback BS returned non-finite price"
        assert price > 0, "Fallback BS returned non-positive price"


# ---------------------------------------------------------------------------
# TestOptionMetrics - unified dispatcher
# ---------------------------------------------------------------------------


class TestOptionMetrics:
    """Verify option_metrics returns correct Greek dictionaries."""

    EXPECTED_KEYS = {"price", "delta", "gamma", "vega", "theta", "rho"}

    def test_bs_returns_analytical_greeks(self):
        result = option_metrics("Black-Scholes", S, K, T, R, Q, SIGMA, BORROW, {}, option_type="call")
        assert set(result.keys()) == self.EXPECTED_KEYS
        assert 0 < result["delta"] < 1, f"Call delta out of range: {result['delta']}"
        assert result["gamma"] > 0, f"Gamma should be positive: {result['gamma']}"

    def test_non_bs_returns_numerical_greeks(self):
        params = {**MC_PARAMS, "heston_kappa": 1.5, "heston_theta": 0.04, "heston_rho": -0.7, "heston_vol_of_vol": 0.3}
        result = option_metrics("Heston MC", S, K, T, R, Q, SIGMA, BORROW, params, option_type="call")
        assert set(result.keys()) == self.EXPECTED_KEYS
        for key, val in result.items():
            assert math.isfinite(val), f"Heston Greek '{key}' is not finite: {val}"

    def test_put_option_negative_delta(self):
        result = option_metrics("Black-Scholes", S, K, T, R, Q, SIGMA, BORROW, {}, option_type="put")
        assert result["delta"] < 0, f"Put delta should be negative: {result['delta']}"


# ---------------------------------------------------------------------------
# TestNumericalGreeks - edge cases
# ---------------------------------------------------------------------------


class TestNumericalGreeks:
    """Verify numerical Greeks remain stable at boundary conditions."""

    def test_near_zero_time(self):
        greeks = numerical_greeks(
            "Black-Scholes", S, K, 0.001, R, Q, SIGMA, BORROW, "call", {}
        )
        for key, val in greeks.items():
            assert math.isfinite(val), f"Near-zero T: Greek '{key}' is not finite: {val}"

    def test_near_zero_sigma(self):
        greeks = numerical_greeks(
            "Black-Scholes", S, K, T, R, Q, 0.01, BORROW, "call", {}
        )
        for key, val in greeks.items():
            assert math.isfinite(val), f"Near-zero sigma: Greek '{key}' is not finite: {val}"


# ---------------------------------------------------------------------------
# TestImpliedVolatility - solver
# ---------------------------------------------------------------------------


class TestImpliedVolatility:
    """Verify implied-volatility solver round-trips correctly."""

    def test_recovers_known_vol(self):
        from models.black_scholes import BlackScholesModel

        target_sigma = 0.3
        model = BlackScholesModel(S, K, T, R, target_sigma)
        price = model.call_price()
        recovered = implied_volatility(price, S, K, T, R, option_type="call")
        assert abs(recovered - target_sigma) < 0.01, (
            f"Expected ~{target_sigma}, got {recovered}"
        )

    def test_put_iv(self):
        from models.black_scholes import BlackScholesModel

        target_sigma = 0.25
        model = BlackScholesModel(S, K, T, R, target_sigma)
        price = model.put_price()
        recovered = implied_volatility(price, S, K, T, R, option_type="put")
        assert abs(recovered - target_sigma) < 0.01, (
            f"Expected ~{target_sigma}, got {recovered}"
        )

    def test_zero_price_returns_none(self):
        """A zero option price cannot be matched by any vol - solver should report failure."""
        recovered = implied_volatility(0.0, S, K, T, R, option_type="call")
        assert recovered is None, f"IV for zero price should be None, got {recovered}"

    def test_unreachable_price_returns_none(self):
        """Price above spot is unreachable for any vol - solver should report failure."""
        recovered = implied_volatility(150.0, S=100, K=100, T=0.5, r=0.05, option_type="call")
        assert recovered is None, f"IV for unreachable price should be None, got {recovered}"

    def test_reasonable_option_returns_valid_float(self):
        """A sensible ATM option with a real BS price should round-trip to a valid IV."""
        from models.black_scholes import BlackScholesModel

        target_sigma = 0.20
        model = BlackScholesModel(S, K, T, R, target_sigma)
        price = model.call_price()
        recovered = implied_volatility(price, S, K, T, R, option_type="call")
        assert recovered is not None, "IV should not be None for a reasonable option"
        assert isinstance(recovered, float), f"IV should be a float, got {type(recovered)}"
        assert abs(recovered - target_sigma) < 0.01, (
            f"Expected ~{target_sigma}, got {recovered}"
        )


# ---------------------------------------------------------------------------
# TestHistoricalVolatility
# ---------------------------------------------------------------------------


class TestHistoricalVolatility:
    """Verify historical volatility calculation."""

    def test_returns_finite_for_valid_series(self):
        np.random.seed(42)
        prices = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.01, 60)))
        hv = calculate_historical_volatility(prices)
        assert math.isfinite(hv), f"HV should be finite, got {hv}"
        assert hv > 0, f"HV should be positive, got {hv}"

    def test_returns_nan_for_empty_series(self):
        hv = calculate_historical_volatility(pd.Series(dtype=float))
        assert math.isnan(hv), f"HV for empty series should be NaN, got {hv}"

    def test_single_price_returns_nan(self):
        hv = calculate_historical_volatility(pd.Series([100.0]))
        assert math.isnan(hv), f"HV for single price should be NaN, got {hv}"


# ---------------------------------------------------------------------------
# TestChainAnalytics
# ---------------------------------------------------------------------------


class TestChainAnalytics:
    """Verify chain analytics functions (ATM IV, risk reversal, butterfly)."""

    def test_atm_iv_finds_nearest_strike(self):
        chain = _make_chain()
        iv = atm_iv_from_chain(chain, "2024-06-21", spot=101.0)
        assert iv is not None, "ATM IV should not be None for valid chain"
        assert abs(iv - 0.22) < 0.02, f"Expected ATM IV ~0.22, got {iv}"

    def test_atm_iv_empty_chain(self):
        chain = _make_chain()
        iv = atm_iv_from_chain(chain, "2099-12-31", spot=100.0)
        assert iv is None, "ATM IV should be None for non-matching expiry"

    def test_risk_reversal_and_fly(self):
        chain = _make_chain()
        result = risk_reversal_and_fly(chain, spot=100.0, expiry="2024-06-21")
        assert result is not None, "risk_reversal_and_fly returned None for valid chain"
        assert "risk_reversal" in result
        assert "butterfly" in result
        for key, val in result.items():
            assert math.isfinite(val), f"'{key}' is not finite: {val}"

    def test_risk_reversal_empty_chain(self):
        chain = _make_chain()
        result = risk_reversal_and_fly(chain, spot=100.0, expiry="2099-12-31")
        assert result is None, "Should return None for non-matching expiry"

    def test_iv_hv_stats_valid(self):
        np.random.seed(42)
        prices = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0, 0.01, 60)))
        hv_series = prices.pct_change().rolling(window=5).std() * np.sqrt(252)
        result = iv_hv_stats(0.25, hv_series)
        assert result is not None, "iv_hv_stats should not be None for valid data"
        assert math.isfinite(result["edge"]), f"Edge should be finite: {result['edge']}"

    def test_iv_hv_stats_empty_series(self):
        result = iv_hv_stats(0.25, pd.Series(dtype=float))
        assert result is None, "iv_hv_stats should return None for empty series"
