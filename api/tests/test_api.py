"""API endpoint tests using FastAPI TestClient."""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from api.app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health & Models
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert len(data["models"]) == 5

    def test_models_endpoint(self):
        r = client.get("/api/models")
        assert r.status_code == 200
        models = r.json()
        assert len(models) == 5
        assert all("name" in m and "params" in m for m in models)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_price_bs_call(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "K": 100, "T": 0.5,
            "r": 0.05, "sigma": 0.2, "option_type": "call",
        })
        assert r.status_code == 200
        data = r.json()
        assert all(
            k in data for k in ["price", "delta", "gamma", "vega", "theta", "rho"]
        )
        assert data["price"] > 0
        assert 0 < data["delta"] < 1  # Call delta

    def test_price_bs_put(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "K": 100, "T": 0.5,
            "r": 0.05, "sigma": 0.2, "option_type": "put",
        })
        assert r.status_code == 200
        assert r.json()["delta"] < 0  # Put delta

    def test_price_invalid_option_type(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "K": 100, "T": 0.5,
            "r": 0.05, "sigma": 0.2, "option_type": "straddle",
        })
        assert r.status_code == 422

    def test_price_negative_strike(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "K": -10, "T": 0.5,
            "r": 0.05, "sigma": 0.2,
        })
        assert r.status_code == 422

    def test_price_zero_spot(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 0, "K": 100, "T": 0.5,
            "r": 0.05, "sigma": 0.2,
        })
        assert r.status_code == 422

    def test_price_sigma_too_high(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "K": 100, "T": 0.5,
            "r": 0.05, "sigma": 10.0,  # Max is 5.0
        })
        assert r.status_code == 422

    def test_price_negative_sigma(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "K": 100, "T": 0.5,
            "r": 0.05, "sigma": -0.1,
        })
        assert r.status_code == 422

    def test_price_missing_required_field(self):
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "T": 0.5,
            "r": 0.05, "sigma": 0.2,
        })
        # K is required (gt=0, no default)
        assert r.status_code == 422

    def test_mc_paths_clamped(self):
        """MC params above cap should be clamped, not rejected."""
        r = client.post("/api/price", json={
            "model": "Black-Scholes", "S": 100, "K": 100, "T": 0.5,
            "r": 0.05, "sigma": 0.2, "option_type": "call",
            "model_params": {"mc_paths": 100000, "mc_steps": 1000},
        })
        # Black-Scholes ignores mc_paths but the request should still succeed
        # because clamping happens at schema level, not rejection
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


class TestHeatmap:
    def test_heatmap_valid(self):
        r = client.post("/api/heatmap", json={
            "K": 100, "T": 0.5, "r": 0.05,
            "spot_range": {"min": 80, "max": 120, "steps": 5},
            "vol_range": {"min": 0.1, "max": 0.5, "steps": 5},
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["spot_values"]) == 5
        assert len(data["vol_values"]) == 5
        assert len(data["call_prices"]) == 5
        assert len(data["put_prices"]) == 5

    def test_heatmap_zero_strike(self):
        r = client.post("/api/heatmap", json={
            "K": 0, "T": 0.5, "r": 0.05,
            "spot_range": {"min": 80, "max": 120, "steps": 5},
            "vol_range": {"min": 0.1, "max": 0.5, "steps": 5},
        })
        assert r.status_code == 422

    def test_heatmap_steps_too_many(self):
        r = client.post("/api/heatmap", json={
            "K": 100, "T": 0.5, "r": 0.05,
            "spot_range": {"min": 80, "max": 120, "steps": 200},
            "vol_range": {"min": 0.1, "max": 0.5, "steps": 5},
        })
        assert r.status_code == 422

    def test_heatmap_steps_below_minimum(self):
        r = client.post("/api/heatmap", json={
            "K": 100, "T": 0.5, "r": 0.05,
            "spot_range": {"min": 80, "max": 120, "steps": 1},
            "vol_range": {"min": 0.1, "max": 0.5, "steps": 5},
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_mc_valid(self):
        r = client.post("/api/monte-carlo", json={
            "S": 100, "K": 100, "T": 0.5, "r": 0.05, "sigma": 0.2,
            "paths": 500, "option_type": "call",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["price"] > 0
        assert data["std_error"] >= 0
        assert len(data["confidence_interval"]) == 2
        assert len(data["terminal_prices"]) <= 1000  # Capped

    def test_mc_put(self):
        r = client.post("/api/monte-carlo", json={
            "S": 100, "K": 100, "T": 0.5, "r": 0.05, "sigma": 0.2,
            "paths": 500, "option_type": "put",
        })
        assert r.status_code == 200
        assert r.json()["price"] > 0

    def test_mc_paths_below_minimum(self):
        r = client.post("/api/monte-carlo", json={
            "S": 100, "K": 100, "T": 0.5, "r": 0.05, "sigma": 0.2,
            "paths": 50,  # Below minimum of 100
        })
        assert r.status_code == 422

    def test_mc_paths_above_maximum(self):
        r = client.post("/api/monte-carlo", json={
            "S": 100, "K": 100, "T": 0.5, "r": 0.05, "sigma": 0.2,
            "paths": 60000,  # Above maximum of 50000
        })
        assert r.status_code == 422

    def test_mc_invalid_option_type(self):
        r = client.post("/api/monte-carlo", json={
            "S": 100, "K": 100, "T": 0.5, "r": 0.05, "sigma": 0.2,
            "paths": 500, "option_type": "straddle",
        })
        assert r.status_code == 422

    def test_mc_negative_spot(self):
        r = client.post("/api/monte-carlo", json={
            "S": -100, "K": 100, "T": 0.5, "r": 0.05, "sigma": 0.2,
            "paths": 500,
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Strategy Payoff
# ---------------------------------------------------------------------------


class TestPayoff:
    def test_payoff_single_call(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [{"type": "call", "strike": 100, "qty": 1, "side": "long"}],
            "spot_range": {"min": 80, "max": 120},
            "S": 100, "T": 0.0833, "r": 0.05, "sigma": 0.2,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["prices"]) == 200  # Default steps
        assert len(data["pnl"]) == 200
        assert len(data["breakevens"]) >= 1

    def test_payoff_single_put(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [{"type": "put", "strike": 100, "qty": 1, "side": "long"}],
            "spot_range": {"min": 80, "max": 120},
            "S": 100, "T": 0.0833, "r": 0.05, "sigma": 0.2,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["prices"]) == 200
        assert len(data["breakevens"]) >= 1

    def test_payoff_straddle(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [
                {"type": "call", "strike": 100, "qty": 1, "side": "long"},
                {"type": "put", "strike": 100, "qty": 1, "side": "long"},
            ],
            "spot_range": {"min": 70, "max": 130},
            "S": 100, "T": 0.0833, "r": 0.05, "sigma": 0.2,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["breakevens"]) == 2  # Straddle has 2 breakevens

    def test_payoff_short_leg(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [{"type": "call", "strike": 100, "qty": 1, "side": "short"}],
            "spot_range": {"min": 80, "max": 120},
            "S": 100, "T": 0.0833, "r": 0.05, "sigma": 0.2,
        })
        assert r.status_code == 200
        data = r.json()
        # Short call: pnl at high spot should be negative
        assert data["pnl"][-1] < 0

    def test_payoff_empty_legs(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [],
            "spot_range": {"min": 80, "max": 120},
            "S": 100,
        })
        assert r.status_code == 422

    def test_payoff_invalid_side(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [{"type": "call", "strike": 100, "qty": 1, "side": "buy"}],
            "spot_range": {"min": 80, "max": 120},
            "S": 100,
        })
        assert r.status_code == 422

    def test_payoff_invalid_type(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [{"type": "forward", "strike": 100, "qty": 1, "side": "long"}],
            "spot_range": {"min": 80, "max": 120},
            "S": 100,
        })
        assert r.status_code == 422

    def test_payoff_zero_strike(self):
        r = client.post("/api/strategy/payoff", json={
            "legs": [{"type": "call", "strike": 0, "qty": 1, "side": "long"}],
            "spot_range": {"min": 80, "max": 120},
            "S": 100,
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Volatility Surface (only validation - real data requires network)
# ---------------------------------------------------------------------------


class TestVolatilitySurface:
    def test_vol_surface_missing_ticker(self):
        r = client.post("/api/volatility-surface", json={})
        assert r.status_code == 422

    def test_vol_surface_invalid_ticker(self):
        """Invalid ticker should return empty surface, not crash."""
        r = client.post("/api/volatility-surface", json={
            "ticker": "ZZZZZZZZZ",
        })
        # Should either return 200 with empty surface or 502
        assert r.status_code in [200, 502]
        if r.status_code == 200:
            data = r.json()
            assert "surface" in data
            assert "coverage" in data


# ---------------------------------------------------------------------------
# Market Route Validation (no real network calls)
# ---------------------------------------------------------------------------


class TestMarketRouteValidation:
    def test_ticker_too_long(self):
        r = client.get("/api/market/ABCDEFGHIJK")  # > 10 chars
        assert r.status_code == 422

    def test_ticker_invalid_chars(self):
        r = client.get("/api/market/../../etc")
        assert r.status_code in [404, 422]  # Path traversal blocked

    def test_chain_ticker_too_long(self):
        r = client.get("/api/chain/ABCDEFGHIJK")  # > 10 chars
        assert r.status_code == 422

    def test_chain_ticker_invalid_chars(self):
        r = client.get("/api/chain/AB@CD")
        assert r.status_code in [404, 422]


# ---------------------------------------------------------------------------
# Backtest Validation (no real network calls)
# ---------------------------------------------------------------------------


class TestBacktestValidation:
    def test_backtest_empty_legs(self):
        r = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "legs": [],
        })
        assert r.status_code == 422

    def test_backtest_invalid_side(self):
        r = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "legs": [{"type": "call", "strike": 150, "expiry": "2025-06-20",
                       "qty": 1, "side": "buy"}],
        })
        assert r.status_code == 422

    def test_backtest_invalid_expiry_format(self):
        r = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "legs": [{"type": "call", "strike": 150, "expiry": "not-a-date",
                       "qty": 1, "side": "long"}],
        })
        assert r.status_code == 422

    def test_backtest_invalid_option_type(self):
        r = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "legs": [{"type": "strangle", "strike": 150, "expiry": "2025-06-20",
                       "qty": 1, "side": "long"}],
        })
        assert r.status_code == 422

    def test_backtest_zero_strike(self):
        r = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "legs": [{"type": "call", "strike": 0, "expiry": "2025-06-20",
                       "qty": 1, "side": "long"}],
        })
        assert r.status_code == 422
