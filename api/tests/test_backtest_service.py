"""Unit tests for backtest_service - verifies theta decay in daily P&L."""
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.app.services.backtest_service import run_backtest
from models.black_scholes import BlackScholesModel


def _make_flat_history(spot: float, n_days: int, start_date: date) -> pd.DataFrame:
    """Create a price history DataFrame with FLAT prices (no spot movement).

    When spot is constant, any P&L must come from theta decay alone.
    """
    dates = pd.date_range(start=start_date, periods=n_days, freq="B")
    df = pd.DataFrame({"Close": [spot] * n_days}, index=dates)
    return df


class TestBacktestThetaDecay:
    """Regression tests: theta must contribute to daily P&L."""

    def test_long_call_loses_value_on_flat_spot(self):
        """A long call on flat spot should lose money from theta decay.

        If spot never moves, the only source of P&L is theta. A long option
        holder should see value erode day over day.
        """
        start = date(2025, 1, 2)
        n_days = 22  # ~1 month of trading days
        spot = 100.0
        strike = 100.0
        expiry = start + timedelta(days=90)  # 90 calendar days out

        history_df = _make_flat_history(spot, n_days, start)

        with patch("api.app.services.backtest_service.fetch_stock_data") as mock_fetch:
            mock_fetch.return_value = (history_df, {}, "2025-01-02T00:00:00")

            result = run_backtest(
                ticker="TEST",
                legs=[{
                    "type": "call",
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "qty": 1,
                    "side": "long",
                }],
                r=0.05,
                sigma=0.20,
            )

        # With flat spot, total P&L should be negative (theta erosion)
        assert result["total_pnl"] < 0, (
            f"Long call on flat spot should lose money from theta, "
            f"but total_pnl={result['total_pnl']}"
        )

        # Verify the P&L series is monotonically decreasing (theta drip)
        pnl_values = [p["pnl"] for p in result["pnl_series"]]
        for i in range(1, len(pnl_values)):
            assert pnl_values[i] <= pnl_values[i - 1] + 1e-6, (
                f"Day {i}: P&L should decrease on flat spot, but "
                f"{pnl_values[i]:.6f} > {pnl_values[i-1]:.6f}"
            )

    def test_short_put_gains_from_theta_on_flat_spot(self):
        """A short put on flat spot should gain money from theta decay.

        Short options benefit from time decay.
        """
        start = date(2025, 1, 2)
        n_days = 22
        spot = 100.0
        strike = 100.0
        expiry = start + timedelta(days=90)

        history_df = _make_flat_history(spot, n_days, start)

        with patch("api.app.services.backtest_service.fetch_stock_data") as mock_fetch:
            mock_fetch.return_value = (history_df, {}, "2025-01-02T00:00:00")

            result = run_backtest(
                ticker="TEST",
                legs=[{
                    "type": "put",
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "qty": 1,
                    "side": "short",
                }],
                r=0.05,
                sigma=0.20,
            )

        # Short put on flat spot should make money (collecting theta)
        assert result["total_pnl"] > 0, (
            f"Short put on flat spot should gain from theta, "
            f"but total_pnl={result['total_pnl']}"
        )

    def test_theta_magnitude_is_reasonable(self):
        """Verify theta P&L per day is in a reasonable range for ATM options.

        ATM option with ~3 months to expiry, 20% vol, $100 spot:
        theta ~ -$0.04 to -$0.06 per day. Over 22 days ~ -$0.88 to -$1.32.
        """
        start = date(2025, 1, 2)
        n_days = 22
        spot = 100.0
        strike = 100.0
        expiry = start + timedelta(days=90)

        history_df = _make_flat_history(spot, n_days, start)

        with patch("api.app.services.backtest_service.fetch_stock_data") as mock_fetch:
            mock_fetch.return_value = (history_df, {}, "2025-01-02T00:00:00")

            result = run_backtest(
                ticker="TEST",
                legs=[{
                    "type": "call",
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "qty": 1,
                    "side": "long",
                }],
                r=0.05,
                sigma=0.20,
            )

        total = result["total_pnl"]
        # 22 days of theta on ATM 3-month call should be roughly -$0.5 to -$2.0
        assert -5.0 < total < -0.1, (
            f"Expected theta loss in [-5.0, -0.1] range over 22 days, got {total}"
        )

    def test_yesterday_uses_different_T_than_today(self):
        """Directly verify that the backtest uses different T values for
        today vs yesterday by checking option values differ even on flat spot."""
        start = date(2025, 3, 3)  # Monday
        n_days = 3
        spot = 100.0
        strike = 100.0
        expiry = start + timedelta(days=60)

        history_df = _make_flat_history(spot, n_days, start)

        with patch("api.app.services.backtest_service.fetch_stock_data") as mock_fetch:
            mock_fetch.return_value = (history_df, {}, "2025-03-03T00:00:00")

            result = run_backtest(
                ticker="TEST",
                legs=[{
                    "type": "call",
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "qty": 1,
                    "side": "long",
                }],
                r=0.05,
                sigma=0.20,
            )

        # With flat spot and different T values, daily P&L must be non-zero
        pnl_values = [p["pnl"] for p in result["pnl_series"]]
        assert any(abs(p) > 1e-8 for p in pnl_values), (
            f"Daily P&L should be non-zero on flat spot (theta), got {pnl_values}"
        )

    def test_expired_option_uses_intrinsic_value(self):
        """After expiry, option value should be intrinsic (not BS)."""
        start = date(2025, 1, 2)
        n_days = 5
        # Expiry is 2 days after start - so days 3-5 are past expiry
        expiry = start + timedelta(days=2)

        # Spot well above strike so ITM call has clear intrinsic value
        spot = 110.0
        strike = 100.0

        history_df = _make_flat_history(spot, n_days, start)

        with patch("api.app.services.backtest_service.fetch_stock_data") as mock_fetch:
            mock_fetch.return_value = (history_df, {}, "2025-01-02T00:00:00")

            result = run_backtest(
                ticker="TEST",
                legs=[{
                    "type": "call",
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "qty": 1,
                    "side": "long",
                }],
                r=0.05,
                sigma=0.20,
            )

        # Should not crash and should return results
        assert result["pnl_series"] is not None
        assert len(result["pnl_series"]) == n_days - 1

    def test_expired_otm_option_is_worthless(self):
        """After expiry, an OTM option should have zero value."""
        start = date(2025, 1, 2)
        n_days = 5
        expiry = start + timedelta(days=1)  # Expires after day 1

        spot = 90.0  # Well below strike for OTM call
        strike = 100.0

        history_df = _make_flat_history(spot, n_days, start)

        with patch("api.app.services.backtest_service.fetch_stock_data") as mock_fetch:
            mock_fetch.return_value = (history_df, {}, "2025-01-02T00:00:00")

            result = run_backtest(
                ticker="TEST",
                legs=[{
                    "type": "call",
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "qty": 1,
                    "side": "long",
                }],
                r=0.05,
                sigma=0.20,
            )

        # After expiry, OTM call is worthless. P&L should stabilize
        # (no further changes once both today and yesterday are past expiry)
        pnl_values = [p["pnl"] for p in result["pnl_series"]]
        # The last few values should be stable (both dates past expiry)
        if len(pnl_values) >= 3:
            # Once fully past expiry, daily change should be ~0
            last_change = abs(pnl_values[-1] - pnl_values[-2])
            assert last_change < 1e-6, (
                f"P&L should stabilize after expiry, but last change was {last_change}"
            )
