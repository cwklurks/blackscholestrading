"""Backtest service - strategy payoff and historical backtesting."""
from datetime import date

import numpy as np
from fastapi import HTTPException

from strategies.multi_leg import MultiLegStrategy
from data_service import fetch_stock_data
from models.black_scholes import BlackScholesModel


def _option_value(
    spot: float, strike: float, T: float, r: float, sigma: float,
    is_call: bool,
) -> float:
    """Compute option value, falling back to intrinsic value when expired.

    Args:
        spot: Current spot price.
        strike: Option strike price.
        T: Time to expiry in years (can be <= 0 if expired).
        r: Risk-free rate.
        sigma: Volatility.
        is_call: True for call, False for put.

    Returns:
        Option value (BS price if T > 0, intrinsic value if expired).
    """
    if T <= 0:
        # Option has expired - use intrinsic value
        if is_call:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    bs = BlackScholesModel(spot, strike, T, r, sigma)
    if is_call:
        return bs.call_price()
    return bs.put_price()


def compute_payoff(legs: list[dict], spot_range: dict, S: float,
                   T: float, r: float, sigma: float) -> dict:
    """Compute multi-leg strategy payoff diagram."""
    strategy = MultiLegStrategy(legs, S=S, T=T, r=r, sigma=sigma)
    return strategy.compute_payoff(spot_range["min"], spot_range["max"])


def run_backtest(ticker: str, legs: list[dict], r: float, sigma: float) -> dict:
    """Run a simple historical backtest for a strategy."""
    try:
        history_df, info, _fetched_at = fetch_stock_data(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail="Market data temporarily unavailable")

    if history_df is None or history_df.empty:
        raise HTTPException(status_code=404, detail=f"No data for '{ticker}'")

    closes = history_df["Close"].values
    dates = [str(idx.date()) if hasattr(idx, "date") else str(idx)
             for idx in history_df.index]

    # Compute daily P&L based on strategy value changes
    pnl_series = []
    cumulative_pnl = 0.0

    for i in range(1, len(closes)):
        day_pnl = 0.0
        for leg in legs:
            strike = leg["strike"]
            qty = leg.get("qty", 1)
            side_mult = 1 if leg["side"] == "long" else -1

            # Compute T_remaining for today and yesterday independently
            expiry_str = leg.get("expiry")
            if expiry_str:
                expiry_date = date.fromisoformat(expiry_str)
                current_date = date.fromisoformat(dates[i])
                yesterday_date = date.fromisoformat(dates[i - 1])
                T_remaining = (expiry_date - current_date).days / 365
                T_remaining_yesterday = (expiry_date - yesterday_date).days / 365
            else:
                T_remaining = leg.get("T_remaining", 0.0833)
                T_remaining_yesterday = T_remaining

            # Value today vs yesterday (use intrinsic if expired)
            is_call = leg["type"] == "call"
            val_today = _option_value(
                closes[i], strike, T_remaining, r, sigma, is_call,
            )
            val_yesterday = _option_value(
                closes[i - 1], strike, T_remaining_yesterday, r, sigma, is_call,
            )

            day_pnl += side_mult * qty * (val_today - val_yesterday)

        cumulative_pnl += day_pnl
        pnl_series.append({"date": dates[i], "pnl": round(cumulative_pnl, 4)})

    if not pnl_series:
        return {
            "pnl_series": [],
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": None,
            "win_rate": 0.0,
        }

    # Compute metrics
    total_pnl = pnl_series[-1]["pnl"]
    pnl_values = [p["pnl"] for p in pnl_series]
    daily_returns = np.diff([0.0] + pnl_values)

    # Max drawdown (peak starts at 0.0 = entry value, not first day's P&L)
    peak = 0.0
    max_dd = 0.0
    for val in pnl_values:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualized)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    else:
        sharpe = None

    # Win rate
    positive_days = sum(1 for ret in daily_returns if ret > 0)
    win_rate = positive_days / len(daily_returns) if daily_returns.size > 0 else 0.0

    return {
        "pnl_series": pnl_series,
        "total_pnl": round(total_pnl, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "win_rate": round(win_rate, 4),
    }
