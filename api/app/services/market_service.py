"""Market data service - wraps data_service.py with error handling."""
from datetime import datetime, UTC

import math

import numpy as np
from fastapi import HTTPException

from data_service import fetch_stock_data, fetch_options_chain
from models.engine import calculate_historical_volatility


def _safe_float(val, default: float = 0.0) -> float:
    """Convert to float, returning default for NaN/None/invalid."""
    try:
        result = float(val)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def get_market_data(ticker: str) -> dict:
    """Fetch current price, history, and historical volatility for a ticker."""
    try:
        history_df, info, fetched_at = fetch_stock_data(ticker)
    except Exception as e:
        error_msg = str(e).lower()
        if "no data" in error_msg or "not found" in error_msg:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
        raise HTTPException(status_code=502, detail="Market data temporarily unavailable")

    if history_df is None or history_df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for '{ticker}'")

    # Extract current price from last close
    price = float(history_df["Close"].iloc[-1])
    if price <= 0:
        raise HTTPException(status_code=404, detail=f"No price data for '{ticker}'")

    # Compute historical volatility
    hv = calculate_historical_volatility(history_df["Close"])
    if np.isnan(hv):
        hv = 0.0

    # Format history rows
    history = []
    for idx, row in history_df.iterrows():
        history.append({
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "open": _safe_float(row.get("Open")),
            "high": _safe_float(row.get("High")),
            "low": _safe_float(row.get("Low")),
            "close": _safe_float(row.get("Close")),
            "volume": _safe_float(row.get("Volume")) if row.get("Volume") is not None else None,
        })

    return {
        "price": price,
        "history": history,
        "historical_vol": float(hv),
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def get_options_chain(ticker: str) -> dict:
    """Fetch options chain for a ticker."""
    try:
        chain_df, expirations, _fetched_at = fetch_options_chain(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail="Options chain data temporarily unavailable")

    if chain_df is None or chain_df.empty:
        return {"calls": [], "puts": [], "expirations": expirations or []}

    calls = []
    puts = []

    for _, row in chain_df.iterrows():
        entry = {
            "strike": _safe_float(row.get("strike")),
            "lastPrice": _safe_float(row.get("lastPrice")),
            "iv": _safe_float(row.get("impliedVolatility")) or None,
            "volume": _safe_float(row.get("volume")) or None,
            "oi": _safe_float(row.get("openInterest")) or None,
        }
        option_type = str(row.get("type", "")).lower()
        if option_type == "call":
            calls.append(entry)
        elif option_type == "put":
            puts.append(entry)

    return {
        "calls": calls,
        "puts": puts,
        "expirations": expirations or [],
    }
