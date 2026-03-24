"""Market data service - wraps data_service.py with error handling."""
from datetime import datetime, UTC

import numpy as np
from fastapi import HTTPException

from data_service import fetch_stock_data, fetch_options_chain
from models.engine import calculate_historical_volatility


def get_market_data(ticker: str) -> dict:
    """Fetch current price, history, and historical volatility for a ticker."""
    try:
        history_df, info, fetched_at = fetch_stock_data(ticker)
    except Exception as e:
        error_msg = str(e).lower()
        if "no data" in error_msg or "not found" in error_msg:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {e}")

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
            "open": float(row.get("Open", 0)),
            "high": float(row.get("High", 0)),
            "low": float(row.get("Low", 0)),
            "close": float(row.get("Close", 0)),
            "volume": float(row.get("Volume", 0)) if row.get("Volume") is not None else None,
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
        raise HTTPException(status_code=502, detail=f"Options chain fetch failed: {e}")

    if chain_df is None or chain_df.empty:
        return {"calls": [], "puts": [], "expirations": expirations or []}

    calls = []
    puts = []

    for _, row in chain_df.iterrows():
        entry = {
            "strike": float(row.get("strike", 0)),
            "lastPrice": float(row.get("lastPrice", 0)),
            "iv": float(row["impliedVolatility"]) if row.get("impliedVolatility") is not None else None,
            "volume": float(row["volume"]) if row.get("volume") is not None else None,
            "oi": float(row["openInterest"]) if row.get("openInterest") is not None else None,
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
