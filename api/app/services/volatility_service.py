"""Volatility surface service - IV computation across strike/expiry grid."""
from datetime import date
from typing import Optional

from data_service import fetch_options_chain
from models.engine import implied_volatility


def _years_to_expiry(expiry_str: str) -> float:
    """Convert expiration date string to time-to-expiry in years."""
    try:
        days = (date.fromisoformat(expiry_str) - date.today()).days
        return max(days, 1) / 365.0
    except (ValueError, TypeError):
        return 30 / 365.0  # Fallback: ~1 month


def compute_volatility_surface(
    ticker: str,
    strikes: Optional[list[float]] = None,
    expirations: Optional[list[str]] = None,
) -> dict:
    """Compute IV surface from live options chain data.

    Returns a dict with surface points, smile data per expiry, and coverage ratio.
    Gracefully handles missing data by returning None for IV where computation fails.
    """
    chain_df, available_exps, _fetched_at = fetch_options_chain(ticker)

    if chain_df is None or chain_df.empty:
        return {"surface": [], "smile_data": {}, "coverage": 0.0}

    exps = expirations or (available_exps[:5] if available_exps else [])

    # Estimate spot price as median strike (rough proxy when spot isn't available)
    spot = float(chain_df["strike"].median())

    surface = []
    smile_data: dict[str, list[dict]] = {}
    total_cells = 0
    valid_cells = 0

    for exp in exps:
        subset = chain_df[chain_df["expiration"] == exp]
        if subset.empty:
            continue

        exp_strikes = strikes or sorted(subset["strike"].unique().tolist())
        T = _years_to_expiry(exp)
        smile_points: list[dict] = []

        for strike in exp_strikes:
            total_cells += 1
            row = subset[(subset["strike"] == strike) & (subset["type"] == "Call")]
            iv = None

            if not row.empty:
                market_price = row.iloc[0].get("lastPrice", 0)
                if market_price and market_price > 0:
                    try:
                        iv = implied_volatility(market_price, spot, strike, T, 0.05)
                        valid_cells += 1
                    except Exception:
                        iv = None

            surface.append({"strike": strike, "expiry": exp, "iv": iv})
            smile_points.append({"strike": strike, "iv": iv})

        smile_data[exp] = smile_points

    coverage = valid_cells / total_cells if total_cells > 0 else 0.0
    return {
        "surface": surface,
        "smile_data": smile_data,
        "coverage": round(coverage, 3),
    }
