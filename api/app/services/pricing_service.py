"""Pricing service - thin wrapper around models/engine.py."""
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
