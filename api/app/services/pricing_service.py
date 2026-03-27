"""Pricing service - thin wrapper around models/engine.py."""
import numpy as np

from models.engine import option_metrics
from models.black_scholes import bs_call_price_vectorized
from models.simulation import monte_carlo_option_price

MAX_TERMINAL_PRICES = 1000


def compute_price(model_name: str, **kwargs) -> dict:
    """Compute option price + Greeks."""
    metrics = option_metrics(model_name=model_name, **kwargs)
    return {"model": model_name, **metrics}


def compute_heatmap(K: float, T: float, r: float, q: float, borrow_cost: float,
                    spot_range: dict, vol_range: dict) -> dict:
    """Compute BS call/put prices across a Spot x Vol grid."""
    spot_values = np.linspace(spot_range["min"], spot_range["max"], spot_range.get("steps", 20))
    vol_values = np.linspace(vol_range["min"], vol_range["max"], vol_range.get("steps", 20))
    spot_grid, vol_grid = np.meshgrid(spot_values, vol_values)

    call_prices = bs_call_price_vectorized(spot_grid, K, T, r, vol_grid, q, borrow_cost)
    # Put via put-call parity: P = C - S*e^(-(q+b)*T) + K*e^(-r*T)
    carry_discount = np.exp(-(q + borrow_cost) * T)
    put_prices = call_prices - spot_grid * carry_discount + K * np.exp(-r * T)

    return {
        "spot_values": spot_values.tolist(),
        "vol_values": vol_values.tolist(),
        "call_prices": call_prices.tolist(),
        "put_prices": put_prices.tolist(),
    }


def compute_monte_carlo(S: float, K: float, T: float, r: float, sigma: float,
                        paths: int, option_type: str, q: float, borrow_cost: float) -> dict:
    """Run Monte Carlo simulation and return price, SE, terminal prices, CI."""
    price, se, terminal = monte_carlo_option_price(
        S=S, K=K, T=T, r=r, sigma=sigma,
        num_simulations=paths, option_type=option_type,
        q=q, borrow_cost=borrow_cost,
    )
    # Cap terminal prices to avoid unbounded payloads
    sample = terminal[:MAX_TERMINAL_PRICES] if len(terminal) > MAX_TERMINAL_PRICES else terminal
    # 95% normal CI
    ci = [price - 1.96 * se, price + 1.96 * se]
    return {
        "price": float(price),
        "std_error": float(se),
        "terminal_prices": sample.tolist(),
        "confidence_interval": ci,
    }
