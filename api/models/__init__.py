"""
Models package for Black-Scholes Trader.

This package contains option pricing models organized by methodology:
- black_scholes: Closed-form analytical solutions (Black-Scholes-Merton)
- numerical: Tree-based methods (Binomial/Trinomial)
- simulation: Monte Carlo engines (Heston, GARCH, Bates jump-diffusion)
- engine: Orchestration layer (routing, Greeks, IV solving, chain analytics)
"""

from models.black_scholes import BlackScholesModel, bs_call_price_vectorized
from models.numerical import binomial_american_option
from models.simulation import (
    heston_mc_price,
    garch_mc_price,
    bates_jump_diffusion_mc_price,
    monte_carlo_option_price,
)
from models.engine import (
    implied_volatility,
    calculate_historical_volatility,
    price_with_model,
    numerical_greeks,
    option_metrics,
    iv_hv_stats,
    atm_iv_from_chain,
    risk_reversal_and_fly,
)

__all__ = [
    "BlackScholesModel",
    "bs_call_price_vectorized",
    "binomial_american_option",
    "heston_mc_price",
    "garch_mc_price",
    "bates_jump_diffusion_mc_price",
    "monte_carlo_option_price",
    "implied_volatility",
    "calculate_historical_volatility",
    "price_with_model",
    "numerical_greeks",
    "option_metrics",
    "iv_hv_stats",
    "atm_iv_from_chain",
    "risk_reversal_and_fly",
]
