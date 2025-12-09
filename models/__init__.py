"""
Models package for Black-Scholes Trader.

This package contains option pricing models organized by methodology:
- black_scholes: Closed-form analytical solutions (Black-Scholes-Merton)
- numerical: Tree-based methods (Binomial/Trinomial)
- simulation: Monte Carlo engines (Heston, GARCH, Bates jump-diffusion)
"""

from models.black_scholes import BlackScholesModel, bs_call_price_vectorized
from models.numerical import binomial_american_option
from models.simulation import (
    heston_mc_price,
    garch_mc_price,
    bates_jump_diffusion_mc_price,
    monte_carlo_option_price,
)

__all__ = [
    "BlackScholesModel",
    "bs_call_price_vectorized",
    "binomial_american_option",
    "heston_mc_price",
    "garch_mc_price",
    "bates_jump_diffusion_mc_price",
    "monte_carlo_option_price",
]
