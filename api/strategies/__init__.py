"""
Strategies package for Black-Scholes Trader.

This package contains trading strategy implementations:
- multi_leg: Multi-leg option strategy payoff computation
"""

from strategies.multi_leg import MultiLegStrategy

__all__ = [
    "MultiLegStrategy",
]
