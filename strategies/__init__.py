"""
Strategies package for Black-Scholes Trader.

This package contains trading strategy implementations and backtesting utilities:
- backtest: Historical strategy replay and P&L tracking
- Strategy: Abstract base class for custom strategy implementations
"""

from strategies.backtest import (
    Strategy,
    SingleOptionStrategy,
    backtest_option_strategy,
    generate_gbm_replay,
)

__all__ = [
    "Strategy",
    "SingleOptionStrategy",
    "backtest_option_strategy",
    "generate_gbm_replay",
]

