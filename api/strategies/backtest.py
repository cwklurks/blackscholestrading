"""
Backtesting framework for options strategies.

This module provides tools for historical replay and P&L analysis of
options strategies. It includes:
- Abstract Strategy base class for custom strategy implementations
- GBM price path generation for synthetic backtesting
- Mark-to-market P&L tracking

The Strategy ABC allows easy extension to complex multi-leg strategies
(straddles, strangles, iron condors, etc.) while maintaining a consistent
interface.
"""

import datetime as dt
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from models.black_scholes import BlackScholesModel
from utils.constants import EPS_TIME, clamp_inputs


class Strategy(ABC):
    """Abstract base class for options trading strategies.
    
    Implement this interface to create custom strategies that can be
    backtested consistently. The `run()` method receives price data
    and should return a DataFrame of P&L records.
    
    Example:
        >>> class MyStraddle(Strategy):
        ...     def __init__(self, strike, expiry, r, sigma):
        ...         self.strike = strike
        ...         self.expiry = expiry
        ...         self.r = r
        ...         self.sigma = sigma
        ...     
        ...     def run(self, prices: pd.Series) -> pd.DataFrame:
        ...         # Implement straddle logic
        ...         pass
    """
    
    @abstractmethod
    def run(self, prices: pd.Series) -> pd.DataFrame:
        """Execute the strategy over a price history.
        
        Args:
            prices: Time series of underlying prices with datetime index
            
        Returns:
            DataFrame with columns for date, spot, option metrics, and P&L
        """
        pass


class SingleOptionStrategy(Strategy):
    """Single-leg option strategy (long/short call or put).
    
    This is the simplest strategy implementation, tracking the P&L
    of a single option position over time using Black-Scholes valuation.
    
    Attributes:
        strike: Option strike price
        expiry: Expiration date
        r: Risk-free rate
        sigma: Implied volatility
        option_type: "call" or "put"
        quantity: Number of contracts
        side: "long" or "short"
    """
    
    def __init__(
        self,
        strike: float,
        expiry: dt.date,
        r: float,
        sigma: float,
        option_type: str = "call",
        quantity: int = 1,
        side: str = "long",
    ):
        """Initialize single option strategy.
        
        Args:
            strike: Strike price
            expiry: Expiration date
            r: Risk-free rate
            sigma: Volatility assumption
            option_type: "call" or "put"
            quantity: Number of contracts
            side: "long" (bought) or "short" (sold)
        """
        self.strike = strike
        self.expiry = expiry
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.quantity = quantity
        self.side = side
        self._side_sign = 1 if side == "long" else -1
    
    def run(self, prices: pd.Series) -> pd.DataFrame:
        """Execute strategy and compute mark-to-market P&L.
        
        Args:
            prices: Time series of spot prices
            
        Returns:
            DataFrame with date, spot, option_price, pnl, time_to_expiry
        """
        if prices is None or prices.empty:
            return pd.DataFrame()
        
        expiry_dt = pd.to_datetime(self.expiry)
        records = []
        entry_option_price = None
        
        for idx, spot in prices.items():
            idx_dt = pd.to_datetime(str(idx))
            # Normalize to tz-naive to avoid comparison errors
            if idx_dt.tzinfo is not None:
                idx_dt = idx_dt.tz_localize(None)
            
            days_to_expiry = max((expiry_dt - idx_dt).days, 0)
            T = days_to_expiry / 365
            
            model = BlackScholesModel(spot, self.strike, T, self.r, self.sigma)
            theo = model.call_price() if self.option_type == "call" else model.put_price()
            
            if entry_option_price is None:
                entry_option_price = theo
            
            pnl = self._side_sign * (theo - entry_option_price) * self.quantity
            
            records.append({
                "date": pd.to_datetime(str(idx)),
                "spot": spot,
                "option_price": theo,
                "pnl": pnl,
                "time_to_expiry": T,
            })
        
        return pd.DataFrame(records).set_index("date")


def generate_gbm_replay(
    start_price: float,
    days: int,
    r: float,
    sigma: float,
    seed: int = 7,
) -> pd.Series:
    """Generate synthetic price path using Geometric Brownian Motion.
    
    Creates a realistic-looking price series for backtesting when
    historical data is unavailable. Useful for:
    - Strategy development and testing
    - Stress testing with different volatility assumptions
    - Educational demonstrations
    
    Args:
        start_price: Initial price level
        days: Number of trading days to simulate
        r: Drift rate (typically risk-free rate)
        sigma: Volatility (annualized)
        seed: Random seed for reproducibility
        
    Returns:
        pd.Series with datetime index and simulated prices
        
    Example:
        >>> prices = generate_gbm_replay(100, 252, 0.05, 0.2, seed=42)
        >>> print(f"Start: ${prices.iloc[0]:.2f}, End: ${prices.iloc[-1]:.2f}")
    """
    _, sigma = clamp_inputs(EPS_TIME, sigma)
    rng = np.random.default_rng(seed)
    dt_step = 1 / 252  # Daily steps assuming 252 trading days
    
    shocks = rng.standard_normal(days)
    prices = [start_price]
    
    for shock in shocks:
        next_price = prices[-1] * np.exp(
            (r - 0.5 * sigma**2) * dt_step + sigma * np.sqrt(dt_step) * shock
        )
        prices.append(next_price)
    
    index = pd.date_range(end=dt.date.today(), periods=len(prices), freq="B")
    return pd.Series(prices, index=index)


def backtest_option_strategy(
    prices: pd.Series,
    strike: float,
    expiry: dt.date,
    r: float,
    sigma: float,
    option_type: str = "call",
    quantity: int = 1,
    side: str = "long",
) -> pd.DataFrame:
    """Backtest a single option strategy over historical prices.
    
    This is a convenience function that wraps SingleOptionStrategy
    for backward compatibility and simple use cases.
    
    Args:
        prices: Time series of underlying spot prices
        strike: Option strike price
        expiry: Expiration date
        r: Risk-free rate
        sigma: Volatility assumption
        option_type: "call" or "put"
        quantity: Number of contracts
        side: "long" or "short"
        
    Returns:
        DataFrame with columns:
        - spot: Underlying price
        - option_price: Theoretical option value
        - pnl: Cumulative P&L from entry
        - time_to_expiry: Years remaining to expiration
        
    Example:
        >>> prices = generate_gbm_replay(100, 60, 0.05, 0.2)
        >>> results = backtest_option_strategy(
        ...     prices, strike=100, expiry=dt.date(2024, 3, 15),
        ...     r=0.05, sigma=0.2, option_type="call", side="long"
        ... )
        >>> print(f"Final P&L: ${results['pnl'].iloc[-1]:.2f}")
    """
    strategy = SingleOptionStrategy(
        strike=strike,
        expiry=expiry,
        r=r,
        sigma=sigma,
        option_type=option_type,
        quantity=quantity,
        side=side,
    )
    return strategy.run(prices)

