"""
Numerical constants and input validation utilities.

These constants prevent numerical instabilities in option pricing calculations
when time-to-expiry or volatility approach zero.
"""

from typing import Tuple

# Small epsilons to avoid NaNs/infs when T or sigma are near zero
EPS_TIME: float = 1e-8
EPS_VOL: float = 1e-8


def clamp_inputs(T: float, sigma: float) -> Tuple[float, float]:
    """Ensure time and volatility stay positive for numerical stability.
    
    Args:
        T: Time to expiration (years)
        sigma: Volatility (annualized)
        
    Returns:
        Tuple of (clamped_T, clamped_sigma) with minimum values of EPS_TIME and EPS_VOL
    """
    return max(T, EPS_TIME), max(sigma, EPS_VOL)

