"""
Utilities package for Black-Scholes Trader.

This package contains shared utilities and constants:
- constants: Numerical stability constants and input clamping
- numba_compat: Numba JIT compatibility layer
"""

from utils.constants import EPS_TIME, EPS_VOL, clamp_inputs
from utils.numba_compat import jit, NUMBA_AVAILABLE

__all__ = [
    "EPS_TIME",
    "EPS_VOL",
    "clamp_inputs",
    "jit",
    "NUMBA_AVAILABLE",
]

