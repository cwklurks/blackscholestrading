import numpy as np
from utils.numba_compat import jit

@jit(nopython=True, cache=True)
def _binomial_american_core(S: float, K: float, T: float, r: float, carry: float, sigma: float, steps: int, is_call: bool) -> float:
    """Numba-optimized core for American option pricing via binomial tree.

    Args:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        carry (float): Cost of carry (r - q - borrow).
        sigma (float): Volatility.
        steps (int): Number of time steps.
        is_call (bool): True for call, False for put.

    Returns:
        float: Option price.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(carry * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    disc = np.exp(-r * dt)

    # Initialize asset prices at maturity
    prices = np.empty(steps + 1)
    for i in range(steps + 1):
        prices[i] = S * (u ** (steps - i)) * (d ** i)

    # Initialize option values at maturity
    option = np.empty(steps + 1)
    if is_call:
        for i in range(steps + 1):
            option[i] = max(prices[i] - K, 0.0)
    else:
        for i in range(steps + 1):
            option[i] = max(K - prices[i], 0.0)

    # Backward induction
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            prices[i] = prices[i] / u
            continuation = disc * (p * option[i] + (1 - p) * option[i + 1])
            if is_call:
                intrinsic = max(prices[i] - K, 0.0)
            else:
                intrinsic = max(K - prices[i], 0.0)
            option[i] = max(continuation, intrinsic)

    return option[0]


def binomial_american_option(S: float, K: float, T: float, r: float, q: float, sigma: float, borrow_cost: float = 0.0, steps: int = 150, option_type: str = "call") -> float:
    """American option pricing via binomial tree with Numba acceleration.

    Args:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        sigma (float): Volatility.
        borrow_cost (float, optional): Cost to borrow. Defaults to 0.0.
        steps (int, optional): Number of time steps. Defaults to 150.
        option_type (str, optional): 'call' or 'put'. Defaults to "call".

    Returns:
        float: Option price.
    """
    steps = max(3, steps)
    carry = r - q - borrow_cost
    is_call = option_type == "call"
    return _binomial_american_core(S, K, T, r, carry, sigma, steps, is_call)
