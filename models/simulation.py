from typing import Tuple, Optional

import numpy as np
from numpy import exp

from utils.numba_compat import jit
from models.utils import clamp_inputs


@jit(nopython=True, cache=True)
def _heston_mc_core(
    S: float,
    K: float,
    T: float,
    r: float,
    drift_adj: float,
    v0: float,
    kappa: float,
    theta: float,
    rho: float,
    vol_of_vol: float,
    z1_all: np.ndarray,
    z2_all: np.ndarray,
    is_call: bool
) -> float:
    """Numba-optimized core for Heston model Monte Carlo.
    
    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        drift_adj: Adjusted drift (r - q - borrow).
        v0: Initial variance.
        kappa: Mean reversion speed.
        theta: Long-run variance.
        rho: Correlation between asset and variance.
        vol_of_vol: Volatility of volatility.
        z1_all: Random normals for asset path.
        z2_all: Random normals for variance path.
        is_call: True if call option.

    Returns:
        float: Option price.
    """
    paths = z1_all.shape[1]
    steps = z1_all.shape[0]
    dt = T / steps
    rho_comp = np.sqrt(1.0 - rho * rho)

    prices = np.full(paths, S, dtype=np.float64)
    v = np.full(paths, v0, dtype=np.float64)

    for t in range(steps):
        z1 = z1_all[t]
        z2 = z2_all[t]
        z2_corr = rho * z1 + rho_comp * z2

        for i in range(paths):
            v_sqrt = np.sqrt(max(v[i], 1e-8))
            v[i] = max(v[i] + kappa * (theta - v[i]) * dt + vol_of_vol * v_sqrt * np.sqrt(dt) * z2_corr[i], 1e-8)
            prices[i] = prices[i] * np.exp((drift_adj - 0.5 * v[i]) * dt + np.sqrt(v[i] * dt) * z1[i])

    total = 0.0
    if is_call:
        for i in range(paths):
            total += max(prices[i] - K, 0.0)
    else:
        for i in range(paths):
            total += max(K - prices[i], 0.0)

    return np.exp(-r * T) * total / paths


def heston_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float = 0.0,
    option_type: str = "call",
    kappa: float = 1.5,
    theta: float = 0.04,
    v0: Optional[float] = None,
    rho: float = -0.7,
    vol_of_vol: float = 0.3,
    paths: int = 4000,
    steps: int = 120,
    seed: int = 42
) -> float:
    """Calculates option price using Heston Stochastic Volatility model.

    Args:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        sigma (float): Current volatility (used for initial variance if v0 is None).
        borrow_cost (float, optional): Cost to borrow. Defaults to 0.0.
        option_type (str, optional): 'call' or 'put'. Defaults to "call".
        kappa (float, optional): Mean reversion speed. Defaults to 1.5.
        theta (float, optional): Long-run average variance. Defaults to 0.04.
        v0 (float, optional): Initial variance. Defaults to None (uses sigma^2).
        rho (float, optional): Correlation between price and volatility. Defaults to -0.7.
        vol_of_vol (float, optional): Volatility of volatility. Defaults to 0.3.
        paths (int, optional): Number of MC paths. Defaults to 4000.
        steps (int, optional): Number of time steps. Defaults to 120.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        float: Option price.
    """
    rng = np.random.default_rng(seed)
    drift_adj = r - q - borrow_cost
    v0_val = sigma ** 2 if v0 is None else v0
    is_call = option_type == "call"

    # Pre-generate all random numbers
    z1_all = rng.standard_normal((steps, paths))
    z2_all = rng.standard_normal((steps, paths))

    return _heston_mc_core(S, K, T, r, drift_adj, v0_val, kappa, theta, rho, vol_of_vol, z1_all, z2_all, is_call)


@jit(nopython=True, cache=True)
def _garch_mc_core(
    S: float,
    K: float,
    T: float,
    r: float,
    drift_adj: float,
    sigma_sq: float,
    alpha0: float,
    alpha1: float,
    beta1: float,
    z_all: np.ndarray,
    is_call: bool
) -> float:
    """Numba-optimized core for GARCH Monte Carlo.
    
    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        drift_adj: Adjusted drift.
        sigma_sq: Initial squared volatility.
        alpha0: GARCH parameter alpha0.
        alpha1: GARCH parameter alpha1.
        beta1: GARCH parameter beta1.
        z_all: Random normals.
        is_call: True if call.

    Returns:
        float: Option price.
    """
    steps = z_all.shape[0]
    paths = z_all.shape[1]
    dt = T / steps

    prices = np.full(paths, S, dtype=np.float64)
    variance = np.full(paths, sigma_sq, dtype=np.float64)

    for t in range(steps):
        z = z_all[t]
        for i in range(paths):
            variance[i] = alpha0 + alpha1 * variance[i] * (z[i] ** 2) + beta1 * variance[i]
            variance[i] = max(variance[i], 1e-8)
            prices[i] = prices[i] * np.exp((drift_adj - 0.5 * variance[i]) * dt + np.sqrt(variance[i] * dt) * z[i])

    total = 0.0
    if is_call:
        for i in range(paths):
            total += max(prices[i] - K, 0.0)
    else:
        for i in range(paths):
            total += max(K - prices[i], 0.0)

    return np.exp(-r * T) * total / paths


def garch_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float = 0.0,
    option_type: str = "call",
    alpha0: float = 2e-6,
    alpha1: float = 0.08,
    beta1: float = 0.9,
    paths: int = 4000,
    seed: int = 42
) -> float:
    """GARCH(1,1) volatility model with Numba acceleration.

    Args:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        sigma (float): Current volatility.
        borrow_cost (float, optional): Cost to borrow. Defaults to 0.0.
        option_type (str, optional): 'call' or 'put'. Defaults to "call".
        alpha0 (float, optional): GARCH parameter. Defaults to 2e-6.
        alpha1 (float, optional): GARCH parameter. Defaults to 0.08.
        beta1 (float, optional): GARCH parameter. Defaults to 0.9.
        paths (int, optional): Number of MC paths. Defaults to 4000.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        float: Option price.
    """
    rng = np.random.default_rng(seed)
    steps = max(10, int(252 * T))
    drift_adj = r - q - borrow_cost
    is_call = option_type == "call"

    # Pre-generate all random numbers
    z_all = rng.standard_normal((steps, paths))

    return _garch_mc_core(S, K, T, r, drift_adj, sigma ** 2, alpha0, alpha1, beta1, z_all, is_call)


@jit(nopython=True, cache=True)
def _bates_mc_core(
    S: float,
    K: float,
    T: float,
    r: float,
    drift_adj: float,
    sigma: float,
    compensator: float,
    mu_jump: float,
    delta_jump: float,
    z_all: np.ndarray,
    jumps_all: np.ndarray,
    jump_normals_all: np.ndarray,
    is_call: bool
) -> float:
    """Numba-optimized core for Bates jump-diffusion Monte Carlo.
    
    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        drift_adj: Adjusted drift.
        sigma: Volatility.
        compensator: Jump compensator term.
        mu_jump: Mean jump size.
        delta_jump: Jump volatility.
        z_all: Random normals for diffusion.
        jumps_all: Poisson jump counts.
        jump_normals_all: Random normals for jump sizes.
        is_call: True if call.

    Returns:
        float: Option price.
    """
    steps = z_all.shape[0]
    paths = z_all.shape[1]
    dt = T / steps
    sigma_sqrt_dt = sigma * np.sqrt(dt)
    drift_term = (drift_adj - 0.5 * sigma * sigma - compensator) * dt

    prices = np.full(paths, S, dtype=np.float64)

    for t in range(steps):
        z = z_all[t]
        jumps = jumps_all[t]
        jump_normals = jump_normals_all[t]

        for i in range(paths):
            jump_size = 0.0
            if jumps[i] > 0:
                jump_size = mu_jump * jumps[i] + delta_jump * np.sqrt(jumps[i]) * jump_normals[i]
            prices[i] = prices[i] * np.exp(drift_term + sigma_sqrt_dt * z[i] + jump_size)

    total = 0.0
    if is_call:
        for i in range(paths):
            total += max(prices[i] - K, 0.0)
    else:
        for i in range(paths):
            total += max(K - prices[i], 0.0)

    return np.exp(-r * T) * total / paths


def bates_jump_diffusion_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float = 0.0,
    option_type: str = "call",
    lambda_jump: float = 0.1,
    mu_jump: float = -0.05,
    delta_jump: float = 0.2,
    paths: int = 4000,
    steps: int = 120,
    seed: int = 42
) -> float:
    """Bates jump-diffusion model with Numba acceleration.

    Args:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        q (float): Dividend yield.
        sigma (float): Volatility.
        borrow_cost (float, optional): Cost to borrow. Defaults to 0.0.
        option_type (str, optional): 'call' or 'put'. Defaults to "call".
        lambda_jump (float, optional): Jump intensity. Defaults to 0.1.
        mu_jump (float, optional): Mean jump size. Defaults to -0.05.
        delta_jump (float, optional): Jump volatility. Defaults to 0.2.
        paths (int, optional): Number of MC paths. Defaults to 4000.
        steps (int, optional): Number of time steps. Defaults to 120.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        float: Option price.
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    drift_adj = r - q - borrow_cost
    compensator = lambda_jump * (exp(mu_jump + 0.5 * delta_jump ** 2) - 1)
    is_call = option_type == "call"

    # Pre-generate all random numbers
    z_all = rng.standard_normal((steps, paths))
    jumps_all = rng.poisson(lambda_jump * dt, (steps, paths)).astype(np.float64)
    jump_normals_all = rng.standard_normal((steps, paths))

    return _bates_mc_core(S, K, T, r, drift_adj, sigma, compensator, mu_jump, delta_jump, z_all, jumps_all, jump_normals_all, is_call)


def monte_carlo_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    num_simulations: int = 10000,
    option_type: str = "call",
    seed: int = 42,
    q: float = 0.0,
    borrow_cost: float = 0.0,
) -> Tuple[float, float, np.ndarray]:
    """Standard Monte Carlo option pricing.

    Args:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        num_simulations (int, optional): Number of simulations. Defaults to 10000.
        option_type (str, optional): 'call' or 'put'. Defaults to "call".
        seed (int, optional): Random seed. Defaults to 42.
        q (float, optional): Dividend yield. Defaults to 0.0.
        borrow_cost (float, optional): Cost to borrow. Defaults to 0.0.

    Returns:
        Tuple[float, float, np.ndarray]: (price, standard_error, terminal_prices).
    """
    T, sigma = clamp_inputs(T, sigma)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(num_simulations)
    drift = r - q - borrow_cost
    ST = S * np.exp((drift - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(num_simulations)

    return option_price, std_error, ST
