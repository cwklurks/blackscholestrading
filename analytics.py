import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy import exp, log, sqrt
from scipy.optimize import minimize_scalar
from scipy.stats import norm

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Small epsilons to avoid NaNs/infs when T or sigma are near zero
EPS_TIME = 1e-8
EPS_VOL = 1e-8


def clamp_inputs(T: float, sigma: float) -> Tuple[float, float]:
    """Ensure time and volatility stay positive for numerical stability."""
    return max(T, EPS_TIME), max(sigma, EPS_VOL)


class BlackScholesModel:
    """Black-Scholes option pricing model with Greeks.
    
    Attributes:
        S: Spot price of the underlying asset
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility (annualized)
        dividend_yield: Continuous dividend yield
        borrow_cost: Cost of borrowing the underlying
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        dividend_yield: float = 0.0,
        borrow_cost: float = 0.0,
    ):
        # Input validation
        if S <= 0:
            raise ValueError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got {K}")
        if T < 0:
            raise ValueError(f"Time to expiry cannot be negative, got {T}")
        if sigma < 0:
            raise ValueError(f"Volatility cannot be negative, got {sigma}")
            
        self.S = S
        self.K = K
        self.T, self.sigma = clamp_inputs(T, sigma)
        self.r = r
        self.dividend_yield = dividend_yield
        self.borrow_cost = borrow_cost
        self.carry = self.r - self.dividend_yield - self.borrow_cost

    def d1(self) -> float:
        return (log(self.S / self.K) + (self.carry + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * sqrt(self.T)
        )

    def d2(self) -> float:
        return self.d1() - self.sigma * sqrt(self.T)

    def call_price(self) -> float:
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return self.S * decay * norm.cdf(self.d1()) - self.K * exp(-self.r * self.T) * norm.cdf(
            self.d2()
        )

    def put_price(self) -> float:
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return self.K * exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * decay * norm.cdf(
            -self.d1()
        )

    def delta_call(self) -> float:
        return exp(-(self.dividend_yield + self.borrow_cost) * self.T) * norm.cdf(self.d1())

    def delta_put(self) -> float:
        return -exp(-(self.dividend_yield + self.borrow_cost) * self.T) * norm.cdf(-self.d1())

    def gamma(self) -> float:
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return decay * norm.pdf(self.d1()) / (self.S * self.sigma * sqrt(self.T))

    def theta_call(self) -> float:
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        term1 = -self.S * decay * norm.pdf(self.d1()) * self.sigma / (2 * sqrt(self.T))
        term2 = -self.r * self.K * exp(-self.r * self.T) * norm.cdf(self.d2())
        term3 = (self.dividend_yield + self.borrow_cost) * self.S * decay * norm.cdf(self.d1())
        return (term1 + term2 + term3) / 365

    def theta_put(self) -> float:
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        term1 = -self.S * decay * norm.pdf(self.d1()) * self.sigma / (2 * sqrt(self.T))
        term2 = self.r * self.K * exp(-self.r * self.T) * norm.cdf(-self.d2())
        term3 = -(self.dividend_yield + self.borrow_cost) * self.S * decay * norm.cdf(-self.d1())
        return (term1 + term2 + term3) / 365

    def vega(self) -> float:
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return self.S * decay * norm.pdf(self.d1()) * sqrt(self.T) / 100

    def rho_call(self) -> float:
        return self.K * self.T * exp(-self.r * self.T) * norm.cdf(self.d2()) / 100

    def rho_put(self) -> float:
        return -self.K * self.T * exp(-self.r * self.T) * norm.cdf(-self.d2()) / 100


def bs_call_price_vectorized(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray,
    dividend_yield: float = 0.0,
    borrow_cost: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes call price calculation.
    
    Efficiently computes call prices for a grid of spot prices and volatilities.
    
    Args:
        S: Array of spot prices (can be 1D or 2D meshgrid)
        K: Strike price (scalar)
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Array of volatilities (same shape as S)
        dividend_yield: Continuous dividend yield
        borrow_cost: Borrow cost for short selling
        
    Returns:
        Array of call prices with same shape as S
    """
    T = max(T, EPS_TIME)
    sigma = np.maximum(sigma, EPS_VOL)
    
    carry = r - dividend_yield - borrow_cost
    decay = np.exp(-(dividend_yield + borrow_cost) * T)
    
    d1 = (np.log(S / K) + (carry + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * decay * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(option_price: float, S: float, K: float, T: float, r: float, q: float = 0.0, borrow_cost: float = 0.0, option_type: str = "call") -> float:
    T, _ = clamp_inputs(T, EPS_VOL)

    def objective(sigma: float) -> float:
        sigma = max(sigma, EPS_VOL)
        model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
        if option_type == "call":
            return abs(model.call_price() - option_price)
        return abs(model.put_price() - option_price)

    result = minimize_scalar(objective, bounds=(0.001, 5), method="bounded")
    return float(result.x)


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
):
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


def calculate_historical_volatility(prices: pd.Series, periods: int = 252) -> float:
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        return np.nan
    return float(returns.std() * np.sqrt(periods))


def iv_hv_stats(implied_vol: float, hv_series: pd.Series) -> Optional[dict]:
    hv_series = hv_series.dropna()
    if hv_series.empty or implied_vol is None:
        return None

    hv_current = hv_series.iloc[-1]
    percentile = float((hv_series < implied_vol).mean())
    rank = float(pd.concat([hv_series, pd.Series([implied_vol])]).rank(pct=True).iloc[-1])
    edge = float(implied_vol - hv_current)
    return {
        "hv_current": hv_current,
        "iv_hv_percentile": percentile,
        "iv_hv_rank": rank,
        "edge": edge,
    }


def atm_iv_from_chain(options_df: pd.DataFrame, expiry: str, spot: float) -> Optional[float]:
    subset = options_df[options_df["expiration"] == expiry]
    if subset.empty:
        return None
    idx = (subset["strike"] - spot).abs().idxmin()
    if pd.isna(idx):
        return None
    return float(subset.loc[idx, "impliedVolatility"])


def risk_reversal_and_fly(
    options_df: pd.DataFrame,
    spot: float,
    expiry: str,
    call_moneyness: float = 1.1,
    put_moneyness: float = 0.9,
) -> Optional[dict]:
    subset = options_df[options_df["expiration"] == expiry]
    if subset.empty or spot <= 0:
        return None

    atm_iv = atm_iv_from_chain(options_df, expiry, spot)

    def _nearest_iv(df: pd.DataFrame, target: float) -> Optional[float]:
        if df.empty:
            return None
        idx = (df["strike"] - target).abs().idxmin()
        if pd.isna(idx):
            return None
        return float(df.loc[idx, "impliedVolatility"])

    calls = subset[subset["type"].str.lower() == "call"]
    puts = subset[subset["type"].str.lower() == "put"]

    rr_call_iv = _nearest_iv(calls, spot * call_moneyness)
    rr_put_iv = _nearest_iv(puts, spot * put_moneyness)

    if atm_iv is None or rr_call_iv is None or rr_put_iv is None:
        return None

    rr = rr_call_iv - rr_put_iv
    fly = 0.5 * (rr_call_iv + rr_put_iv) - atm_iv
    return {"risk_reversal": rr, "butterfly": fly, "atm_iv": atm_iv}


def generate_gbm_replay(
    start_price: float,
    days: int,
    r: float,
    sigma: float,
    seed: int = 7,
) -> pd.Series:
    """Create a GBM price path as a proxy for missing historical data."""
    _, sigma = clamp_inputs(EPS_TIME, sigma)
    rng = np.random.default_rng(seed)
    dt_step = 1 / 252
    shocks = rng.standard_normal(days)
    prices = [start_price]
    for shock in shocks:
        next_price = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt_step + sigma * np.sqrt(dt_step) * shock)
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
    """Mark option P&L across a price history."""
    if prices is None or prices.empty:
        return pd.DataFrame()

    expiry_dt = pd.to_datetime(expiry)
    option_type = option_type.lower()
    side_sign = 1 if side == "long" else -1

    records = []
    entry_option_price = None

    for idx, spot in prices.items():
        idx_dt = pd.to_datetime(idx)
        # Normalize to tz-naive to avoid tz-naive vs tz-aware subtraction errors
        if idx_dt.tzinfo is not None:
            idx_dt = idx_dt.tz_localize(None)
        days_to_expiry = max((expiry_dt - idx_dt).days, 0)
        T = days_to_expiry / 365
        model = BlackScholesModel(spot, strike, T, r, sigma)
        theo = model.call_price() if option_type == "call" else model.put_price()
        if entry_option_price is None:
            entry_option_price = theo
        pnl = side_sign * (theo - entry_option_price) * quantity
        records.append({
            "date": pd.to_datetime(idx),
            "spot": spot,
            "option_price": theo,
            "pnl": pnl,
            "time_to_expiry": T,
        })

    return pd.DataFrame(records).set_index("date")


@jit(nopython=True, cache=True)
def _binomial_american_core(S, K, T, r, carry, sigma, steps, is_call):
    """Numba-optimized core for American option pricing via binomial tree."""
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


def binomial_american_option(S, K, T, r, q, sigma, borrow_cost=0.0, steps=150, option_type="call"):
    """American option pricing via binomial tree with Numba acceleration."""
    steps = max(3, steps)
    carry = r - q - borrow_cost
    is_call = option_type == "call"
    return _binomial_american_core(S, K, T, r, carry, sigma, steps, is_call)


@jit(nopython=True, cache=True)
def _heston_mc_core(S, K, T, r, drift_adj, v0, kappa, theta, rho, vol_of_vol, z1_all, z2_all, is_call):
    """Numba-optimized core for Heston model Monte Carlo."""
    paths = z1_all.shape[1]
    steps = z1_all.shape[0]
    dt = T / steps
    rho_comp = np.sqrt(1.0 - rho * rho)

    prices = np.full(paths, S)
    v = np.full(paths, v0)

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


def heston_mc_price(S, K, T, r, q, sigma, borrow_cost=0.0, option_type="call", kappa=1.5, theta=0.04, v0=None, rho=-0.7, vol_of_vol=0.3, paths=4000, steps=120, seed=42):
    """Heston stochastic volatility model with Numba acceleration."""
    rng = np.random.default_rng(seed)
    drift_adj = r - q - borrow_cost
    v0_val = sigma ** 2 if v0 is None else v0
    is_call = option_type == "call"

    # Pre-generate all random numbers
    z1_all = rng.standard_normal((steps, paths))
    z2_all = rng.standard_normal((steps, paths))

    return _heston_mc_core(S, K, T, r, drift_adj, v0_val, kappa, theta, rho, vol_of_vol, z1_all, z2_all, is_call)


@jit(nopython=True, cache=True)
def _garch_mc_core(S, K, T, r, drift_adj, sigma_sq, alpha0, alpha1, beta1, z_all, is_call):
    """Numba-optimized core for GARCH Monte Carlo."""
    steps = z_all.shape[0]
    paths = z_all.shape[1]
    dt = T / steps

    prices = np.full(paths, S)
    variance = np.full(paths, sigma_sq)

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


def garch_mc_price(S, K, T, r, q, sigma, borrow_cost=0.0, option_type="call", alpha0=2e-6, alpha1=0.08, beta1=0.9, paths=4000, seed=42):
    """GARCH(1,1) volatility model with Numba acceleration."""
    rng = np.random.default_rng(seed)
    steps = max(10, int(252 * T))
    drift_adj = r - q - borrow_cost
    is_call = option_type == "call"

    # Pre-generate all random numbers
    z_all = rng.standard_normal((steps, paths))

    return _garch_mc_core(S, K, T, r, drift_adj, sigma ** 2, alpha0, alpha1, beta1, z_all, is_call)


@jit(nopython=True, cache=True)
def _bates_mc_core(S, K, T, r, drift_adj, sigma, compensator, mu_jump, delta_jump, z_all, jumps_all, jump_normals_all, is_call):
    """Numba-optimized core for Bates jump-diffusion Monte Carlo."""
    steps = z_all.shape[0]
    paths = z_all.shape[1]
    dt = T / steps
    sigma_sqrt_dt = sigma * np.sqrt(dt)
    drift_term = (drift_adj - 0.5 * sigma * sigma - compensator) * dt

    prices = np.full(paths, S)

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


def bates_jump_diffusion_mc_price(S, K, T, r, q, sigma, borrow_cost=0.0, option_type="call", lambda_jump=0.1, mu_jump=-0.05, delta_jump=0.2, paths=4000, steps=120, seed=42):
    """Bates jump-diffusion model with Numba acceleration."""
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


def price_with_model(model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params):
    if model_name == "Black-Scholes":
        model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
        return model.call_price() if option_type == "call" else model.put_price()
    if model_name == "Binomial (American)":
        return binomial_american_option(S, K, T, r, q, sigma, borrow_cost, steps=model_params.get("binomial_steps", 150), option_type=option_type)
    if model_name == "Heston MC":
        return heston_mc_price(
            S,
            K,
            T,
            r,
            q,
            sigma,
            borrow_cost,
            option_type,
            kappa=model_params.get("heston_kappa", 1.5),
            theta=model_params.get("heston_theta", 0.04),
            v0=model_params.get("heston_v0", sigma**2),
            rho=model_params.get("heston_rho", -0.7),
            vol_of_vol=model_params.get("heston_vol_of_vol", 0.3),
            paths=model_params.get("mc_paths", 4000),
            steps=model_params.get("mc_steps", 120),
        )
    if model_name == "GARCH MC":
        return garch_mc_price(
            S,
            K,
            T,
            r,
            q,
            sigma,
            borrow_cost,
            option_type,
            alpha0=model_params.get("garch_alpha0", 2e-6),
            alpha1=model_params.get("garch_alpha1", 0.08),
            beta1=model_params.get("garch_beta1", 0.9),
            paths=model_params.get("mc_paths", 4000),
        )
    if model_name == "Bates Jump-Diffusion":
        return bates_jump_diffusion_mc_price(
            S,
            K,
            T,
            r,
            q,
            sigma,
            borrow_cost,
            option_type,
            lambda_jump=model_params.get("jump_lambda", 0.1),
            mu_jump=model_params.get("jump_mu", -0.05),
            delta_jump=model_params.get("jump_delta", 0.2),
            paths=model_params.get("mc_paths", 4000),
            steps=model_params.get("mc_steps", 120),
        )
    model = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
    return model.call_price() if option_type == "call" else model.put_price()


def numerical_greeks(model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params):
    base_price = price_with_model(model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params)
    ds = max(0.01, S * 0.01)
    dv = max(0.0001, sigma * 0.05)
    dt_step = min(1 / 365, T * 0.5) if T > 0 else 1 / 365
    dr = 0.0001

    price_up = price_with_model(model_name, S + ds, K, T, r, q, sigma, borrow_cost, option_type, model_params)
    price_dn = price_with_model(model_name, S - ds, K, T, r, q, sigma, borrow_cost, option_type, model_params)
    delta = (price_up - price_dn) / (2 * ds)
    gamma = (price_up - 2 * base_price + price_dn) / (ds**2)

    price_vol_up = price_with_model(model_name, S, K, T, r, q, sigma + dv, borrow_cost, option_type, model_params)
    price_vol_dn = price_with_model(model_name, S, K, T, r, q, sigma - dv, borrow_cost, option_type, model_params)
    vega = (price_vol_up - price_vol_dn) / (2 * dv) / 100

    T_forward = max(T - dt_step, EPS_TIME)
    price_time = price_with_model(model_name, S, K, T_forward, r, q, sigma, borrow_cost, option_type, model_params)
    theta = (price_time - base_price) / dt_step / -365

    price_r_up = price_with_model(model_name, S, K, T, r + dr, q, sigma, borrow_cost, option_type, model_params)
    price_r_dn = price_with_model(model_name, S, K, T, r - dr, q, sigma, borrow_cost, option_type, model_params)
    rho = (price_r_up - price_r_dn) / (2 * dr) / 100

    return {"price": base_price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def option_metrics(model_name, S, K, T, r, q, sigma, borrow_cost, model_params, option_type="call"):
    if model_name == "Black-Scholes":
        bs = BlackScholesModel(S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost)
        return {
            "price": bs.call_price() if option_type == "call" else bs.put_price(),
            "delta": bs.delta_call() if option_type == "call" else bs.delta_put(),
            "gamma": bs.gamma(),
            "vega": bs.vega(),
            "theta": bs.theta_call() if option_type == "call" else bs.theta_put(),
            "rho": bs.rho_call() if option_type == "call" else bs.rho_put(),
        }
    return numerical_greeks(model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params)
