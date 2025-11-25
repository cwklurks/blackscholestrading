import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy import exp, log, sqrt
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# Small epsilons to avoid NaNs/infs when T or sigma are near zero
EPS_TIME = 1e-8
EPS_VOL = 1e-8


def clamp_inputs(T: float, sigma: float) -> Tuple[float, float]:
    """Ensure time and volatility stay positive for numerical stability."""
    return max(T, EPS_TIME), max(sigma, EPS_VOL)


class BlackScholesModel:
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
        days_to_expiry = max((expiry_dt - pd.to_datetime(idx)).days, 0)
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


def binomial_american_option(S, K, T, r, q, sigma, borrow_cost=0.0, steps=150, option_type="call"):
    steps = max(3, steps)
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    carry = r - q - borrow_cost
    p = (exp(carry * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    disc = exp(-r * dt)

    prices = np.array([S * (u ** (steps - i)) * (d ** i) for i in range(steps + 1)])
    option = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)

    for step in range(steps - 1, -1, -1):
        prices = prices[: step + 1] / u
        continuation = disc * (p * option[: step + 1] + (1 - p) * option[1 : step + 2])
        intrinsic = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)
        option = np.maximum(continuation, intrinsic)

    return option[0]


def heston_mc_price(S, K, T, r, q, sigma, borrow_cost=0.0, option_type="call", kappa=1.5, theta=0.04, v0=None, rho=-0.7, vol_of_vol=0.3, paths=4000, steps=120):
    np.random.seed(42)
    dt = T / steps
    v = np.full(paths, sigma**2 if v0 is None else v0)
    prices = np.full(paths, S, dtype=float)
    drift_adj = r - q - borrow_cost

    for _ in range(steps):
        z1 = np.random.standard_normal(paths)
        z2 = np.random.standard_normal(paths)
        z2 = rho * z1 + sqrt(1 - rho**2) * z2
        v = np.maximum(
            v + kappa * (theta - v) * dt + vol_of_vol * np.sqrt(np.maximum(v, 1e-8)) * sqrt(dt) * z2,
            1e-8,
        )
        prices = prices * np.exp((drift_adj - 0.5 * v) * dt + np.sqrt(v * dt) * z1)

    payoff = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)
    disc = exp(-r * T)
    return disc * np.mean(payoff)


def garch_mc_price(S, K, T, r, q, sigma, borrow_cost=0.0, option_type="call", alpha0=2e-6, alpha1=0.08, beta1=0.9, paths=4000):
    np.random.seed(42)
    steps = max(10, int(252 * T))
    dt = T / steps if steps else 1 / 252
    prices = np.full(paths, S, dtype=float)
    variance = np.full(paths, sigma**2, dtype=float)
    drift_adj = r - q - borrow_cost

    for _ in range(steps):
        z = np.random.standard_normal(paths)
        variance = alpha0 + alpha1 * variance * (z**2) + beta1 * variance
        variance = np.maximum(variance, 1e-8)
        prices = prices * np.exp((drift_adj - 0.5 * variance) * dt + np.sqrt(variance * dt) * z)

    payoff = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)
    return exp(-r * T) * np.mean(payoff)


def bates_jump_diffusion_mc_price(S, K, T, r, q, sigma, borrow_cost=0.0, option_type="call", lambda_jump=0.1, mu_jump=-0.05, delta_jump=0.2, paths=4000, steps=120):
    np.random.seed(42)
    dt = T / steps
    drift_adj = r - q - borrow_cost
    prices = np.full(paths, S, dtype=float)
    compensator = lambda_jump * (exp(mu_jump + 0.5 * delta_jump**2) - 1)

    for _ in range(steps):
        z = np.random.standard_normal(paths)
        jumps = np.random.poisson(lambda_jump * dt, paths)
        jump_sizes = np.where(jumps > 0, np.random.normal(mu_jump * jumps, delta_jump * np.sqrt(jumps)), 0.0)
        prices = prices * np.exp((drift_adj - 0.5 * sigma**2 - compensator) * dt + sigma * sqrt(dt) * z + jump_sizes)

    payoff = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)
    return exp(-r * T) * np.mean(payoff)


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
