import datetime as dt
from typing import Optional, Dict, Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Import models and utils
from models.black_scholes import BlackScholesModel, bs_call_price_vectorized
from models.numerical import binomial_american_option
from models.simulation import (
    heston_mc_price,
    garch_mc_price,
    bates_jump_diffusion_mc_price,
    monte_carlo_option_price,
)
from models.utils import clamp_inputs, EPS_VOL, EPS_TIME


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    borrow_cost: float = 0.0,
    option_type: str = "call",
) -> float:
    """Calculate implied volatility using Brent's method (via minimize_scalar)."""
    T, _ = clamp_inputs(T, EPS_VOL)

    def objective(sigma: float) -> float:
        sigma = max(sigma, EPS_VOL)
        model = BlackScholesModel(
            S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost
        )
        if option_type == "call":
            return abs(model.call_price() - option_price)
        return abs(model.put_price() - option_price)

    result = minimize_scalar(objective, bounds=(0.001, 5), method="bounded")
    return float(result.x)


def calculate_historical_volatility(prices: pd.Series, periods: int = 252) -> float:
    """Calculate annualized historical volatility from price series."""
    returns = pd.Series(np.log(prices / prices.shift(1))).dropna()
    if returns.empty:
        return np.nan
    return float(returns.std() * np.sqrt(periods))


def iv_hv_stats(implied_vol: float, hv_series: pd.Series) -> Optional[dict]:
    """Compare Implied Volatility to Historical Volatility distribution."""
    hv_series = hv_series.dropna()
    if hv_series.empty or implied_vol is None:
        return None

    hv_current = hv_series.iloc[-1]
    percentile = float(cast(float, (hv_series < implied_vol).mean()))
    rank = float(
        cast(float, pd.concat([hv_series, pd.Series([implied_vol])]).rank(pct=True).iloc[-1])
    )
    edge = float(implied_vol - hv_current)
    return {
        "hv_current": hv_current,
        "iv_hv_percentile": percentile,
        "iv_hv_rank": rank,
        "edge": edge,
    }


def atm_iv_from_chain(
    options_df: pd.DataFrame, expiry: str, spot: float
) -> Optional[float]:
    """Extract At-The-Money Implied Volatility from options chain."""
    subset = options_df[options_df["expiration"] == expiry]
    if subset.empty:
        return None
    idx = (subset["strike"] - spot).abs().idxmin()
    if pd.isna(idx):
        return None
    return float(cast(float, subset.loc[idx, "impliedVolatility"]))


def risk_reversal_and_fly(
    options_df: pd.DataFrame,
    spot: float,
    expiry: str,
    call_moneyness: float = 1.1,
    put_moneyness: float = 0.9,
) -> Optional[dict]:
    """Calculate Risk Reversal and Butterfly skew metrics."""
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
        return float(cast(float, df.loc[idx, "impliedVolatility"]))

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
    """Mark option P&L across a price history."""
    if prices is None or prices.empty:
        return pd.DataFrame()

    expiry_dt = pd.to_datetime(expiry)
    option_type = option_type.lower()
    side_sign = 1 if side == "long" else -1

    records = []
    entry_option_price = None

    for idx, spot in prices.items():
        idx_dt = pd.to_datetime(str(idx))
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
            "date": pd.to_datetime(str(idx)),
            "spot": spot,
            "option_price": theo,
            "pnl": pnl,
            "time_to_expiry": T,
        })

    return pd.DataFrame(records).set_index("date")


def price_with_model(
    model_name: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float,
    option_type: str,
    model_params: Dict[str, Any],
) -> float:
    """Route pricing request to the selected model."""
    if model_name == "Black-Scholes":
        model = BlackScholesModel(
            S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost
        )
        return model.call_price() if option_type == "call" else model.put_price()
    if model_name == "Binomial (American)":
        return binomial_american_option(
            S,
            K,
            T,
            r,
            q,
            sigma,
            borrow_cost,
            steps=model_params.get("binomial_steps", 150),
            option_type=option_type,
        )
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
    # Default fallback
    model = BlackScholesModel(
        S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost
    )
    return model.call_price() if option_type == "call" else model.put_price()


def numerical_greeks(
    model_name: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float,
    option_type: str,
    model_params: Dict[str, Any],
) -> Dict[str, float]:
    """Calculate Greeks using finite difference method for any model."""
    base_price = price_with_model(
        model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params
    )
    ds = max(0.01, S * 0.01)
    dv = max(0.0001, sigma * 0.05)
    dt_step = min(1 / 365, T * 0.5) if T > 0 else 1 / 365
    dr = 0.0001

    price_up = price_with_model(
        model_name,
        S + ds,
        K,
        T,
        r,
        q,
        sigma,
        borrow_cost,
        option_type,
        model_params,
    )
    price_dn = price_with_model(
        model_name,
        S - ds,
        K,
        T,
        r,
        q,
        sigma,
        borrow_cost,
        option_type,
        model_params,
    )
    delta = (price_up - price_dn) / (2 * ds)
    gamma = (price_up - 2 * base_price + price_dn) / (ds**2)

    price_vol_up = price_with_model(
        model_name,
        S,
        K,
        T,
        r,
        q,
        sigma + dv,
        borrow_cost,
        option_type,
        model_params,
    )
    price_vol_dn = price_with_model(
        model_name,
        S,
        K,
        T,
        r,
        q,
        sigma - dv,
        borrow_cost,
        option_type,
        model_params,
    )
    vega = (price_vol_up - price_vol_dn) / (2 * dv) / 100

    T_forward = max(T - dt_step, EPS_TIME)
    price_time = price_with_model(
        model_name,
        S,
        K,
        T_forward,
        r,
        q,
        sigma,
        borrow_cost,
        option_type,
        model_params,
    )
    theta = (price_time - base_price) / dt_step / -365

    price_r_up = price_with_model(
        model_name,
        S,
        K,
        T,
        r + dr,
        q,
        sigma,
        borrow_cost,
        option_type,
        model_params,
    )
    price_r_dn = price_with_model(
        model_name,
        S,
        K,
        T,
        r - dr,
        q,
        sigma,
        borrow_cost,
        option_type,
        model_params,
    )
    rho = (price_r_up - price_r_dn) / (2 * dr) / 100

    return {
        "price": base_price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def option_metrics(
    model_name: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    borrow_cost: float,
    model_params: Dict[str, Any],
    option_type: str = "call",
) -> Dict[str, float]:
    """Calculate option price and Greeks."""
    if model_name == "Black-Scholes":
        bs = BlackScholesModel(
            S, K, T, r, sigma, dividend_yield=q, borrow_cost=borrow_cost
        )
        return {
            "price": bs.call_price() if option_type == "call" else bs.put_price(),
            "delta": bs.delta_call() if option_type == "call" else bs.delta_put(),
            "gamma": bs.gamma(),
            "vega": bs.vega(),
            "theta": bs.theta_call() if option_type == "call" else bs.theta_put(),
            "rho": bs.rho_call() if option_type == "call" else bs.rho_put(),
        }
    return numerical_greeks(
        model_name, S, K, T, r, q, sigma, borrow_cost, option_type, model_params
    )
