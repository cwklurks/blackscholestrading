import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analytics import (  # noqa: E402
    backtest_option_strategy,
    iv_hv_stats,
)
from models import (
    BlackScholesModel,
    monte_carlo_option_price,
)


def test_greeks_align_with_bump_and_revalue():
    model = BlackScholesModel(100, 100, 0.5, 0.02, 0.25)
    base_call = model.call_price()

    spot = 100
    ds = 0.01 * spot
    price_up = BlackScholesModel(spot + ds, 100, 0.5, 0.02, 0.25).call_price()
    price_dn = BlackScholesModel(spot - ds, 100, 0.5, 0.02, 0.25).call_price()
    delta_fd = (price_up - price_dn) / (2 * ds)
    gamma_fd = (price_up - 2 * base_call + price_dn) / (ds**2)

    dv = 0.01
    price_vol_up = BlackScholesModel(spot, 100, 0.5, 0.02, 0.25 + dv).call_price()
    price_vol_dn = BlackScholesModel(spot, 100, 0.5, 0.02, 0.25 - dv).call_price()
    vega_fd = (price_vol_up - price_vol_dn) / (2 * dv) / 100

    assert math.isclose(model.delta_call(), delta_fd, rel_tol=5e-3)
    assert math.isclose(model.gamma(), gamma_fd, rel_tol=1e-2)
    assert math.isclose(model.vega(), vega_fd, rel_tol=1e-2)


def test_monte_carlo_tracks_closed_form():
    S, K, T, r, sigma = 105, 100, 0.75, 0.01, 0.2
    bs_price = BlackScholesModel(S, K, T, r, sigma).call_price()
    mc_price, se, _ = monte_carlo_option_price(S, K, T, r, sigma, num_simulations=20000, option_type="call", seed=7)
    assert abs(mc_price - bs_price) < max(3 * se, 0.4)


def test_inputs_are_clamped_and_no_nan():
    model = BlackScholesModel(50, 50, 0.0, 0.01, 0.0)
    assert math.isfinite(model.call_price())
    assert math.isfinite(model.gamma())


def test_iv_hv_stats_and_backtest_helpers():
    prices = pd.Series(
        np.linspace(90, 110, 30),
        index=pd.date_range(end=pd.Timestamp.today(), periods=30, freq="B"),
    )
    hv_series = prices.pct_change().rolling(window=5).std() * np.sqrt(252)
    hv_stats = iv_hv_stats(0.25, hv_series)
    assert hv_stats is not None and math.isfinite(hv_stats["edge"])
    bt_df = backtest_option_strategy(
        prices,
        strike=100,
        expiry=pd.Timestamp.today() + pd.Timedelta(days=30),
        r=0.01,
        sigma=0.2,
    )
    assert bt_df.empty is False
