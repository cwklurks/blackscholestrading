
from hypothesis import given, strategies as st
import numpy as np
from analytics import BlackScholesModel

# Strategies for valid inputs
# Avoid extremely small T or sigma to prevent numerical issues
spot_strategy = st.floats(min_value=1.0, max_value=1000.0)
strike_strategy = st.floats(min_value=1.0, max_value=1000.0)
time_strategy = st.floats(min_value=0.01, max_value=5.0)
rate_strategy = st.floats(min_value=0.0, max_value=0.2)
vol_strategy = st.floats(min_value=0.05, max_value=2.0)

@given(
    S=spot_strategy,
    K=strike_strategy,
    T=time_strategy,
    r=rate_strategy,
    sigma=vol_strategy
)
def test_call_price_positive_delta(S, K, T, r, sigma):
    """Call Price must strictly increase as Spot increases (positive Delta)."""
    model1 = BlackScholesModel(S, K, T, r, sigma)
    price1 = model1.call_price()
    
    # Increase spot slightly
    S_new = S * 1.01
    model2 = BlackScholesModel(S_new, K, T, r, sigma)
    price2 = model2.call_price()
    
    # Strictly increasing if price is significant, otherwise non-decreasing
    if price1 > 1e-7:
        assert price2 > price1, f"Call price did not increase with spot: {price1} -> {price2} for S: {S} -> {S_new}"
    else:
        assert price2 >= price1, f"Call price decreased with spot (unexpected): {price1} -> {price2}"

@given(
    S=spot_strategy,
    K=strike_strategy,
    T=time_strategy,
    r=rate_strategy,
    sigma=vol_strategy
)
def test_option_price_theta_decay(S, K, T, r, sigma):
    """Option Price must strictly decrease as Time decreases (Theta decay), assuming ATM/OTM calls often exhibit this.
    Note: Theta isn't always negative for all options (e.g. deep ITM puts), 
    but for simple cases without dividends it generally holds for calls.
    We will test specific condition or generally relaxing for deep ITM.
    Let's focus on ATM calls for this property as per instructions.
    """
    # Force ATM-ish
    K = S 
    
    model1 = BlackScholesModel(S, K, T, r, sigma)
    price1 = model1.call_price()
    
    # Decrease time
    T_new = T * 0.9
    model2 = BlackScholesModel(S, K, T_new, r, sigma)
    price2 = model2.call_price()
    
    # With r >= 0, ATM call value should decrease as time decreases
    assert price2 < price1, f"ATM Call price did not decrease with time decay: {price1} -> {price2} for T: {T} -> {T_new}"

@given(
    S=spot_strategy,
    K=strike_strategy,
    T=time_strategy,
    r=rate_strategy,
    sigma=vol_strategy
)
def test_put_call_parity(S, K, T, r, sigma):
    """Assert Call - Put == Spot - Strike * exp(-rT)"""
    model = BlackScholesModel(S, K, T, r, sigma)
    call_price = model.call_price()
    put_price = model.put_price()
    
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    
    assert np.isclose(lhs, rhs, atol=1e-5), f"Put-Call Parity failed: {lhs} != {rhs}"


