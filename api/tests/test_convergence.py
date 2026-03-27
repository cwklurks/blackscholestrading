
import pytest
import numpy as np
from models import monte_carlo_option_price, BlackScholesModel

def test_monte_carlo_convergence():
    """
    Test that Monte Carlo simulation converges to the analytical Black-Scholes price
    and that the standard error decreases as expected.
    """
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    
    # Analytical Price
    bs_model = BlackScholesModel(S, K, T, r, sigma)
    analytical_price = bs_model.call_price()
    
    simulations = [1000, 10000, 100000]
    results = []
    
    for n in simulations:
        mc_price, std_err, _ = monte_carlo_option_price(
            S, K, T, r, sigma, num_simulations=n, option_type="call", seed=42
        )
        results.append((n, mc_price, std_err))
        
        # Check convergence within 2 standard errors
        diff = abs(mc_price - analytical_price)
        assert diff < 2 * std_err + 0.05, f"MC price {mc_price} not within 2*stderr of analytical {analytical_price} for N={n}"
        
    # Check standard error decrease
    # Std error should decrease by roughly sqrt(N_ratio)
    # Compare 1000 to 100000 (100x increase in N -> 10x decrease in SE)
    se_1k = results[0][2]
    se_100k = results[2][2]
    
    expected_decrease = np.sqrt(100000 / 1000) # 10
    actual_decrease = se_1k / se_100k
    
    # Allow some wiggle room, but it should be close to 10
    assert 8 < actual_decrease < 12, f"Standard error did not decrease as expected: {actual_decrease} vs {expected_decrease}"


