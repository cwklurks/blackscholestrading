"""Tests for advanced pricing models in analytics.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analytics import (
    BlackScholesModel,
    binomial_american_option,
    heston_mc_price,
    garch_mc_price,
    bates_jump_diffusion_mc_price,
    bs_call_price_vectorized,
)


class TestInputValidation:
    """Test input validation for BlackScholesModel."""
    
    def test_negative_spot_raises_error(self):
        with pytest.raises(ValueError, match="Spot price must be positive"):
            BlackScholesModel(-100, 100, 0.5, 0.05, 0.2)
            
    def test_zero_spot_raises_error(self):
        with pytest.raises(ValueError, match="Spot price must be positive"):
            BlackScholesModel(0, 100, 0.5, 0.05, 0.2)
    
    def test_negative_strike_raises_error(self):
        with pytest.raises(ValueError, match="Strike price must be positive"):
            BlackScholesModel(100, -100, 0.5, 0.05, 0.2)
            
    def test_negative_time_raises_error(self):
        with pytest.raises(ValueError, match="Time to expiry cannot be negative"):
            BlackScholesModel(100, 100, -0.5, 0.05, 0.2)
            
    def test_negative_volatility_raises_error(self):
        with pytest.raises(ValueError, match="Volatility cannot be negative"):
            BlackScholesModel(100, 100, 0.5, 0.05, -0.2)


class TestPutCallParity:
    """Test put-call parity for Black-Scholes model."""
    
    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (100, 100, 1.0, 0.05, 0.2),
        (100, 110, 0.5, 0.02, 0.3),
        (100, 90, 0.25, 0.01, 0.15),
        (50, 55, 0.75, 0.03, 0.25),
        (200, 180, 2.0, 0.04, 0.35),
    ])
    def test_put_call_parity(self, S, K, T, r, sigma):
        """Verify C - P = S - K*exp(-r*T) for European options."""
        model = BlackScholesModel(S, K, T, r, sigma)
        call = model.call_price()
        put = model.put_price()
        parity = call - put - (S - K * np.exp(-r * T))
        assert abs(parity) < 1e-8, f"Put-call parity violated: {parity}"


class TestBinomialAmericanOption:
    """Tests for binomial American option pricing."""
    
    def test_american_put_geq_european_put(self):
        """American put should be worth at least as much as European put."""
        S, K, T, r, q, sigma = 100, 105, 0.5, 0.05, 0.0, 0.25
        
        american_put = binomial_american_option(S, K, T, r, q, sigma, option_type="put")
        european_put = BlackScholesModel(S, K, T, r, sigma).put_price()
        
        assert american_put >= european_put - 1e-6, \
            f"American put ({american_put}) < European put ({european_put})"
    
    def test_american_call_no_dividend_equals_european(self):
        """Without dividends, American call equals European call."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.2
        
        american_call = binomial_american_option(S, K, T, r, 0.0, sigma, option_type="call")
        european_call = BlackScholesModel(S, K, T, r, sigma).call_price()
        
        # Allow for numerical error from discrete tree
        assert abs(american_call - european_call) < 0.1, \
            f"American call ({american_call}) != European call ({european_call})"
    
    def test_deep_itm_american_put_early_exercise(self):
        """Deep ITM American put with high rates should show early exercise value."""
        S, K, T, r, q, sigma = 50, 100, 1.0, 0.10, 0.0, 0.2
        
        american_put = binomial_american_option(S, K, T, r, q, sigma, option_type="put")
        intrinsic = max(K - S, 0)
        
        # American put should be worth at least intrinsic value
        assert american_put >= intrinsic - 1e-6


class TestMonteCarloModels:
    """Tests for Monte Carlo pricing models."""
    
    def test_heston_converges_to_bs_low_vol_of_vol(self):
        """Heston should converge to BS when vol-of-vol is near zero."""
        S, K, T, r, q, sigma = 100, 100, 0.5, 0.05, 0.0, 0.2
        
        bs_price = BlackScholesModel(S, K, T, r, sigma).call_price()
        heston_price = heston_mc_price(
            S, K, T, r, q, sigma,
            kappa=2.0,
            theta=sigma**2,
            v0=sigma**2,
            rho=0.0,
            vol_of_vol=0.01,  # Very low vol-of-vol
            paths=10000,
            steps=100,
            seed=42
        )
        
        # Allow for MC sampling error
        assert abs(heston_price - bs_price) < 1.0, \
            f"Heston ({heston_price}) too far from BS ({bs_price})"
    
    def test_garch_produces_reasonable_prices(self):
        """GARCH MC should produce prices in a reasonable range."""
        S, K, T, r, q, sigma = 100, 100, 0.5, 0.05, 0.0, 0.2
        
        garch_price = garch_mc_price(
            S, K, T, r, q, sigma,
            alpha0=1e-6,
            alpha1=0.05,
            beta1=0.90,
            paths=5000,
            seed=42
        )
        
        bs_price = BlackScholesModel(S, K, T, r, sigma).call_price()
        
        # GARCH price should be within reasonable range of BS
        assert 0.5 * bs_price < garch_price < 2.0 * bs_price, \
            f"GARCH price ({garch_price}) unreasonable vs BS ({bs_price})"
    
    def test_bates_converges_to_bs_no_jumps(self):
        """Bates should converge to BS when jump intensity is zero."""
        S, K, T, r, q, sigma = 100, 100, 0.5, 0.05, 0.0, 0.2
        
        bs_price = BlackScholesModel(S, K, T, r, sigma).call_price()
        bates_price = bates_jump_diffusion_mc_price(
            S, K, T, r, q, sigma,
            lambda_jump=0.0,  # No jumps
            mu_jump=0.0,
            delta_jump=0.0,
            paths=10000,
            steps=100,
            seed=42
        )
        
        assert abs(bates_price - bs_price) < 0.5, \
            f"Bates ({bates_price}) too far from BS ({bs_price}) with no jumps"


class TestVectorizedPricing:
    """Tests for vectorized pricing functions."""
    
    def test_vectorized_matches_scalar(self):
        """Vectorized BS pricing should match scalar implementation."""
        K, T, r, q, borrow = 100, 0.5, 0.05, 0.0, 0.0
        
        spot_range = np.array([90, 100, 110])
        vol_range = np.array([0.15, 0.20, 0.25])
        
        spot_grid, vol_grid = np.meshgrid(spot_range, vol_range)
        
        vectorized_prices = bs_call_price_vectorized(
            spot_grid, K, T, r, vol_grid, q, borrow
        )
        
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                scalar_price = BlackScholesModel(spot, K, T, r, vol, q, borrow).call_price()
                assert abs(vectorized_prices[i, j] - scalar_price) < 1e-10, \
                    f"Mismatch at S={spot}, sigma={vol}"
    
    def test_vectorized_handles_1d_arrays(self):
        """Vectorized function should work with 1D arrays."""
        K, T, r = 100, 0.5, 0.05
        spots = np.linspace(80, 120, 5)
        vols = np.full_like(spots, 0.2)
        
        prices = bs_call_price_vectorized(spots, K, T, r, vols)
        
        assert prices.shape == spots.shape
        assert all(np.isfinite(prices))


class TestReproducibility:
    """Test that seeded functions produce reproducible results."""
    
    def test_heston_reproducible(self):
        """Heston MC with same seed should produce same result."""
        params = dict(S=100, K=100, T=0.5, r=0.05, q=0.0, sigma=0.2, paths=1000, steps=50)
        
        price1 = heston_mc_price(**params, seed=123)
        price2 = heston_mc_price(**params, seed=123)
        price3 = heston_mc_price(**params, seed=456)
        
        assert price1 == price2, "Same seed should produce same result"
        assert price1 != price3, "Different seeds should produce different results"
    
    def test_garch_reproducible(self):
        """GARCH MC with same seed should produce same result."""
        params = dict(S=100, K=100, T=0.5, r=0.05, q=0.0, sigma=0.2, paths=1000)
        
        price1 = garch_mc_price(**params, seed=123)
        price2 = garch_mc_price(**params, seed=123)
        
        assert price1 == price2, "Same seed should produce same result"
    
    def test_bates_reproducible(self):
        """Bates MC with same seed should produce same result."""
        params = dict(S=100, K=100, T=0.5, r=0.05, q=0.0, sigma=0.2, paths=1000, steps=50)
        
        price1 = bates_jump_diffusion_mc_price(**params, seed=123)
        price2 = bates_jump_diffusion_mc_price(**params, seed=123)
        
        assert price1 == price2, "Same seed should produce same result"

