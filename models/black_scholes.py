import numpy as np
from numpy import exp, log, sqrt
from scipy.stats import norm

from models.utils import clamp_inputs, EPS_TIME, EPS_VOL


class BlackScholesModel:
    """Black-Scholes option pricing model with Greeks.

    Attributes:
        S (float): Spot price of the underlying asset.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility (annualized).
        dividend_yield (float): Continuous dividend yield.
        borrow_cost (float): Cost of borrowing the underlying.
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
        """Initialize the Black-Scholes model.

        Args:
            S (float): Spot price of the underlying asset.
            K (float): Strike price.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility (annualized).
            dividend_yield (float, optional): Continuous dividend yield. Defaults to 0.0.
            borrow_cost (float, optional): Cost of borrowing the underlying. Defaults to 0.0.

        Raises:
            ValueError: If inputs are invalid (negative prices, time, volatility).
        """
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
        """Calculate the d1 term in the Black-Scholes formula.

        Returns:
            float: The d1 value.
        """
        return (log(self.S / self.K) + (self.carry + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * sqrt(self.T)
        )

    def d2(self) -> float:
        """Calculate the d2 term in the Black-Scholes formula.

        Returns:
            float: The d2 value.
        """
        return self.d1() - self.sigma * sqrt(self.T)

    def call_price(self) -> float:
        """Calculate the price of a call option.

        Returns:
            float: Call option price.
        """
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return self.S * decay * norm.cdf(self.d1()) - self.K * exp(
            -self.r * self.T
        ) * norm.cdf(self.d2())

    def put_price(self) -> float:
        """Calculate the price of a put option.

        Returns:
            float: Put option price.
        """
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return self.K * exp(-self.r * self.T) * norm.cdf(
            -self.d2()
        ) - self.S * decay * norm.cdf(-self.d1())

    def delta_call(self) -> float:
        """Calculate the delta of a call option.

        Returns:
            float: Call delta.
        """
        return exp(-(self.dividend_yield + self.borrow_cost) * self.T) * norm.cdf(
            self.d1()
        )

    def delta_put(self) -> float:
        """Calculate the delta of a put option.

        Returns:
            float: Put delta.
        """
        return -exp(-(self.dividend_yield + self.borrow_cost) * self.T) * norm.cdf(
            -self.d1()
        )

    def gamma(self) -> float:
        """Calculate the gamma of the option (same for call and put).

        Returns:
            float: Gamma.
        """
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return decay * norm.pdf(self.d1()) / (self.S * self.sigma * sqrt(self.T))

    def theta_call(self) -> float:
        """Calculate the theta of a call option.

        Returns:
            float: Call theta (per day).
        """
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        term1 = (
            -self.S * decay * norm.pdf(self.d1()) * self.sigma / (2 * sqrt(self.T))
        )
        term2 = -self.r * self.K * exp(-self.r * self.T) * norm.cdf(self.d2())
        term3 = (
            (self.dividend_yield + self.borrow_cost)
            * self.S
            * decay
            * norm.cdf(self.d1())
        )
        return (term1 + term2 + term3) / 365

    def theta_put(self) -> float:
        """Calculate the theta of a put option.

        Returns:
            float: Put theta (per day).
        """
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        term1 = (
            -self.S * decay * norm.pdf(self.d1()) * self.sigma / (2 * sqrt(self.T))
        )
        term2 = self.r * self.K * exp(-self.r * self.T) * norm.cdf(-self.d2())
        term3 = (
            -(self.dividend_yield + self.borrow_cost)
            * self.S
            * decay
            * norm.cdf(-self.d1())
        )
        return (term1 + term2 + term3) / 365

    def vega(self) -> float:
        """Calculate the vega of the option.

        Returns:
            float: Vega (per 1% change in volatility).
        """
        decay = exp(-(self.dividend_yield + self.borrow_cost) * self.T)
        return self.S * decay * norm.pdf(self.d1()) * sqrt(self.T) / 100

    def rho_call(self) -> float:
        """Calculate the rho of a call option.

        Returns:
            float: Call rho (per 1% change in rate).
        """
        return (
            self.K * self.T * exp(-self.r * self.T) * norm.cdf(self.d2()) / 100
        )

    def rho_put(self) -> float:
        """Calculate the rho of a put option.

        Returns:
            float: Put rho (per 1% change in rate).
        """
        return (
            -self.K * self.T * exp(-self.r * self.T) * norm.cdf(-self.d2()) / 100
        )


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
        S (np.ndarray): Array of spot prices (can be 1D or 2D meshgrid).
        K (float): Strike price (scalar).
        T (float): Time to expiry in years.
        r (float): Risk-free rate.
        sigma (np.ndarray): Array of volatilities (same shape as S).
        dividend_yield (float, optional): Continuous dividend yield. Defaults to 0.0.
        borrow_cost (float, optional): Borrow cost for short selling. Defaults to 0.0.

    Returns:
        np.ndarray: Array of call prices with same shape as S.
    """
    T = max(T, EPS_TIME)
    sigma = np.maximum(sigma, EPS_VOL)

    carry = r - dividend_yield - borrow_cost
    decay = np.exp(-(dividend_yield + borrow_cost) * T)

    d1 = (np.log(S / K) + (carry + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * decay * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
