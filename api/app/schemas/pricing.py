"""Pydantic schemas for pricing endpoints."""
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class RangeSpec(BaseModel):
    min: float
    max: float
    steps: int = Field(ge=2, le=100, default=20)


class PricingRequest(BaseModel):
    model: str = Field(default="Black-Scholes", description="Pricing model name")
    S: float = Field(gt=0, description="Spot price")
    K: float = Field(gt=0, description="Strike price")
    T: float = Field(ge=0, description="Time to expiry in years")
    r: float = Field(description="Risk-free rate")
    sigma: float = Field(ge=0, le=5.0, description="Volatility")
    q: float = Field(default=0.0, description="Dividend yield")
    borrow_cost: float = Field(default=0.0, description="Borrow cost")
    option_type: str = Field(default="call", pattern="^(call|put)$")
    model_params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_model_params(self):
        """Whitelist and clamp all model-specific params to prevent CPU abuse."""
        PARAM_BOUNDS = {
            "mc_paths": (100, 50000),
            "mc_steps": (10, 500),
            "binomial_steps": (10, 500),
            "heston_kappa": (0.01, 20.0),
            "heston_theta": (0.0001, 1.0),
            "heston_v0": (0.0001, 1.0),
            "heston_rho": (-1.0, 1.0),
            "heston_vol_of_vol": (0.01, 2.0),
            "garch_alpha0": (1e-8, 0.01),
            "garch_alpha1": (0.001, 0.5),
            "garch_beta1": (0.5, 0.999),
            "jump_lambda": (0.0, 2.0),
            "jump_mu": (-0.5, 0.5),
            "jump_delta": (0.01, 1.0),
        }
        params = {}
        for key, value in self.model_params.items():
            if key in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[key]
                params[key] = max(lo, min(float(value), hi))
        self.model_params = params
        return self

    def to_engine_kwargs(self) -> dict:
        return {
            "S": self.S, "K": self.K, "T": self.T, "r": self.r,
            "q": self.q, "sigma": self.sigma, "borrow_cost": self.borrow_cost,
            "model_params": self.model_params, "option_type": self.option_type,
        }


class PricingResponse(BaseModel):
    model: str
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class HeatmapRequest(BaseModel):
    K: float = Field(gt=0)
    T: float = Field(ge=0)
    r: float
    q: float = 0.0
    borrow_cost: float = 0.0
    spot_range: RangeSpec
    vol_range: RangeSpec

    @model_validator(mode="after")
    def validate_ranges(self):
        """Enforce sane range bounds to prevent NaN/Inf grids."""
        if self.spot_range.min <= 0:
            raise ValueError("spot_range.min must be > 0")
        if self.spot_range.min >= self.spot_range.max:
            raise ValueError("spot_range.min must be < spot_range.max")
        if self.spot_range.max > 1_000_000:
            raise ValueError("spot_range.max must be <= 1,000,000")
        if self.vol_range.min <= 0:
            raise ValueError("vol_range.min must be > 0")
        if self.vol_range.min >= self.vol_range.max:
            raise ValueError("vol_range.min must be < vol_range.max")
        if self.vol_range.max > 10.0:
            raise ValueError("vol_range.max must be <= 10.0")
        return self


class HeatmapResponse(BaseModel):
    spot_values: list[float]
    vol_values: list[float]
    call_prices: list[list[float]]
    put_prices: list[list[float]]


class MonteCarloRequest(BaseModel):
    S: float = Field(gt=0)
    K: float = Field(gt=0)
    T: float = Field(ge=0)
    r: float
    sigma: float = Field(ge=0)
    paths: int = Field(ge=100, le=50000, default=10000)
    option_type: str = Field(default="call", pattern="^(call|put)$")
    q: float = 0.0
    borrow_cost: float = 0.0


class MonteCarloResponse(BaseModel):
    price: float
    std_error: float
    terminal_prices: list[float]
    confidence_interval: list[float]


class VolSurfaceRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-\^]+$")
    strikes: Optional[list[float]] = Field(default=None, max_length=200)
    expirations: Optional[list[str]] = Field(default=None, max_length=20)
    r: float = Field(default=0.05, description="Risk-free rate for IV computation")


class VolSurfacePoint(BaseModel):
    strike: float
    expiry: str
    iv: Optional[float] = None


class VolSurfaceResponse(BaseModel):
    surface: list[VolSurfacePoint]
    smile_data: dict[str, list[dict]]
    coverage: float
