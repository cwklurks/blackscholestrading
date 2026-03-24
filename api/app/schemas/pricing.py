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
    def clamp_mc_params(self):
        """Enforce mc_paths <= 50000 and mc_steps <= 500 (eng review #6)."""
        params = dict(self.model_params)
        if "mc_paths" in params:
            params["mc_paths"] = min(params["mc_paths"], 50000)
        if "mc_steps" in params:
            params["mc_steps"] = min(params["mc_steps"], 500)
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
    ticker: str
    strikes: Optional[list[float]] = None
    expirations: Optional[list[str]] = None


class VolSurfacePoint(BaseModel):
    strike: float
    expiry: str
    iv: Optional[float] = None


class VolSurfaceResponse(BaseModel):
    surface: list[VolSurfacePoint]
    smile_data: dict[str, list[dict]]
    coverage: float
