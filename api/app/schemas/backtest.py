"""Pydantic schemas for strategy and backtest endpoints."""
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class StrategyLeg(BaseModel):
    type: str = Field(pattern="^(call|put)$")
    strike: float = Field(gt=0)
    qty: int = Field(ge=1, default=1)
    side: str = Field(pattern="^(long|short)$")
    entry_price: Optional[float] = None


class SpotRange(BaseModel):
    min: float = Field(gt=0)
    max: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self):
        if self.min >= self.max:
            raise ValueError("spot_range.min must be < spot_range.max")
        return self


class PayoffRequest(BaseModel):
    legs: list[StrategyLeg] = Field(min_length=1)
    spot_range: SpotRange
    S: float = Field(gt=0, description="Current spot for entry pricing")
    T: float = Field(ge=0, default=0.0833, description="Time to expiry for entry pricing")
    r: float = Field(default=0.05, ge=-0.5, le=2.0)
    sigma: float = Field(ge=0, default=0.2)


class PayoffResponse(BaseModel):
    prices: list[float]
    pnl: list[float]
    breakevens: list[float]
    max_profit: Optional[float]
    max_loss: Optional[float]


class BacktestLeg(BaseModel):
    type: str = Field(pattern="^(call|put)$")
    strike: float = Field(gt=0)
    expiry: str
    qty: int = Field(ge=1, default=1)
    side: str = Field(pattern="^(long|short)$")

    @field_validator("expiry")
    @classmethod
    def validate_expiry(cls, v: str) -> str:
        date.fromisoformat(v)
        return v


class BacktestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-\^]+$")
    legs: list[BacktestLeg] = Field(min_length=1)
    r: float = Field(default=0.05, ge=-0.5, le=2.0)
    sigma: float = Field(ge=0, default=0.2)


class PnLPoint(BaseModel):
    date: str
    pnl: float


class BacktestResponse(BaseModel):
    pnl_series: list[PnLPoint]
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    win_rate: float
