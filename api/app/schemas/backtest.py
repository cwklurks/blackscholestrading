"""Pydantic schemas for strategy and backtest endpoints."""
from typing import Optional
from pydantic import BaseModel, Field


class StrategyLeg(BaseModel):
    type: str = Field(pattern="^(call|put)$")
    strike: float = Field(gt=0)
    qty: int = Field(ge=1, default=1)
    side: str = Field(pattern="^(long|short)$")
    entry_price: Optional[float] = None


class PayoffRequest(BaseModel):
    legs: list[StrategyLeg] = Field(min_length=1)
    spot_range: dict  # {min, max}
    S: float = Field(gt=0, description="Current spot for entry pricing")
    T: float = Field(ge=0, default=0.0833, description="Time to expiry for entry pricing")
    r: float = Field(default=0.05)
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


class BacktestRequest(BaseModel):
    ticker: str
    legs: list[BacktestLeg] = Field(min_length=1)
    r: float = Field(default=0.05)
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
