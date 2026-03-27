"""Pydantic schemas for market data endpoints."""
from typing import Optional
from pydantic import BaseModel


class OHLCVRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class MarketResponse(BaseModel):
    price: float
    history: list[OHLCVRow]
    historical_vol: float
    fetched_at: str


class ChainRow(BaseModel):
    strike: float
    lastPrice: float
    iv: Optional[float] = None
    volume: Optional[float] = None
    oi: Optional[float] = None


class ChainResponse(BaseModel):
    calls: list[ChainRow]
    puts: list[ChainRow]
    expirations: list[str]
