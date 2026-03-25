"""Strategy and backtest API endpoints."""
from fastapi import APIRouter

from api.app.schemas.backtest import (
    PayoffRequest, PayoffResponse,
    BacktestRequest, BacktestResponse,
)
from api.app.services.backtest_service import compute_payoff, run_backtest

router = APIRouter(tags=["backtest"])


@router.post("/strategy/payoff", response_model=PayoffResponse)
async def strategy_payoff(request: PayoffRequest):
    legs = [leg.model_dump() for leg in request.legs]
    return compute_payoff(
        legs=legs,
        spot_range=request.spot_range.model_dump(),
        S=request.S, T=request.T, r=request.r, sigma=request.sigma,
    )


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest):
    legs = [leg.model_dump() for leg in request.legs]
    return run_backtest(
        ticker=request.ticker.upper(),
        legs=legs, r=request.r, sigma=request.sigma,
    )
