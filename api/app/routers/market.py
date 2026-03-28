"""Market data API endpoints."""
from fastapi import APIRouter, Path
from starlette.concurrency import run_in_threadpool

from api.app.schemas.market import MarketResponse, ChainResponse
from api.app.services.market_service import get_market_data, get_options_chain

router = APIRouter(tags=["market"])

TICKER_PATH = Path(..., min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.\-\^]+$")


@router.get("/market/{ticker}", response_model=MarketResponse)
async def market_data(ticker: str = TICKER_PATH):
    return await run_in_threadpool(lambda: get_market_data(ticker.upper()))


@router.get("/chain/{ticker}", response_model=ChainResponse)
async def options_chain(ticker: str = TICKER_PATH):
    return await run_in_threadpool(lambda: get_options_chain(ticker.upper()))
