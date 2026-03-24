"""Market data API endpoints."""
from fastapi import APIRouter

from api.app.schemas.market import MarketResponse, ChainResponse
from api.app.services.market_service import get_market_data, get_options_chain

router = APIRouter(tags=["market"])


@router.get("/market/{ticker}", response_model=MarketResponse)
async def market_data(ticker: str):
    return get_market_data(ticker.upper())


@router.get("/chain/{ticker}", response_model=ChainResponse)
async def options_chain(ticker: str):
    return get_options_chain(ticker.upper())
