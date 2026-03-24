"""Pricing API endpoints."""
from fastapi import APIRouter

from api.app.schemas.pricing import (
    PricingRequest, PricingResponse,
    HeatmapRequest, HeatmapResponse,
    MonteCarloRequest, MonteCarloResponse,
)
from api.app.services.pricing_service import compute_price, compute_heatmap
from models.simulation import monte_carlo_option_price

router = APIRouter(tags=["pricing"])


@router.post("/price", response_model=PricingResponse)
async def price_option(request: PricingRequest):
    result = compute_price(model_name=request.model, **request.to_engine_kwargs())
    return result


@router.post("/heatmap", response_model=HeatmapResponse)
async def compute_heatmap_endpoint(request: HeatmapRequest):
    result = compute_heatmap(
        K=request.K, T=request.T, r=request.r, q=request.q,
        borrow_cost=request.borrow_cost,
        spot_range=request.spot_range.model_dump(),
        vol_range=request.vol_range.model_dump(),
    )
    return result


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def monte_carlo(request: MonteCarloRequest):
    price, se, terminal = monte_carlo_option_price(
        S=request.S, K=request.K, T=request.T, r=request.r,
        sigma=request.sigma, num_simulations=request.paths,
        option_type=request.option_type, q=request.q,
        borrow_cost=request.borrow_cost,
    )
    ci = [price - 1.96 * se, price + 1.96 * se]
    return {
        "price": price,
        "std_error": se,
        "terminal_prices": terminal.tolist(),
        "confidence_interval": ci,
    }
