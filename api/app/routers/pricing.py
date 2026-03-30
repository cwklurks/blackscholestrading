"""Pricing API endpoints."""
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

try:
    from api.app.schemas.pricing import (
        PricingRequest, PricingResponse,
        HeatmapRequest, HeatmapResponse,
        MonteCarloRequest, MonteCarloResponse,
        VolSurfaceRequest, VolSurfaceResponse,
    )
    from api.app.services.pricing_service import compute_price, compute_heatmap, compute_monte_carlo
    from api.app.services.volatility_service import compute_volatility_surface
except ImportError:
    from app.schemas.pricing import (
        PricingRequest, PricingResponse,
        HeatmapRequest, HeatmapResponse,
        MonteCarloRequest, MonteCarloResponse,
        VolSurfaceRequest, VolSurfaceResponse,
    )
    from app.services.pricing_service import compute_price, compute_heatmap, compute_monte_carlo
    from app.services.volatility_service import compute_volatility_surface

router = APIRouter(tags=["pricing"])


@router.post("/price", response_model=PricingResponse)
async def price_option(request: PricingRequest):
    result = await run_in_threadpool(lambda: compute_price(model_name=request.model, **request.to_engine_kwargs()))
    return result


@router.post("/heatmap", response_model=HeatmapResponse)
async def compute_heatmap_endpoint(request: HeatmapRequest):
    result = await run_in_threadpool(lambda: compute_heatmap(
        K=request.K, T=request.T, r=request.r, q=request.q,
        borrow_cost=request.borrow_cost,
        spot_range=request.spot_range.model_dump(),
        vol_range=request.vol_range.model_dump(),
    ))
    return result


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def monte_carlo(request: MonteCarloRequest):
    result = await run_in_threadpool(lambda: compute_monte_carlo(
        S=request.S, K=request.K, T=request.T, r=request.r,
        sigma=request.sigma, paths=request.paths,
        option_type=request.option_type, q=request.q,
        borrow_cost=request.borrow_cost,
    ))
    return result


@router.post("/volatility-surface", response_model=VolSurfaceResponse)
async def volatility_surface(request: VolSurfaceRequest):
    result = await run_in_threadpool(lambda: compute_volatility_surface(
        request.ticker, request.strikes, request.expirations, request.r,
    ))
    return result
