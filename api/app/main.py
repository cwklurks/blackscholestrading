"""FastAPI entrypoint for the Black-Scholes Pricing Engine."""
import asyncio
import os
import sys
from pathlib import Path

# Ensure api/ is on sys.path so bare imports (models.*, utils.*, etc.) resolve
# whether running from project root or from within api/.
_api_root = str(Path(__file__).resolve().parent.parent)
if _api_root not in sys.path:
    sys.path.insert(0, _api_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

try:
    from api.app.routers import pricing, market, backtest
except ImportError:
    from app.routers import pricing, market, backtest

app = FastAPI(
    title="Black-Scholes Pricing Engine",
    description="Production-grade options pricing with Heston, GARCH, and Bates models",
    version="1.0.0",
)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Timeout middleware - 30-second request limit."""

    async def dispatch(self, request, call_next):
        try:
            # NOTE: call_next cannot be truly cancelled. The 408 is returned to the
            # client, but the downstream handler continues executing. Actual
            # computation cancellation must be handled at the service layer.
            return await asyncio.wait_for(call_next(request), timeout=30.0)
        except asyncio.TimeoutError:
            return JSONResponse(
                {"detail": "Request timeout (30s limit)"}, status_code=408
            )


app.add_middleware(TimeoutMiddleware)

# CORS - configurable via environment variable
cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pricing.router, prefix="/api")
app.include_router(market.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")


MODEL_REGISTRY = [
    {
        "name": "Black-Scholes",
        "description": "Closed-form analytical solution (European options)",
        "params": {},
    },
    {
        "name": "Binomial (American)",
        "description": "CRR binomial tree for American options",
        "params": {
            "binomial_steps": {
                "type": "int",
                "default": 150,
                "description": "Number of tree steps",
            },
        },
    },
    {
        "name": "Heston MC",
        "description": "Heston stochastic volatility model via Monte Carlo",
        "params": {
            "heston_kappa": {
                "type": "float",
                "default": 1.5,
                "description": "Mean reversion speed",
            },
            "heston_theta": {
                "type": "float",
                "default": 0.04,
                "description": "Long-run variance",
            },
            "heston_rho": {
                "type": "float",
                "default": -0.7,
                "description": "Price-vol correlation",
            },
            "heston_vol_of_vol": {
                "type": "float",
                "default": 0.3,
                "description": "Volatility of volatility",
            },
            "mc_paths": {
                "type": "int",
                "default": 4000,
                "description": "Simulation paths (max 50000)",
            },
            "mc_steps": {
                "type": "int",
                "default": 120,
                "description": "Time steps (max 500)",
            },
        },
    },
    {
        "name": "GARCH MC",
        "description": "GARCH(1,1) volatility model via Monte Carlo",
        "params": {
            "garch_alpha0": {
                "type": "float",
                "default": 2e-6,
                "description": "GARCH constant",
            },
            "garch_alpha1": {
                "type": "float",
                "default": 0.08,
                "description": "ARCH coefficient",
            },
            "garch_beta1": {
                "type": "float",
                "default": 0.9,
                "description": "GARCH coefficient",
            },
            "mc_paths": {
                "type": "int",
                "default": 4000,
                "description": "Simulation paths (max 50000)",
            },
        },
    },
    {
        "name": "Bates Jump-Diffusion",
        "description": "Jump-diffusion model for tail risk pricing",
        "params": {
            "jump_lambda": {
                "type": "float",
                "default": 0.1,
                "description": "Jump intensity",
            },
            "jump_mu": {
                "type": "float",
                "default": -0.05,
                "description": "Mean jump size",
            },
            "jump_delta": {
                "type": "float",
                "default": 0.2,
                "description": "Jump volatility",
            },
            "mc_paths": {
                "type": "int",
                "default": 4000,
                "description": "Simulation paths (max 50000)",
            },
            "mc_steps": {
                "type": "int",
                "default": 120,
                "description": "Time steps (max 500)",
            },
        },
    },
]


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models": [m["name"] for m in MODEL_REGISTRY],
    }


@app.get("/api/models")
async def list_models():
    """List all available pricing models with their parameters."""
    return MODEL_REGISTRY
