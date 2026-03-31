"""Vercel serverless function entry point for the FastAPI backend."""
from app.main import app  # noqa: F401 - Vercel detects the ASGI app
