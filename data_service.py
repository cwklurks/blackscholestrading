"""Data fetching services for the Black-Scholes Trader app."""

import json
import shutil
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Any
from pathlib import Path

import pandas as pd

from data.provider import MarketDataProvider
from data.yahoo_provider import YahooProvider
from data.mock_provider import MockProvider

# Configuration
CACHE_DIR = Path(".cache")
CACHE_EXPIRY = timedelta(hours=1) # Cache validity duration
USE_MOCK = False # Toggle to True to use MockProvider

def get_provider() -> MarketDataProvider:
    """Factory to get the configured data provider."""
    if USE_MOCK:
        return MockProvider()
    return YahooProvider()

def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cache_path(key: str, extension: str) -> Path:
    """Get the file path for a cache key."""
    return CACHE_DIR / f"{key}.{extension}"

def _load_from_cache(key: str, data_type: str = "parquet") -> Optional[Any]:
    """Load data from cache if it exists and is valid."""
    _ensure_cache_dir()
    path = _get_cache_path(key, data_type)
    meta_path = _get_cache_path(key + "_meta", "json")
    
    if not path.exists() or not meta_path.exists():
        return None
        
    try:
        # Check expiry
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        fetched_at = datetime.fromisoformat(meta['fetched_at'])
        if datetime.utcnow() - fetched_at > CACHE_EXPIRY:
            return None # Expired
            
        if data_type == "parquet":
            return pd.read_parquet(path), fetched_at
        elif data_type == "json":
            with open(path, 'r') as f:
                return json.load(f), fetched_at
        return None
    except Exception:
        return None

def _save_to_cache(key: str, data: Any, data_type: str = "parquet"):
    """Save data to cache."""
    _ensure_cache_dir()
    path = _get_cache_path(key, data_type)
    meta_path = _get_cache_path(key + "_meta", "json")
    
    try:
        fetched_at = datetime.utcnow()
        if data_type == "parquet":
            data.to_parquet(path)
        elif data_type == "json":
            with open(path, 'w') as f:
                json.dump(data, f)
        
        with open(meta_path, 'w') as f:
            json.dump({'fetched_at': fetched_at.isoformat()}, f)
    except Exception as e:
        print(f"Failed to cache data: {e}")

def fetch_stock_data(ticker: str, period: str = "1y", force_refresh: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[dict], datetime]:
    """Fetch historical stock data using the configured provider and caching.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: History period (e.g., '1y', '6mo', '1mo')
        force_refresh: Whether to bypass cache and force a fresh fetch
        
    Returns:
        Tuple of (history DataFrame, info dict, fetch timestamp)
    """
    fetched_at = datetime.utcnow()
    
    if not ticker or not ticker.strip():
        return None, {"error": "Invalid ticker: empty string"}, fetched_at
        
    cache_key_hist = f"{ticker}_{period}_history"
    cache_key_info = f"{ticker}_info"
    
    # Try cache first unless forced refresh
    if not force_refresh:
        cached_hist_data = _load_from_cache(cache_key_hist, "parquet")
        cached_info_data = _load_from_cache(cache_key_info, "json")
        
        if cached_hist_data and cached_info_data:
            hist, hist_ts = cached_hist_data
            info, _ = cached_info_data
            return hist, info, hist_ts

    # Fetch from provider
    provider = get_provider()
    hist, info = provider.get_history(ticker, period)
    
    if hist is None:
        # info might contain error dict
        return None, info if info else {"error": f"No data found for ticker '{ticker}'"}, fetched_at
        
    # Save to cache
    _save_to_cache(cache_key_hist, hist, "parquet")
    if info:
        _save_to_cache(cache_key_info, info, "json")
        
    return hist, info, fetched_at


def fetch_options_chain(ticker: str, force_refresh: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], datetime]:
    """Fetch options chain data using the configured provider and caching.
    
    Args:
        ticker: Stock ticker symbol
        force_refresh: Whether to bypass cache and force a fresh fetch
        
    Returns:
        Tuple of (options DataFrame, list of expiration dates, fetch timestamp)
    """
    fetched_at = datetime.utcnow()
    
    if not ticker or not ticker.strip():
        return None, None, fetched_at
        
    cache_key_opts = f"{ticker}_options"
    cache_key_exps = f"{ticker}_expirations"
    
    # Try cache unless forced refresh
    if not force_refresh:
        cached_opts_data = _load_from_cache(cache_key_opts, "parquet")
        cached_exps_data = _load_from_cache(cache_key_exps, "json")
        
        if cached_opts_data and cached_exps_data:
            opts, opts_ts = cached_opts_data
            exps, _ = cached_exps_data
            return opts, exps, opts_ts
        
    # Fetch from provider
    provider = get_provider()
    opts, exps = provider.get_chain(ticker)
    
    if opts is None:
        return None, None, fetched_at
        
    # Save to cache
    _save_to_cache(cache_key_opts, opts, "parquet")
    if exps:
        _save_to_cache(cache_key_exps, exps, "json")
        
    return opts, exps, fetched_at


def parse_uploaded_prices(upload) -> Optional[pd.Series]:
    """Parse a CSV upload into a price series.
    
    Args:
        upload: Streamlit file upload object
        
    Returns:
        Pandas Series with datetime index and price values, or None on failure
    """
    if upload is None:
        return None
    try:
        df = pd.read_csv(upload)
        date_col = [c for c in df.columns if 'date' in c.lower()]
        price_col = [c for c in df.columns if c.lower() in {'close', 'price'}]
        if not date_col or not price_col:
            return None
        series = pd.Series(df[price_col[0]].values, index=pd.to_datetime(df[date_col[0]]))
        series = series.sort_index()
        return series
    except Exception:
        return None

def clear_cache():
    """Clear the data cache."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
