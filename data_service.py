"""Data fetching services for the Black-Scholes Trader app."""

from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[dict], datetime]:
    """Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: History period (e.g., '1y', '6mo', '1mo')
        
    Returns:
        Tuple of (history DataFrame, info dict, fetch timestamp)
    """
    fetched_at = datetime.utcnow()
    
    if not ticker or not ticker.strip():
        return None, {"error": "Invalid ticker: empty string"}, fetched_at
        
    try:
        stock = yf.Ticker(ticker.strip().upper())
        hist = stock.history(period=period)
        info = stock.info
        
        if hist is None or hist.empty:
            return None, {"error": f"No data found for ticker '{ticker}'"}, fetched_at
            
        return hist, info, fetched_at
        
    except ConnectionError:
        return None, {"error": "Network error: Unable to connect to Yahoo Finance"}, fetched_at
    except TimeoutError:
        return None, {"error": "Request timed out. Please try again."}, fetched_at
    except Exception as exc:
        error_msg = str(exc)
        if "rate limit" in error_msg.lower():
            return None, {"error": "Rate limited by Yahoo Finance. Please wait and try again."}, fetched_at
        return None, {"error": f"Failed to fetch data: {error_msg}"}, fetched_at


@st.cache_data(show_spinner=False)
def fetch_options_chain(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], datetime]:
    """Fetch options chain data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (options DataFrame, list of expiration dates, fetch timestamp)
    """
    fetched_at = datetime.utcnow()
    
    if not ticker or not ticker.strip():
        return None, None, fetched_at
        
    try:
        stock = yf.Ticker(ticker.strip().upper())
        expirations = stock.options

        if not expirations:
            return None, None, fetched_at

        all_options = []

        for exp in expirations[:5]:
            try:
                opt = stock.option_chain(exp)
                calls = opt.calls.copy()
                puts = opt.puts.copy()

                calls['expiration'] = exp
                calls['type'] = 'Call'
                puts['expiration'] = exp
                puts['type'] = 'Put'

                all_options.append(calls)
                all_options.append(puts)
            except Exception:
                # Skip this expiration if there's an issue
                continue

        if all_options:
            options_df = pd.concat(all_options, ignore_index=True)
            return options_df, list(expirations), fetched_at
            
        return None, None, fetched_at
        
    except Exception:
        return None, None, fetched_at


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

