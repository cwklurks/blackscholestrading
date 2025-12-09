import yfinance as yf
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
from .provider import MarketDataProvider

class YahooProvider(MarketDataProvider):
    """Yahoo Finance implementation of MarketDataProvider."""

    def get_history(self, ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        if not ticker or not ticker.strip():
            return None, {"error": "Invalid ticker: empty string"}
            
        try:
            stock = yf.Ticker(ticker.strip().upper())
            hist = stock.history(period=period)
            info = stock.info
            
            if hist is None or hist.empty:
                return None, {"error": f"No data found for ticker '{ticker}'"}
                
            return hist, info
            
        except Exception as exc:
            # We return None, error_dict here to allow the caller to handle the error
            # In a more robust system, we might raise custom exceptions
            return None, {"error": str(exc)}

    def get_chain(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        if not ticker or not ticker.strip():
            return None, None
            
        try:
            stock = yf.Ticker(ticker.strip().upper())
            expirations = stock.options

            if not expirations:
                return None, None

            all_options = []
            # Limit to first 5 expirations as in the original code
            # In a real app we might want to configure this
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
                    continue

            if all_options:
                options_df = pd.concat(all_options, ignore_index=True)
                return options_df, list(expirations)
                
            return None, None
            
        except Exception:
            return None, None

