import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta
from .provider import MarketDataProvider

class MockProvider(MarketDataProvider):
    """Mock implementation of MarketDataProvider for testing/offline use."""

    def get_history(self, ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        # Generate some random price data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='B') # Approx 1 year
        prices = 100 + np.random.randn(len(dates)).cumsum()
        
        df = pd.DataFrame(index=dates)
        df['Open'] = prices
        df['High'] = prices + 2
        df['Low'] = prices - 2
        df['Close'] = prices
        df['Volume'] = 1000000
        
        info = {
            "symbol": ticker,
            "shortName": f"Mock {ticker}",
            "currentPrice": prices[-1],
            "marketCap": 1000000000,
            "sector": "Technology",
            "industry": "Software - Infrastructure"
        }
        
        return df, info

    def get_chain(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        # Generate some random options data
        today = datetime.now()
        expirations = [
            (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            (today + timedelta(days=60)).strftime('%Y-%m-%d'),
            (today + timedelta(days=90)).strftime('%Y-%m-%d')
        ]
        
        options = []
        current_price = 100.0
        strikes = np.arange(80, 120, 5)
        
        for exp in expirations:
            for strike in strikes:
                # Simple mock pricing
                call_price = max(0, current_price - strike) + 5
                put_price = max(0, strike - current_price) + 5
                
                options.append({
                    'contractSymbol': f"{ticker}{exp.replace('-', '')}C{strike}",
                    'lastTradeDate': today,
                    'strike': strike,
                    'lastPrice': call_price,
                    'bid': call_price - 0.5,
                    'ask': call_price + 0.5,
                    'change': 0.0,
                    'percentChange': 0.0,
                    'volume': 100,
                    'openInterest': 500,
                    'impliedVolatility': 0.3,
                    'inTheMoney': current_price > strike,
                    'contractSize': 'REGULAR',
                    'currency': 'USD',
                    'expiration': exp,
                    'type': 'Call'
                })
                
                options.append({
                    'contractSymbol': f"{ticker}{exp.replace('-', '')}P{strike}",
                    'lastTradeDate': today,
                    'strike': strike,
                    'lastPrice': put_price,
                    'bid': put_price - 0.5,
                    'ask': put_price + 0.5,
                    'change': 0.0,
                    'percentChange': 0.0,
                    'volume': 100,
                    'openInterest': 500,
                    'impliedVolatility': 0.3,
                    'inTheMoney': current_price < strike,
                    'contractSize': 'REGULAR',
                    'currency': 'USD',
                    'expiration': exp,
                    'type': 'Put'
                })
                
        df = pd.DataFrame(options)
        return df, expirations

