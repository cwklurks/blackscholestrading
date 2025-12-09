from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_history(self, ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Fetch historical stock data.
        
        Returns:
            Tuple of (DataFrame with history, info dictionary)
        """
        ...

    @abstractmethod
    def get_chain(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """Fetch options chain.
        
        Returns:
            Tuple of (DataFrame with options chain, list of expiration dates)
        """
        ...

