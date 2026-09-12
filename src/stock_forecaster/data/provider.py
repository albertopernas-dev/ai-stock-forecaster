"""Market data provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd


class MarketDataProvider(ABC):
    """Provide normalized historical market prices."""

    @abstractmethod
    def download_prices(
        self,
        tickers: Sequence[str],
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> pd.DataFrame:
        """Download daily prices in the project's internal long format."""
        raise NotImplementedError
