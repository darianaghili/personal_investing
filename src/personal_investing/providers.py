from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class UniverseProvider(ABC):
    @abstractmethod
    def get_universe(self) -> list[str]:
        """Return ETF universe tickers."""


class DataProvider(ABC):
    @abstractmethod
    def get_adjusted_close(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Return adjusted close prices indexed by date with columns=tickers."""
