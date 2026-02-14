from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from personal_investing.providers import DataProvider


class YFinanceDataProvider(DataProvider):
    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def _load_cached(self, ticker: str) -> pd.Series | None:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if "adj_close" not in df.columns:
            return None
        s = df["adj_close"]
        s.index = pd.to_datetime(df.index)
        return s.sort_index()

    def _save_cached(self, ticker: str, series: pd.Series) -> None:
        df = pd.DataFrame({"adj_close": series}).sort_index()
        df.to_parquet(self._cache_path(ticker))

    def _download(self, ticker: str, start: date, end: date) -> pd.Series:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            progress=False,
        )
        if df.empty or "Adj Close" not in df.columns:
            return pd.Series(dtype=float)
        series = df["Adj Close"].dropna()
        series.name = ticker
        return series

    def get_adjusted_close(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        all_series: list[pd.Series] = []
        for ticker in tickers:
            cached = self._load_cached(ticker)
            need_download = True
            if cached is not None and not cached.empty:
                if cached.index.min().date() <= start and cached.index.max().date() >= end:
                    need_download = False
                    series = cached
                else:
                    series = cached
            else:
                series = pd.Series(dtype=float)

            if need_download:
                logger.info(f"Downloading data for {ticker}")
                dl = self._download(ticker, start, end)
                if not dl.empty:
                    merged = pd.concat([series, dl]).sort_index()
                    merged = merged[~merged.index.duplicated(keep="last")]
                    self._save_cached(ticker, merged)
                    series = merged

            if not series.empty:
                windowed = series.loc[
                    (series.index.date >= start) & (series.index.date <= end)
                ]
                windowed.name = ticker
                all_series.append(windowed)

        if not all_series:
            return pd.DataFrame()
        prices = pd.concat(all_series, axis=1).sort_index()
        return prices


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    monthly = prices.resample("ME").last().pct_change().dropna(how="all")
    return monthly
