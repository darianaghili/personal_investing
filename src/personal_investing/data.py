from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class YFinanceDataProvider:
    """
    Robust Yahoo Finance adjusted close downloader with:
    - Per-ticker parquet caching
    - Retry logic
    - Graceful failure handling
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    # ---------------------------
    # Public API
    # ---------------------------

    def get_adjusted_close(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        all_series: list[pd.Series] = []
        failed: list[str] = []

        for ticker in tickers:
            ticker = str(ticker).strip().upper()
            if not ticker:
                continue

            try:
                series = self._get_one_ticker(ticker, start, end)
                if series is None or series.empty:
                    failed.append(ticker)
                    continue

                windowed = series.loc[
                    (series.index.date >= start)
                    & (series.index.date <= end)
                ]

                if windowed.empty:
                    failed.append(ticker)
                    continue

                windowed.name = ticker
                all_series.append(windowed)

            except Exception as e:
                logger.warning(f"Unexpected error for {ticker}: {e}")
                failed.append(ticker)
                continue

        if failed:
            logger.warning(f"Skipped {len(failed)} tickers due to download issues.")

        if not all_series:
            raise ValueError("No valid price data downloaded for any tickers.")

        prices = pd.concat(all_series, axis=1).sort_index()
        return prices

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _get_one_ticker(
        self, ticker: str, start: date, end: date
    ) -> Optional[pd.Series]:

        cached = self._load_cached(ticker)

        if (
            cached is not None
            and not cached.empty
            and cached.index.min().date() <= start
            and cached.index.max().date() >= end
        ):
            return cached

        dl = self._download_with_retries(ticker, start, end)

        if dl is None or dl.empty:
            return cached

        if cached is not None and not cached.empty:
            merged = pd.concat([cached, dl]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
        else:
            merged = dl

        self._save_cached(ticker, merged)
        return merged

    def _download_with_retries(
        self, ticker: str, start: date, end: date
    ) -> Optional[pd.Series]:

        delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Downloading data for {ticker} (attempt {attempt})")
                df = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )

                if df is None or df.empty:
                    raise ValueError("Empty download")

                if "Adj Close" in df.columns:
                    s = df["Adj Close"].dropna()
                elif "Close" in df.columns:
                    s = df["Close"].dropna()
                else:
                    raise ValueError("No price columns returned")

                s.index = pd.to_datetime(s.index)
                s.name = ticker
                return s.astype(float)

            except Exception as e:
                logger.warning(f"Download failed for {ticker}: {e}")
                time.sleep(delay)
                delay *= 2

        return None

    # ---------------------------
    # Cache
    # ---------------------------

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_")
        return CACHE_DIR / f"{safe}.parquet"

    def _load_cached(self, ticker: str) -> Optional[pd.Series]:
        path = self._cache_path(ticker)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception as e:
            logger.warning(f"Failed to read cache for {ticker}: {e}")
            return None

    def _save_cached(self, ticker: str, series: pd.Series) -> None:
        try:
            df = series.to_frame(name=ticker)
            df.to_parquet(self._cache_path(ticker))
        except Exception as e:
            logger.warning(f"Failed to write cache for {ticker}: {e}")


# ---------------------------------
# Monthly Returns Helper
# ---------------------------------

def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily prices to monthly returns.
    """
    monthly_prices = prices.resample("M").last()
    returns = monthly_prices.pct_change().dropna(how="all")
    return returns
