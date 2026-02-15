# personal_investing/data.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger

# Optional dependency (recommended)
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None


DateLike = Union[str, date, datetime, pd.Timestamp]


def _to_date(d: DateLike) -> date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    # assume string
    return pd.to_datetime(d).date()


def _to_timestamp(d: DateLike) -> pd.Timestamp:
    if isinstance(d, pd.Timestamp):
        return d.normalize()
    if isinstance(d, datetime):
        return pd.Timestamp(d).normalize()
    if isinstance(d, date):
        return pd.Timestamp(d).normalize()
    return pd.Timestamp(pd.to_datetime(d)).normalize()


@dataclass(frozen=True)
class CacheConfig:
    cache_dir: Path
    ttl_days: int = 7  # refresh cached history if older than this


def _default_cache_dir() -> Path:
    # ~/.cache/personal_investing (Linux/macOS); Windows still works
    return Path.home() / ".cache" / "personal_investing"


CACHE = CacheConfig(cache_dir=_default_cache_dir(), ttl_days=7)


def _ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def _sanitize_symbol(symbol: str) -> str:
    # File-safe, but keep it human-readable
    return (
        symbol.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("^", "_")
    )


def _cache_path(symbol: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_sanitize_symbol(symbol)}.parquet"


def _is_cache_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    try:
        age_seconds = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age_seconds <= ttl_days * 24 * 60 * 60


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        # normalize index/columns expectations
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        logger.warning(f"Failed reading cache {path}: {e}")
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        out = df.copy()
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.copy()
            out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
        _ensure_cache_dir(path.parent)
        out.to_parquet(path)
    except Exception as e:
        logger.warning(f"Failed writing cache {path}: {e}")


def _download_history_yfinance(symbol: str, start: date, end: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed/available. Install it with: pip install yfinance"
        )

    # yfinance end is exclusive-ish depending on API; add a buffer day
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)

    # auto_adjust=False ensures "Adj Close" exists; actions=True captures splits/divs if needed
    df = yf.download(
        tickers=symbol,
        start=start_ts,
        end=end_ts,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns a multiindex if tickers is list-like; normalize
    if isinstance(df.columns, pd.MultiIndex):
        # Take the first level as field names
        df.columns = df.columns.get_level_values(-1)

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    # Standardize columns we care about
    keep = []
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            keep.append(c)
    df = df[keep]

    return df


def get_price_history(
    symbol: str,
    start: DateLike,
    end: DateLike,
    *,
    cache_dir: Optional[Path] = None,
    ttl_days: Optional[int] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Returns daily price history for `symbol` between [start, end] inclusive (best effort).
    Uses a local parquet cache to reduce repeated downloads.
    """
    start_d = _to_date(start)
    end_d = _to_date(end)
    if end_d < start_d:
        raise ValueError(f"end < start: {end_d} < {start_d}")

    cache_dir = cache_dir or CACHE.cache_dir
    ttl_days = CACHE.ttl_days if ttl_days is None else ttl_days

    path = _cache_path(symbol, cache_dir)
    cached = _read_cache(path)

    # If cache is fresh and covers requested range, serve it
    if not force_refresh and cached is not None and _is_cache_fresh(path, ttl_days):
        # If it contains the requested dates, return slice
        lo = pd.Timestamp(start_d)
        hi = pd.Timestamp(end_d)
        if cached.index.min() <= lo and cached.index.max() >= hi:
            return cached.loc[lo:hi].copy()

    # Decide download window: if cache exists, only extend missing edges
    download_start = start_d
    download_end = end_d

    if cached is not None and not cached.empty and not force_refresh:
        cached_min = cached.index.min().date()
        cached_max = cached.index.max().date()
        # Extend slightly to avoid off-by-one issues around market holidays
        buffer_days = 5
        if start_d >= cached_min and end_d <= cached_max:
            # Cache covers it but stale or freshness failed; re-download small window
            download_start = max(cached_min, start_d - timedelta(days=buffer_days))
            download_end = min(cached_max, end_d + timedelta(days=buffer_days))
        else:
            # Need to extend beyond cached bounds
            download_start = min(start_d, cached_min) - timedelta(days=buffer_days)
            download_end = max(end_d, cached_max) + timedelta(days=buffer_days)

    # Download
    df_new = pd.DataFrame()
    try:
        df_new = _download_history_yfinance(symbol, download_start, download_end)
    except Exception as e:
        # Keep the log message format you already have
        logger.warning(f"Unexpected error for {symbol}: {e}")
        return pd.DataFrame()

    if df_new.empty:
        return pd.DataFrame()

    # Merge with cache (if any)
    if cached is not None and not cached.empty:
        combined = pd.concat([cached, df_new], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df_new

    # Persist combined cache and then slice request range
    _write_cache(path, combined)

    lo = pd.Timestamp(start_d)
    hi = pd.Timestamp(end_d)
    return combined.loc[lo:hi].copy()


def get_adjusted_close(
    symbol: str,
    asof: DateLike,
    *,
    lookback_days: int = 10,
    cache_dir: Optional[Path] = None,
    ttl_days: Optional[int] = None,
    force_refresh: bool = False,
) -> Optional[float]:
    """
    Returns the adjusted close for `symbol` as of `asof`.
    If `asof` is a non-trading day / missing, walks backward up to `lookback_days`.
    Returns None if not available.
    """
    asof_d = _to_date(asof)
    start_d = asof_d - timedelta(days=lookback_days)
    end_d = asof_d

    df = get_price_history(
        symbol,
        start_d,
        end_d,
        cache_dir=cache_dir,
        ttl_days=ttl_days,
        force_refresh=force_refresh,
    )

    if df.empty:
        return None

    # Prefer Adj Close, fall back to Close
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        return None

    # Find last available value on/before asof
    target = _to_timestamp(asof_d)
    df2 = df.loc[:target]
    if df2.empty:
        return None

    val = df2[price_col].dropna()
    if val.empty:
        return None

    return float(val.iloc[-1])


def get_adjusted_close_series(
    symbols: list[str],
    asof: DateLike,
    *,
    lookback_days: int = 10,
    cache_dir: Optional[Path] = None,
    ttl_days: Optional[int] = None,
    force_refresh: bool = False,
) -> pd.Series:
    """
    Convenience: vectorized-ish fetch of adjusted closes for many tickers.
    Returns a Series indexed by symbol (missing -> NaN).
    """
    out: dict[str, float] = {}
    for s in symbols:
        try:
            px = get_adjusted_close(
                s,
                asof,
                lookback_days=lookback_days,
                cache_dir=cache_dir,
                ttl_days=ttl_days,
                force_refresh=force_refresh,
            )
            out[s] = float("nan") if px is None else float(px)
        except Exception as e:
            logger.warning(f"Unexpected error for {s}: {e}")
            out[s] = float("nan")
    return pd.Series(out, name=pd.Timestamp(_to_date(asof)))


# Optional: a tiny self-test hook you can run directly
if __name__ == "__main__":
    # Example:
    # python -m personal_investing.data
    test_symbols = ["SPY", "VTI", ".DJI"]
    asof = "2025-02-15"
    s = get_adjusted_close_series(test_symbols, asof)
    print(s)
