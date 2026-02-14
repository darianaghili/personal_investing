from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from pandas_datareader import data as pdr


def load_ff5_monthly(cache_path: str | Path) -> pd.DataFrame:
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)

    ds = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")[0]
    ff = ds.copy()
    ff.index = pd.PeriodIndex(ff.index, freq="M").to_timestamp("M")
    ff = ff / 100.0
    ff.to_parquet(cache)
    return ff


def run_ff5_regression(
    portfolio_returns: pd.Series,
    ff5: pd.DataFrame,
) -> dict[str, float]:
    df = pd.DataFrame({"port": portfolio_returns}).join(ff5, how="inner").dropna()
    df["excess"] = df["port"] - df["RF"]
    X = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    X = sm.add_constant(X)
    y = df["excess"]
    model = sm.OLS(y, X).fit()
    alpha = float(model.params["const"])
    tstat = float(model.tvalues["const"])
    alpha_ann = (1 + alpha) ** 12 - 1
    return {
        "alpha_monthly": alpha,
        "alpha_annualized": alpha_ann,
        "alpha_tstat": tstat,
        "n_obs": float(len(df)),
    }
