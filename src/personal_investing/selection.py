from __future__ import annotations

import pandas as pd


def trailing_compounded_returns(monthly_returns_df: pd.DataFrame) -> pd.Series:
    return (1.0 + monthly_returns_df).prod(axis=0) - 1.0


def select_top_n(monthly_returns_df: pd.DataFrame, top_n: int) -> pd.Series:
    compounded = trailing_compounded_returns(monthly_returns_df)
    ranked = compounded.sort_values(ascending=False)
    return ranked.head(top_n)
