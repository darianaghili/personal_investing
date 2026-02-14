from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def _solve_mv(
    returns: pd.DataFrame,
    risk_aversion_lambda: float,
    max_weight: float,
) -> pd.Series:
    mu = returns.mean().values
    sigma = returns.cov().values
    n = len(mu)
    if n == 0:
        return pd.Series(dtype=float)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - risk_aversion_lambda * cp.quad_form(w, sigma))
    constraints = [w >= 0, cp.sum(w) == 1, w <= max_weight]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimization failed")

    weights = np.clip(np.asarray(w.value).ravel(), 0.0, None)
    if weights.sum() <= 0:
        raise RuntimeError("Optimization produced non-positive weights")
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns)


def pragmatic_cardinality_mv(
    returns: pd.DataFrame,
    risk_aversion_lambda: float,
    max_weight: float,
    max_positions: int,
) -> pd.Series:
    stage1 = _solve_mv(returns, risk_aversion_lambda, max_weight)
    selected = stage1.sort_values(ascending=False).head(max_positions)
    stage2_returns = returns[selected.index]
    stage2 = _solve_mv(stage2_returns, risk_aversion_lambda, max_weight)
    nonzero = stage2[stage2 > 1e-6]
    nonzero = nonzero.sort_values(ascending=False).head(max_positions)
    final = nonzero / nonzero.sum()
    return final


def portfolio_stats(monthly_returns: pd.Series) -> dict[str, float]:
    mean_m = float(monthly_returns.mean())
    vol_m = float(monthly_returns.std(ddof=1))
    sharpe_m = mean_m / vol_m if vol_m > 0 else float("nan")
    mean_a = (1 + mean_m) ** 12 - 1
    vol_a = vol_m * np.sqrt(12)
    sharpe_a = mean_a / vol_a if vol_a > 0 else float("nan")
    return {
        "mean_monthly": mean_m,
        "vol_monthly": vol_m,
        "sharpe_monthly": sharpe_m,
        "mean_annual": mean_a,
        "vol_annual": vol_a,
        "sharpe_annual": sharpe_a,
    }
