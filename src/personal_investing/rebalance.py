from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta
from loguru import logger

from personal_investing.config import load_config
from personal_investing.data import YFinanceDataProvider, monthly_returns
from personal_investing.dates import first_trading_day_of_quarter, quarter_label
from personal_investing.optimizer import portfolio_stats, pragmatic_cardinality_mv
from personal_investing.regression import load_ff5_monthly, run_ff5_regression
from personal_investing.selection import select_top_n
from personal_investing.universe import CSVUniverseProvider


def run_rebalance(asof: date) -> tuple[Path, Path]:
    cfg = load_config()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    rebalance_date = first_trading_day_of_quarter(asof)
    window_start = rebalance_date - relativedelta(years=5)
    window_end = rebalance_date

    universe = CSVUniverseProvider(cfg.universe_file).get_universe()
    prices = YFinanceDataProvider(cfg.cache_dir).get_adjusted_close(
        universe, window_start, window_end
    )
    rets = monthly_returns(prices)
    eligible = rets.count()[rets.count() >= cfg.min_observations].index
    rets = rets[eligible].dropna(axis=1, how="all")

    top = select_top_n(rets, cfg.top_n)
    top_rets = rets[top.index].dropna(how="all")
    weights = pragmatic_cardinality_mv(
        top_rets,
        risk_aversion_lambda=cfg.risk_aversion_lambda,
        max_weight=cfg.max_weight,
        max_positions=cfg.max_positions,
    )

    port_rets = top_rets[weights.index].mul(weights, axis=1).sum(axis=1)
    stats = portfolio_stats(port_rets)

    ff5 = load_ff5_monthly(cfg.ff5_cache_file)
    ff5_metrics = run_ff5_regression(port_rets, ff5)

    qlabel = quarter_label(rebalance_date)
    weights_path = cfg.results_dir / f"weights_{qlabel}.csv"
    report_path = cfg.results_dir / f"report_{qlabel}.md"

    weights.rename("weight").to_csv(weights_path, index_label="ticker")

    top_lines = "\n".join([f"- {k}: {v:.2%}" for k, v in top.items()])
    weight_lines = "\n".join([f"- {k}: {v:.2%}" for k, v in weights.items()])
    report = f"""# Rebalance Report {qlabel}

- Rebalance date: {rebalance_date}
- Window: {window_start} to {window_end}

## Top {len(top)} ETFs by trailing 5Y compounded return
{top_lines}

## Final portfolio weights (<= {cfg.max_positions} ETFs)
{weight_lines}

## In-sample metrics
- Expected return (monthly): {stats['mean_monthly']:.4%}
- Volatility (monthly): {stats['vol_monthly']:.4%}
- Sharpe (monthly): {stats['sharpe_monthly']:.4f}
- Expected return (annualized): {stats['mean_annual']:.4%}
- Volatility (annualized): {stats['vol_annual']:.4%}
- Sharpe (annualized): {stats['sharpe_annual']:.4f}

## Fama-French 5-factor alpha
- Alpha (monthly): {ff5_metrics['alpha_monthly']:.4%}
- Alpha (annualized): {ff5_metrics['alpha_annualized']:.4%}
- Alpha t-stat: {ff5_metrics['alpha_tstat']:.4f}
- Regression observations: {ff5_metrics['n_obs']:.0f}

## Notes / limitations
- Adjusted close from yfinance is used as a proxy for total return.
- Universe may have survivorship bias depending on ETF_universe.csv maintenance.
- Data quality and missing observations can affect rankings and optimization.
- Optimization is a pragmatic 2-stage approximation for cardinality constraints.
"""
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Saved weights to {weights_path}")
    logger.info(f"Saved report to {report_path}")
    return weights_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quarterly ETF rebalance")
    parser.add_argument("--asof", required=True, help="Date inside target quarter (YYYY-MM-DD)")
    args = parser.parse_args()
    asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
    run_rebalance(asof)


if __name__ == "__main__":
    main()
