from __future__ import annotations

from datetime import date

from loguru import logger

from personal_investing.config import load_config
from personal_investing.dates import most_recent_rebalance_date, quarter_label
from personal_investing.rebalance import run_rebalance


def main() -> None:
    cfg = load_config()
    rebalance_date = most_recent_rebalance_date(date.today())
    qlabel = quarter_label(rebalance_date)
    weight_file = cfg.results_dir / f"weights_{qlabel}.csv"
    if weight_file.exists():
        logger.info(f"Rebalance already exists for {qlabel}: {weight_file}")
        return
    logger.info(f"Running rebalance for {qlabel} ({rebalance_date})")
    run_rebalance(rebalance_date)


if __name__ == "__main__":
    main()
