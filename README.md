# personal_investing

Quarterly ETF portfolio rebalancing project (long-only, max 10 ETFs) with two-stage mean-variance optimization and Fama-French 5-factor alpha evaluation.

**Rebalances occur on the first trading day of each quarter.**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Create `ETF_universe.csv` in the repo root with tickers in the first column.

## Run one rebalance

```bash
python -m personal_investing.rebalance --asof 2025-02-15
```

The command computes the quarter containing `--asof`, finds its **first NYSE trading day**, then runs with a trailing 5-year window ending on that rebalance date.

Outputs:
- `results/weights_{YYYYQn}.csv`
- `results/report_{YYYYQn}.md`

## Quarterly scheduling

### Windows Task Scheduler
- Create a basic task that runs on Jan/Apr/Jul/Oct 1st.
- Action program: path to Python executable.
- Arguments: `-m personal_investing.schedule`
- Start in: repository folder.

### Linux/macOS cron

```cron
0 7 1 1,4,7,10 * cd /path/to/personal_investing && /path/to/python -m personal_investing.schedule
```

### Optional GitHub Actions outline
- Trigger via cron on quarter boundaries.
- Set up Python 3.11.
- Install dependencies and run `python -m personal_investing.schedule`.
- Upload `results/*.csv` and `results/*.md` as artifacts or commit via bot.

## Notes
- Price data: yfinance adjusted close proxy for total return.
- FF5 factors: Ken French monthly 5-factor data via `pandas_datareader`.
- Cardinality handled with pragmatic two-stage optimization.
