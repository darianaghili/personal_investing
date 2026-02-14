from __future__ import annotations

from datetime import date

import pandas as pd
import pandas_market_calendars as mcal


def quarter_start(asof: date) -> date:
    q_month = ((asof.month - 1) // 3) * 3 + 1
    return date(asof.year, q_month, 1)


def quarter_label(dt: date) -> str:
    q = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{q}"


def first_trading_day_of_quarter(asof: date, calendar: str = "NYSE") -> date:
    start = quarter_start(asof)
    cal = mcal.get_calendar(calendar)
    sched = cal.schedule(start_date=start, end_date=start + pd.Timedelta(days=14))
    if sched.empty:
        raise ValueError("Could not determine first trading day for quarter")
    first_day = sched.index[0].date()
    return first_day


def most_recent_rebalance_date(today: date, calendar: str = "NYSE") -> date:
    candidate_asof = today
    for _ in range(8):
        reb = first_trading_day_of_quarter(candidate_asof, calendar=calendar)
        if reb <= today:
            return reb
        prev_quarter_month = ((candidate_asof.month - 1) // 3) * 3 - 2
        year = candidate_asof.year
        if prev_quarter_month <= 0:
            prev_quarter_month += 12
            year -= 1
        candidate_asof = date(year, prev_quarter_month, 1)
    raise ValueError("Unable to determine recent rebalance date")
