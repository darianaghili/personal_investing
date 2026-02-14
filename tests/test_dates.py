from datetime import date

from personal_investing.dates import first_trading_day_of_quarter, quarter_start


def test_quarter_start():
    assert quarter_start(date(2024, 5, 10)) == date(2024, 4, 1)


def test_first_trading_day_weekend_new_year_2022():
    # Jan 1, 2022 was Saturday; first NYSE trading day was Jan 3, 2022
    assert first_trading_day_of_quarter(date(2022, 1, 15)) == date(2022, 1, 3)


def test_first_trading_day_holiday_new_year_2023():
    # Jan 1, 2023 Sunday and Jan 2 observed holiday; first trading day Jan 3
    assert first_trading_day_of_quarter(date(2023, 2, 1)) == date(2023, 1, 3)
