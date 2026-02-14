import pandas as pd

from personal_investing.selection import select_top_n


def test_select_top_n_returns_sorted_subset():
    cols = [f"ETF{i:02d}" for i in range(60)]
    data = {c: [0.01] * 60 for c in cols}
    data["ETF59"] = [0.03] * 60
    data["ETF58"] = [0.025] * 60
    df = pd.DataFrame(data)
    top = select_top_n(df, 50)
    assert len(top) == 50
    assert top.index[0] == "ETF59"
    assert top.index[1] == "ETF58"
