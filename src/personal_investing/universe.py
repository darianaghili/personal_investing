from __future__ import annotations

from pathlib import Path

import pandas as pd

from personal_investing.providers import UniverseProvider


class CSVUniverseProvider(UniverseProvider):
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def get_universe(self) -> list[str]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Universe file not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if df.empty:
            return []
        first_col = df.columns[0]
        tickers = (
            df[first_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        return tickers
