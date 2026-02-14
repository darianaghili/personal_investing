from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    max_weight: float = Field(default=0.2, gt=0.0, le=1.0)
    risk_aversion_lambda: float = Field(default=5.0, ge=0.0)
    max_positions: int = Field(default=10, gt=0)
    top_n: int = Field(default=50, gt=0)
    min_observations: int = Field(default=12, gt=0)
    cache_dir: Path = Path("data/cache")
    results_dir: Path = Path("results")
    universe_file: Path = Path("ETF_universe.csv")
    ff5_cache_file: Path = Path("data/cache/ff5_monthly.parquet")


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return AppConfig()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)
