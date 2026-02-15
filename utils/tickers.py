# personal_investing/utils/tickers.py
from typing import List
import os

def load_tickers_from_file(path: str) -> List[str]:
    """
    Returns deduplicated, upper-cased tickers preserving order.
    Accepts simple .txt (one per line) or .csv (first column).
    Ignores blank lines and lines starting with '#'.
    """
    if not path:
        raise ValueError("path must be provided")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tickers file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ext == ".csv":
                # take first column
                token = line.split(",")[0].strip()
            else:
                token = line
            token = token.upper()
            if token:
                tickers.append(token)

    # dedupe while preserving order
    seen = set()
    deduped: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped
