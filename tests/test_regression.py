import numpy as np
import pandas as pd

from personal_investing.regression import run_ff5_regression


def test_ff5_regression_outputs_scalars():
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(0)
    ff = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.005, 0.02, len(idx)),
            "SMB": rng.normal(0.0, 0.01, len(idx)),
            "HML": rng.normal(0.0, 0.01, len(idx)),
            "RMW": rng.normal(0.0, 0.01, len(idx)),
            "CMA": rng.normal(0.0, 0.01, len(idx)),
            "RF": np.full(len(idx), 0.001),
        },
        index=idx,
    )
    port = 0.002 + 0.8 * ff["Mkt-RF"] + ff["RF"] + rng.normal(0, 0.01, len(idx))
    out = run_ff5_regression(port, ff)
    assert isinstance(out["alpha_monthly"], float)
    assert isinstance(out["alpha_tstat"], float)
