import numpy as np
import pandas as pd

from personal_investing.optimizer import pragmatic_cardinality_mv


def test_optimizer_constraints():
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(
        rng.normal(0.01, 0.05, size=(60, 20)),
        columns=[f"ETF{i:02d}" for i in range(20)],
    )
    w = pragmatic_cardinality_mv(
        rets,
        risk_aversion_lambda=3.0,
        max_weight=0.2,
        max_positions=10,
    )
    assert (w >= -1e-8).all()
    assert abs(w.sum() - 1.0) < 1e-5
    assert len(w) <= 10
    assert (w <= 0.2001).all()
