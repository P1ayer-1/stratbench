"""Offline, synthetic end-to-end run of perpkit's factor-panel harness.

TEACHING EXAMPLE, NO KNOWN EDGE. The toy world below is rigged: each coin has
a persistent funding level, and shorts collect it. So `carry_7` (short the
coins whose longs pay the most) earns through its FUNDING leg while its price
leg is noise, and it beats its shuffled controls. Momentum has nothing to
find here and should not. On real data, a factor that does not beat its
shuffles, net of cost and with --lag 1, is not an edge.

    python examples/perpkit_factor_example.py

Needs numpy (`pip install .[perps]`). No network, no keys.
"""

from __future__ import annotations

import random
import tempfile
from datetime import date, timedelta
from pathlib import Path

from perpkit.analysis import factor_panel
from perpkit.analysis.panel import write_panel


def synthetic_rows(n_coins: int = 40, n_days: int = 500, seed: int = 7):
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    rows = []
    for c in range(n_coins):
        symbol = "COIN{:02d}-USDT".format(c)
        level = rng.gauss(2.0, 4.0)           # this coin's persistent funding, bps/day
        price = 100.0
        for d in range(n_days):
            ret = rng.gauss(0.0, 0.01)          # a random walk: nothing to forecast
            open_, price = price, price * (1.0 + ret)
            rows.append({
                "date": (start + timedelta(days=d)).isoformat(),
                "symbol": symbol,
                "open": open_, "high": max(open_, price) * 1.01,
                "low": min(open_, price) * 0.99, "close": price,
                "quote_volume": 5e7, "trades": 10_000, "taker_buy_frac": 0.5,
                "minutes": 1440, "rv_bps": 300.0,
                "funding_bps": level + rng.gauss(0.0, 0.5),
                "funding_periods": 3,
            })
    return rows


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        panel = Path(tmp) / "toy-daily.csv"
        write_panel(synthetic_rows(), panel)
        return factor_panel.main([
            "--panel", str(panel),
            "--factors", "carry_7,mom_14",
            "--hold-days", "7", "--top-frac", "0.2",
            "--cost-bps", "6", "--lag", "1",
            "--min-volume", "1e6", "--min-history", "30",
            "--control-seeds", "50",
            "--detail", "carry_7",
        ])


if __name__ == "__main__":
    raise SystemExit(main())
