"""A self-contained backtest: synthetic markets, a noisy model, real fee tables.

    python examples/backtest_example.py

No network, no archive, no keys. It builds labelled rows the same way
`predkit.label` does (through `backtest.label_row`, which refuses a label
written before resolution or from a source the venue does not settle on),
attaches a `signal` from a toy model, and scores `simple_taker` with
`backtest.run`: holds to resolution, charges the Kalshi taker fee, and
compares the result with shuffled-label controls.

The toy world is rigged so the model knows something the book does not:
each market's true probability is drawn between 0.3 and 0.7, the book sits
near 50 cents as if it had not priced the news, and the model sees the
truth plus a little noise. The shuffled-label control should come out near
minus the fees and the half-spread (the book is fair against the pooled base
rate), and the real
result should beat nearly every shuffle. If your own result does not beat
its shuffles, the "edge" is not in the labels. Swap `signal_for`
for your own model and `synthetic_rows` for `backtest.read_rows(...)` on
rows from `predkit.label` to backtest for real.
"""

from __future__ import annotations

import random
import sys
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predkit.backtest import WindowRow, label_row, run  # noqa: E402
from predkit.schema import Contract, utc  # noqa: E402
from predkit.strategies import load  # noqa: E402

CENT = Decimal("0.01")


def clamp(p: float) -> Decimal:
    return Decimal(str(min(0.95, max(0.05, round(p, 2))))).quantize(CENT)


def synthetic_rows(n: int = 400, seed: int = 7) -> List[WindowRow]:
    rng = random.Random(seed)
    start = utc(2026, 1, 1, 0, 0)
    rows = []
    for i in range(n):
        resolves_at = start + timedelta(hours=6 * i)
        contract = Contract("kalshi", f"EXAMPLE-{i:04d}", "synthetic yes/no", resolves_at,
                            "kalshi:EXAMPLE", "default")
        truth = rng.uniform(0.3, 0.7)
        quoted = clamp(0.5 + rng.gauss(0, 0.01))            # a market that has not priced the news
        bid, ask = quoted - CENT, quoted + CENT
        row = label_row(contract, resolved_yes=rng.random() < truth,
                        resolution_source=contract.resolution_source,
                        now_ms=contract.resolves_at_ms + 60_000,
                        opens_at_ms=contract.resolves_at_ms - 3 * 3600 * 1000,
                        entry_price=ask, entry_bid=bid,
                        extra={"truth": f"{truth:.4f}"})
        rows.append(row)
    return rows


def signal_for(row: WindowRow, rng: random.Random) -> Decimal:
    """The toy model: the truth plus a little noise. Replace with your own."""
    return clamp(float(row.extra["truth"]) + rng.gauss(0, 0.03))


def main() -> int:
    rng = random.Random(11)
    rows = [replace(r, signal=signal_for(r, rng)) for r in synthetic_rows()]
    strategy = load("simple_taker")
    result = run(rows, strategy.decide, shuffles=200, seed=0)
    print(f"simple_taker on {len(rows)} synthetic Kalshi rows")
    for line in result.lines():
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
