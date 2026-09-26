"""Backtests scored as money, with the three rules that keep them honest.

**Labels never see the future.** A row for a window is written by
`label_row` only once `now_ms` is past the window's resolution, and its
label comes from the contract's own `resolution_source`. A spot exchange is
not a resolution source: Polymarket's 5-minute markets settle on a Chainlink
60-second TWAP and Kalshi's on its own index, and a backtest labelled from
Binance's last print will show an edge in the settlement minute that the
venue never pays. `label_row` refuses a source that is not the contract's.

**Non-overlapping holds.** A position is held to resolution, so a market is
occupied from entry until it resolves and a second decision on the same
market inside that time is skipped and counted. The number of trades taken
IS the effective sample size; nothing here corrects for overlap because
nothing here overlaps.

**Cost on turnover, control by shuffling.** The entry pays the venue fee
for its role (taker for an aggressive entry, maker for a resting one) plus
the builder fee; settlement is free. The control shuffles the labels across
rows many times and re-scores the same decisions: the MEAN of the shuffles
and the percentile the real result beat are reported, never the best draw.
The control mean should land near minus the cost. If it does not, the cost
model and the decisions disagree and the result is not to be trusted.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from predkit import fees
from predkit.schema import ONE, ZERO, Contract, Outcome, check_price, to_decimal

SPOT_SOURCES = ("binance", "coinbase", "bybit", "okx", "kraken", "spot")


@dataclass(frozen=True)
class WindowRow:
    """One resolved window of one market, with what was observable at entry."""

    venue: str
    market_id: str
    fee_tier: str
    opens_at_ms: int
    resolves_at_ms: int
    resolution_source: str
    resolved_yes: bool
    entry_price: Decimal          # best YES ask at decision time
    entry_bid: Decimal            # best YES bid at decision time
    signal: Optional[Decimal]     # the strategy's fair YES probability, if it had one
    labelled_at_ms: int
    extra: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        row = asdict(self)
        for key in ("entry_price", "entry_bid", "signal"):
            row[key] = None if row[key] is None else str(row[key])
        return json.dumps(row, separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> "WindowRow":
        row = json.loads(line)
        for key in ("entry_price", "entry_bid", "signal"):
            row[key] = None if row[key] is None else Decimal(row[key])
        return WindowRow(**row)


def label_row(contract: Contract, *, resolved_yes: bool, resolution_source: str, now_ms: int,
              opens_at_ms: int, entry_price, entry_bid, signal=None,
              extra: Optional[Dict[str, str]] = None) -> WindowRow:
    """Build a row, refusing anything that would let the label see the future
    or come from somewhere the venue does not settle on."""
    if now_ms < contract.resolves_at_ms:
        raise ValueError(f"{contract.market_id} resolves at {contract.resolves_at_ms} and it is "
                         f"{now_ms}: a row may only be written after resolution")
    source = resolution_source.strip()
    if source != contract.resolution_source:
        raise ValueError(f"{contract.market_id}: label source {source!r} is not the contract's "
                         f"{contract.resolution_source!r}; the venue's own resolution decides")
    if any(name in source.lower() for name in SPOT_SOURCES):
        raise ValueError(f"{contract.market_id}: {source!r} is a spot exchange, not a resolution "
                         "source; the settlement minute is where they differ")
    if opens_at_ms >= contract.resolves_at_ms:
        raise ValueError(f"{contract.market_id}: entry at {opens_at_ms} is not before resolution")
    return WindowRow(
        venue=contract.venue, market_id=contract.market_id, fee_tier=contract.fee_tier,
        opens_at_ms=opens_at_ms, resolves_at_ms=contract.resolves_at_ms,
        resolution_source=source, resolved_yes=bool(resolved_yes),
        entry_price=check_price(entry_price), entry_bid=check_price(entry_bid),
        signal=None if signal is None else check_price(signal),
        labelled_at_ms=now_ms, extra=dict(extra or {}),
    )


def write_rows(path: Path, rows: Iterable[WindowRow]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.to_json() + "\n")
            count += 1
    return count


def read_rows(path: Path) -> List[WindowRow]:
    with open(path, encoding="utf-8") as handle:
        return [WindowRow.from_json(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class Decision:
    """What a strategy would do on a row: buy `size` of `outcome` at `price`,
    hold to resolution. `role` says whether that entry rests or crosses."""

    outcome: Outcome
    price: Decimal
    size: Decimal
    role: str = "taker"


Decide = Callable[[WindowRow], Optional[Decision]]


@dataclass
class Result:
    trades: int = 0
    skipped_overlap: int = 0
    gross: Decimal = ZERO
    fees: Decimal = ZERO
    notional: Decimal = ZERO
    wins: int = 0
    control_mean: Optional[float] = None
    control_percentile: Optional[float] = None
    shuffles: int = 0

    @property
    def net(self) -> Decimal:
        return self.gross - self.fees

    def lines(self) -> List[str]:
        out = [f"trades {self.trades} (skipped for overlap {self.skipped_overlap}), "
               f"notional {self.notional:.2f}",
               f"gross {self.gross:+.2f}  fees {self.fees:.2f}  net {self.net:+.2f}  "
               f"wins {self.wins}/{self.trades}"]
        if self.trades:
            out.append(f"net per contract-dollar {float(self.net) / float(self.notional):+.4f}")
        if self.control_mean is not None:
            out.append(f"shuffled-label control over {self.shuffles}: mean net "
                       f"{self.control_mean:+.2f}, real result beat "
                       f"{self.control_percentile:.0%} of shuffles")
        return out


def _score(rows: List[WindowRow], decisions: List[Optional[Decision]], labels: List[bool]) -> Result:
    result = Result()
    occupied: Dict[str, int] = {}
    for row, decision, resolved_yes in zip(rows, decisions, labels):
        if decision is None:
            continue
        busy_until = occupied.get(row.market_id)
        if busy_until is not None and row.opens_at_ms < busy_until:
            result.skipped_overlap += 1
            continue
        occupied[row.market_id] = row.resolves_at_ms
        won = (decision.outcome is Outcome.YES) == resolved_yes
        payout = decision.size if won else ZERO
        cost = decision.price * decision.size
        fee = fees.total_fee(row.venue, row.fee_tier, decision.role, decision.price, decision.size)
        result.trades += 1
        result.wins += int(won)
        result.gross += payout - cost
        result.fees += fee
        result.notional += cost
    return result


def run(rows: List[WindowRow], decide: Decide, *, shuffles: int = 200, seed: int = 0) -> Result:
    rows = sorted(rows, key=lambda r: (r.opens_at_ms, r.market_id))
    decisions = [decide(row) for row in rows]
    real = _score(rows, decisions, [r.resolved_yes for r in rows])
    if shuffles and real.trades:
        rng = random.Random(seed)
        labels = [r.resolved_yes for r in rows]
        nets: List[float] = []
        for _ in range(shuffles):
            rng.shuffle(labels)
            nets.append(float(_score(rows, decisions, labels).net))
        real.shuffles = shuffles
        real.control_mean = statistics.fmean(nets)
        real.control_percentile = sum(n < float(real.net) for n in nets) / shuffles
    return real


def main(argv: Optional[List[str]] = None) -> int:
    """Score a registered strategy's `decide` over labelled rows.

        python -m predkit.backtest --rows data/rows/kalshi.jsonl --strategy simple_taker

    Rows written by `predkit.label` carry no `signal`; a strategy that needs
    one decides nothing on them. Fill `signal` from your own model (see
    `examples/backtest_example.py`) before expecting trades.
    """
    import argparse

    from predkit.strategies import NAMES, load

    parser = argparse.ArgumentParser(description="Backtest a strategy on labelled rows, net of fees.")
    parser.add_argument("--rows", required=True, help="JSON Lines of WindowRow, e.g. data/rows/kalshi.jsonl")
    parser.add_argument("--strategy", required=True, choices=NAMES)
    parser.add_argument("--shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = read_rows(Path(args.rows))
    with_signal = sum(r.signal is not None for r in rows)
    print(f"{len(rows)} rows, {with_signal} with a signal")
    result = run(rows, load(args.strategy).decide, shuffles=args.shuffles, seed=args.seed)
    for line in result.lines():
        print(line)
    return 0


__all__ = ["Decide", "Decision", "Result", "WindowRow", "label_row", "main", "read_rows", "run", "write_rows"]


if __name__ == "__main__":
    raise SystemExit(main())
