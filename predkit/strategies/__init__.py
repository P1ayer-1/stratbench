"""One strategy per directory, each with the same three verbs.

    plan     what to do, and every reason not to.  SENDS NOTHING.
    execute  put it on, or leave nothing behind trying.  Dry by default.
    monitor  read it back and score it against the exchange.  READ ONLY.

The split is a safety property: every
sizing and fee question gets answered while the answer is still only text,
and the code that can move money is a separate thing asked for by name.

The contract, verb by verb
--------------------------
**plan** computes intents and cannot send them. There is no code path from a
`plan.py` to a venue's `place_order`, which `tests/test_strategy_contract.py`
checks with a grep. A planner returns its refusals PLURAL, every failing
gate, because the second reason a trade is bad does not stop mattering once
the first is found.

**execute** is dry unless a caller explicitly says otherwise and reports the
same shape either way, so a rehearsal and the real thing differ by one flag.
It owns leg ordering and unwinds, and it accumulates its OWN fills from the
venue and closes only that quantity: closing a position it did not open
is the one thing an executor must never do.

**monitor** has no confirm flag and no path to an order endpoint. It verifies
against the VENUE, not against the plan, because a plan that agrees with
itself has established nothing. `critical` is a bool a pager can key on.

**decide** is the same strategy's verdict on a labelled backtest row, so the
backtest and the live plan share their thresholds by construction.

Why Protocols and no base class
-------------------------------
Strategies can have nothing in common but the verbs: a resting maker on a
5-minute book, a once-a-day limit order held to settlement, a two-leg
cross-venue lock. A base class extracted from any one of them would encode
that one's accidents. So the lifecycle is stated as Protocols, satisfied structurally
and inherited from never; the contract test asserts each strategy conforms
without importing this module.

`Context` is what the runner hands `plan`: everything observable now, and
nothing the strategy could use to send.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from predkit.backtest import Decision, WindowRow
from predkit.schema import Book, Contract, Fill, OrderIntent


@dataclass
class Context:
    contract: Contract
    book: Book
    now_ms: int
    fair: Optional[Decimal] = None            # the strategy's own probability estimate, if any
    reference: Dict[str, Any] = field(default_factory=dict)   # leader feed, odds, other venue
    feed_lag_ms: Optional[float] = None
    position: Decimal = Decimal("0")          # signed YES-equivalent contracts held (own)
    avg_cost: Decimal = Decimal("0")
    own_open_orders: int = 0


@runtime_checkable
class Plan(Protocol):
    market_id: str
    reasons: List[str]
    warnings: List[str]
    intents: List[OrderIntent]

    @property
    def ok(self) -> bool: ...


@runtime_checkable
class Outcome(Protocol):
    market_id: str
    dry_run: bool
    problems: List[str]
    order_ids: List[str]

    @property
    def ok(self) -> bool: ...


@runtime_checkable
class Report(Protocol):
    market_id: str
    alerts: List[str]

    @property
    def critical(self) -> bool: ...


@runtime_checkable
class Strategy(Protocol):
    name: str

    def plan(self, context: Context) -> Plan: ...
    def execute(self, plan: Plan, venue: Any, *, dry_run: bool = True) -> Outcome: ...
    def monitor(self, venue: Any, market_id: str) -> Report: ...
    def decide(self, row: WindowRow) -> Optional[Decision]: ...


NAMES = ("simple_maker", "simple_taker")


def load(name: str, **params: Any) -> Strategy:
    """The registry. Import lazily so a broken strategy cannot break the others.

    To add a strategy: create `strategies/<name>/` with `plan.py`,
    `execute.py`, `monitor.py` and an `__init__.py` exposing a class with
    `name`, `plan`, `execute`, `monitor` and `decide`; add `<name>` to
    NAMES and a branch here. `tests/test_strategy_contract.py` then checks
    it against the Protocols above automatically.
    """
    if name == "simple_maker":
        from predkit.strategies.simple_maker import MakerBot
        return MakerBot(**params)
    if name == "simple_taker":
        from predkit.strategies.simple_taker import TakerBot
        return TakerBot(**params)
    raise KeyError(f"unknown strategy {name!r}; known: {list(NAMES)}")


__all__ = ["Context", "NAMES", "Outcome", "Plan", "Report", "Strategy", "load"]
