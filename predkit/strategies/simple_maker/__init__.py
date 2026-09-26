"""Example maker: rest one tick inside the touch when your fair value says
the resting side is cheap, never take.

    plan.py     where to rest, and every reason not to.  SENDS NOTHING.
    execute.py  post, cancel, exit; dry by default; closes only its own fills.
    monitor.py  the venue's orders and fills against the ledger.  READ ONLY.

This is a teaching example, not a strategy with a known edge. It shows the
three verbs on a resting order: the planner posts one tick inside the touch
so the order is alone at its level (which keeps the optimistic fill bound
honest), refuses when the feed is stale or resolution is near, and exits at
the touch when fair moves through the position's cost by `stop`.

Why a maker example at all: on the fee curves in `fees.py` a taker pays up
to 1.75 cents a contract at 50 cents, while makers pay nothing on most tiers
(and on Polymarket are paid a rebate share, not modelled here). Any edge that
is smaller than the taker fee can only be harvested passively, and passive
fills are exactly what `replay.MakerFillModel` brackets.

`fair` is supplied by the runner (`Context.fair`): the probability of YES
from the user's own model, for example of the settlement rule given where a
leader feed (`venues/binance_reference.py`) is now. That is the part of this
strategy a user replaces. Without it the planner refuses, with the reason.

On markets that settle on an average over the final minute (Polymarket's
5-minute crypto markets settle on a 60-second TWAP), the last minute is a
different game; the risk veto's resolution window refuses new risk inside it
and this strategy adds nothing there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, List, Optional

from predkit.backtest import Decision, WindowRow
from predkit.strategies.simple_maker.execute import ExecutionResult, cancel_own, execute_quote
from predkit.strategies.simple_maker.monitor import MonitorReport, monitor_quote
from predkit.strategies.simple_maker.plan import EDGE, STOP, QuotePlan, decide_row, plan_quote
from predkit.ledger import OwnLedger, RunLog


@dataclass
class MakerBot:
    name: str = "simple_maker"
    edge: Decimal = EDGE
    stop: Decimal = STOP
    size: Decimal = Decimal("5")
    ttl_ms: int = 10_000
    ledger: OwnLedger = field(default_factory=OwnLedger)
    log: RunLog = field(default_factory=lambda: RunLog(None))

    def plan(self, context) -> QuotePlan:
        return plan_quote(context, edge=self.edge, stop=self.stop, size=self.size)

    def execute(self, plan: QuotePlan, venue: Any, *, dry_run: bool = True) -> ExecutionResult:
        return execute_quote(plan, venue, self.ledger, self.log, dry_run=dry_run)

    def cancel_all(self, venue: Any, *, dry_run: bool = True) -> ExecutionResult:
        return cancel_own(venue, self.ledger, self.log, dry_run=dry_run)

    def monitor(self, venue: Any, market_id: str) -> MonitorReport:
        return monitor_quote(venue, self.ledger, market_id)

    def decide(self, row: WindowRow) -> Optional[Decision]:
        return decide_row(row, edge=self.edge, size=self.size)


__all__ = ["ExecutionResult", "MakerBot", "MonitorReport", "QuotePlan", "plan_quote"]
