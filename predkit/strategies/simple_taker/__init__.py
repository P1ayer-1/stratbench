"""Example taker: buy the side your fair value says is cheap, net of the fee.

    plan.py     compare a fair probability to the touch, net of fees.  SENDS NOTHING.
    execute.py  one limit order at the touch; dry by default.
    monitor.py  the fills against the plan, from the venue.  READ ONLY.

This is a teaching example, not a strategy with a known edge. It crosses the
spread (pays the taker fee), holds one position per market to settlement,
and trades only when

    fair - ask - taker fee - builder fee >= min_edge        (buy YES)
    (1 - fair) - (1 - bid) - fees        >= min_edge        (buy NO)

`fair` is the part a user supplies. The runner hands it in as
`Context.fair`, or in `Context.reference` as one of:

    model_prob                       your own model's P(YES)
    sharp_prob                       a probability from another market
    sharp_yes_odds, sharp_no_odds    two-way decimal odds, de-vigged here

Without one the planner refuses, with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from predkit.backtest import Decision, WindowRow
from predkit.ledger import OwnLedger, RunLog
from predkit.strategies.simple_taker.execute import ExecutionResult, execute_take
from predkit.strategies.simple_taker.monitor import MonitorReport, monitor_take
from predkit.strategies.simple_taker.plan import MIN_EDGE, TakePlan, decide_row, devig, plan_take


@dataclass
class TakerBot:
    name: str = "simple_taker"
    min_edge: Decimal = MIN_EDGE
    size: Decimal = Decimal("10")
    ledger: OwnLedger = field(default_factory=OwnLedger)
    log: RunLog = field(default_factory=lambda: RunLog(None))

    def plan(self, context) -> TakePlan:
        return plan_take(context, min_edge=self.min_edge, size=self.size)

    def execute(self, plan: TakePlan, venue: Any, *, dry_run: bool = True) -> ExecutionResult:
        return execute_take(plan, venue, self.ledger, self.log, dry_run=dry_run)

    def monitor(self, venue: Any, market_id: str) -> MonitorReport:
        return monitor_take(venue, self.ledger, market_id)

    def decide(self, row: WindowRow) -> Optional[Decision]:
        return decide_row(row, min_edge=self.min_edge, size=self.size)


__all__ = ["ExecutionResult", "MonitorReport", "TakePlan", "TakerBot", "devig", "plan_take"]
