"""Post, cancel and exit. Dry by default; the result says which.

Every intent in a plan is sent post-only; every exit is sent as a plain
limit at the touch (it may take, and pays the taker fee, which the stop
already priced). Order ids are recorded in the ledger before anything else
happens, so a crash between send and record cannot orphan an order this
process will later refuse to recognise as its own.

`cancel_own` cancels the ledger's orders and nothing else. There is no
"cancel all" against the venue here, on purpose.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, List

from predkit.ledger import OwnLedger, RunLog
from predkit.schema import OrderIntent

_dry_ids = itertools.count(1)


@dataclass
class ExecutionResult:
    market_id: str
    dry_run: bool
    problems: List[str] = field(default_factory=list)
    order_ids: List[str] = field(default_factory=list)
    cancelled: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _send(intent: OrderIntent, venue: Any, ledger: OwnLedger, log: RunLog, result: ExecutionResult,
          dry_run: bool) -> None:
    fields = dict(market=intent.contract.market_id, side=intent.side.value, outcome=intent.outcome.value,
                  price=str(intent.price), size=str(intent.size), post_only=intent.post_only)
    if dry_run:
        order_id = f"dry-{next(_dry_ids)}"
        log.write("intent", dry_run=True, order_id=order_id, **fields)
        result.order_ids.append(order_id)
        ledger.record_order(order_id, intent)
        return
    try:
        order_id = venue.place_order(intent)
    except Exception as exc:
        result.problems.append(f"place failed: {type(exc).__name__}: {exc}")
        log.write("place_failed", error=str(exc), **fields)
        return
    ledger.record_order(order_id, intent)
    log.write("placed", order_id=order_id, **fields)
    result.order_ids.append(order_id)


def execute_quote(plan, venue: Any, ledger: OwnLedger, log: RunLog, *, dry_run: bool = True) -> ExecutionResult:
    result = ExecutionResult(plan.market_id, dry_run)
    if not plan.ok:
        result.problems.append("plan refused: " + "; ".join(plan.reasons))
        return result
    for intent in plan.exits:          # getting out comes before getting in
        _send(intent, venue, ledger, log, result, dry_run)
    for intent in plan.intents:
        _send(intent, venue, ledger, log, result, dry_run)
    return result


def cancel_own(venue: Any, ledger: OwnLedger, log: RunLog, *, dry_run: bool = True,
               market_id: str = "") -> ExecutionResult:
    result = ExecutionResult(market_id, dry_run)
    for order_id in ledger.open_order_ids(market_id or None):
        if dry_run or order_id.startswith("dry-"):
            log.write("cancel", dry_run=True, order_id=order_id)
        else:
            try:
                venue.cancel(order_id)
                log.write("cancelled", order_id=order_id)
            except Exception as exc:
                result.problems.append(f"cancel {order_id} failed: {exc}")
                continue
        ledger.forget_order(order_id)
        result.cancelled.append(order_id)
    return result


__all__ = ["ExecutionResult", "cancel_own", "execute_quote"]
