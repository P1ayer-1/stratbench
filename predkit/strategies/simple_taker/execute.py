"""One order per market, held to settlement. Dry by default; the result says which."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, List

from predkit.ledger import OwnLedger, RunLog

_dry_ids = itertools.count(1)


@dataclass
class ExecutionResult:
    market_id: str
    dry_run: bool
    problems: List[str] = field(default_factory=list)
    order_ids: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def execute_take(plan, venue: Any, ledger: OwnLedger, log: RunLog, *, dry_run: bool = True) -> ExecutionResult:
    result = ExecutionResult(plan.market_id, dry_run)
    if not plan.ok:
        result.problems.append("plan refused: " + "; ".join(plan.reasons))
        return result
    if len(plan.intents) > 1:
        result.problems.append(f"plan carries {len(plan.intents)} intents; this example sends one per market")
        return result
    for intent in plan.intents:
        fields = dict(market=intent.contract.market_id, side=intent.side.value, outcome=intent.outcome.value,
                      price=str(intent.price), size=str(intent.size))
        if dry_run:
            order_id = f"dry-{next(_dry_ids)}"
            log.write("intent", dry_run=True, order_id=order_id, **fields)
        else:
            try:
                order_id = venue.place_order(intent)
            except Exception as exc:
                result.problems.append(f"place failed: {type(exc).__name__}: {exc}")
                log.write("place_failed", error=str(exc), **fields)
                continue
            log.write("placed", order_id=order_id, **fields)
        ledger.record_order(order_id, intent)
        result.order_ids.append(order_id)
    return result


__all__ = ["ExecutionResult", "execute_take"]
