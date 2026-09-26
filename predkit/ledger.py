"""What this process sent and what came back: the executor's own memory.

An executor may only ever close what it opened. A bot that reads the
account's positions at shutdown and flattens everything will close
positions it never put on (a manual trade, another bot's hedge), which is
precisely the behaviour the three-verb split exists to forbid. So the ledger records every order id
this process placed, attributes the venue's fills to those ids and nothing
else, and any fill or position outside that set is a PROBLEM to report,
never something to act on.

The run log is JSON Lines with a receive timestamp per event, one file per
run, so `monitor` can read a run back without the process that wrote it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from predkit.schema import ZERO, Fill, OrderIntent


class RunLog:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.events: List[Dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        row = {"t": int(time.time() * 1000), "event": event, **fields}
        self.events.append(row)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")

    @staticmethod
    def read(path: Path) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


@dataclass
class OwnLedger:
    """Orders this process placed, keyed by venue order id, and the fills the
    venue attributed to them."""

    orders: Dict[str, OrderIntent] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    _seen_fill_ids: set = field(default_factory=set)

    def record_order(self, order_id: str, intent: OrderIntent) -> None:
        self.orders[order_id] = intent

    def forget_order(self, order_id: str) -> None:
        self.orders.pop(order_id, None)

    def reconcile(self, venue_fills: Iterable[Fill]) -> List[Fill]:
        """Attribute venue fills to own orders. Returns the NEW own fills;
        fills on orders this process never placed are ignored here and
        surfaced by `foreign`."""
        new: List[Fill] = []
        for fill in venue_fills:
            key = fill.fill_id or f"{fill.order_id}:{fill.ts_ms}:{fill.price}:{fill.size}"
            if fill.order_id in self.orders and key not in self._seen_fill_ids:
                self._seen_fill_ids.add(key)
                self.fills.append(fill)
                new.append(fill)
        return new

    def foreign(self, venue_fills: Iterable[Fill]) -> List[Fill]:
        return [f for f in venue_fills if f.order_id not in self.orders]

    def position(self, market_id: str) -> Decimal:
        return sum((f.yes_delta for f in self.fills if f.market_id == market_id), ZERO)

    def cost_basis(self, market_id: str) -> Decimal:
        """Dollars paid for what is still held, average-cost."""
        held = ZERO
        cost = ZERO
        for fill in (f for f in self.fills if f.market_id == market_id):
            delta = fill.yes_delta
            if held != ZERO and (held > ZERO) != (delta > ZERO):
                closed = min(abs(delta), abs(held))
                cost -= (cost / abs(held)) * closed if held else ZERO
                held += delta
                if abs(delta) > closed:
                    cost += fill.price * (abs(delta) - closed)
            else:
                held += delta
                cost += fill.price * fill.size
        return cost

    def avg_cost(self, market_id: str) -> Decimal:
        held = self.position(market_id)
        return self.cost_basis(market_id) / abs(held) if held else ZERO

    def open_order_ids(self, market_id: Optional[str] = None) -> List[str]:
        return [oid for oid, intent in self.orders.items()
                if market_id is None or intent.contract.market_id == market_id]


__all__ = ["OwnLedger", "RunLog"]
