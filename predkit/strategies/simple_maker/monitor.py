"""Read the venue back and score it against the ledger. READ ONLY.

No confirm flag, no path to an order endpoint. It asks the venue for the
open orders and fills on the market and compares them with what this
process believes it sent:

  foreign open order     an order on this market the ledger did not place.
                         Reported, left alone. Critical, because the
                         account is doing something this process cannot
                         account for.
  own order missing      the venue no longer shows an order the ledger holds
                         open and no fill explains it: cancelled from
                         elsewhere. Warning.
  new own fill           attributed to the ledger. Informational.
  foreign fill           a fill on an order this process never placed.
                         Critical for the same reason as a foreign order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, List

from predkit.ledger import OwnLedger


@dataclass
class MonitorReport:
    market_id: str
    alerts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    position: Decimal = Decimal("0")
    critical: bool = False


def monitor_quote(venue: Any, ledger: OwnLedger, market_id: str) -> MonitorReport:
    report = MonitorReport(market_id)
    try:
        open_orders = venue.open_orders(market_id)
    except Exception as exc:
        report.alerts.append(f"could not read open orders: {type(exc).__name__}: {exc}")
        report.critical = True
        open_orders = []
    try:
        venue_fills = venue.fills(market_id)
    except Exception as exc:
        report.alerts.append(f"could not read fills: {type(exc).__name__}: {exc}")
        report.critical = True
        venue_fills = []

    venue_ids = {str(o.get("id") or o.get("order_id") or o.get("orderID")) for o in open_orders}
    own_ids = set(ledger.open_order_ids(market_id))
    for order_id in sorted(venue_ids - own_ids):
        report.alerts.append(f"foreign open order {order_id} on {market_id}: not this process's; left alone")
        report.critical = True
    new = ledger.reconcile(venue_fills)
    for fill in new:
        report.notes.append(f"own fill {fill.side.value} {fill.outcome.value} {fill.size} @ {fill.price}")
    filled_ids = {f.order_id for f in ledger.fills}
    for order_id in sorted(own_ids - venue_ids):
        if order_id.startswith("dry-"):
            continue
        if order_id not in filled_ids:
            report.alerts.append(f"own order {order_id} is gone from the venue with no fill: "
                                 "cancelled from elsewhere?")
    for fill in ledger.foreign(venue_fills):
        report.alerts.append(f"foreign fill {fill.fill_id or fill.order_id} on {market_id}: "
                             "an order this process did not place")
        report.critical = True
    report.position = ledger.position(market_id)
    return report


__all__ = ["MonitorReport", "monitor_quote"]
