"""The fills against the plan, from the venue. READ ONLY."""

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


def monitor_take(venue: Any, ledger: OwnLedger, market_id: str) -> MonitorReport:
    report = MonitorReport(market_id)
    try:
        venue_fills = venue.fills(market_id)
    except Exception as exc:
        report.alerts.append(f"could not read fills: {type(exc).__name__}: {exc}")
        report.critical = True
        return report
    for fill in ledger.reconcile(venue_fills):
        report.notes.append(f"own fill {fill.side.value} {fill.outcome.value} {fill.size} @ {fill.price}")
    foreign = ledger.foreign(venue_fills)
    if foreign:
        report.alerts.append(f"{len(foreign)} fill(s) on {market_id} from orders this process did not place")
        report.critical = True
    report.position = ledger.position(market_id)
    intended = sum((i.yes_delta for i in ledger.orders.values() if i.contract.market_id == market_id),
                   Decimal("0"))
    if intended and report.position == Decimal("0"):
        report.notes.append("order resting, nothing filled yet")
    elif intended and abs(report.position) < abs(intended):
        report.notes.append(f"partial: {report.position} of {intended}")
    return report


__all__ = ["MonitorReport", "monitor_take"]
