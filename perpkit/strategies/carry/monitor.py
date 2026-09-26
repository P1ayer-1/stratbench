"""Watches a carry that is already on, and scores realised against predicted.

`carry.py` sizes the position, `carry_executor.py` puts it on. Both finish in
seconds. The trade they produce takes a month, and until now nothing looked at
it again: the plan printed "+170.4 bps over 30 days", the executor printed
"CARRY IS ON", and the two numbers that would settle whether either was true -
funding actually credited, and the delta actually carried - were never read.

That is the gap this closes. It reads both legs back, works out what funding
has really been paid, and compares it to what the plan said. It sends nothing.

Funding, and why it is derived rather than read
-----------------------------------------------
BloFin publishes no account-bills endpoint on this API version:
`/api/v1/account/bills` and `/api/v1/account/bills-archive` both answer "This
operation is not supported", and `/api/v1/asset/bills` covers transfers, not
futures funding. So there is no line item saying "funding, +$0.14".

What there is: `realizedPnl` on the position, which accumulates fees, closed
trade PnL and funding together. Everything in that sum except funding is
separately observable from the fills, so funding falls out of the difference:

    funding = realizedPnl + fees - fillPnl                    (per positionId)

Verified against the position at open, before any funding period had passed:
`realizedPnl` was -0.11975592 and the single perp fill's fee was 0.11975592,
so the derivation returns exactly zero when the true answer is exactly zero.
Signs are taken raw rather than through `abs()`, so a maker rebate reported as
a negative fee still cancels correctly.

**A derived number gets a control.** The same funding is independently
estimated from the public funding-rate history - every settlement between the
open and now, times the notional - and the two are compared. They will not
agree exactly, because the implied one prices every period at today's notional
rather than the notional at each settlement. They should agree closely, and
when they do not the report says so instead of quietly preferring the one it
computed itself.

What a snapshot cannot reconstruct
-----------------------------------
Spreads at the moment of entry. The plan's 25.82 bps round trip was two
spreads plus four fees, and the spreads are gone the instant the book moves.
So the realised round trip here is fees only - measured from the fills, at the
rate actually charged rather than the rate configured - and the exit is priced
at what the entry cost. That understates the true round trip by the spreads.
It is stated in the report rather than hidden, and it is still a better
scoreboard than the plan's, which used a fee tier the account was not on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...risk import RiskLimits

ZERO = Decimal("0")
BPS = Decimal("10000")
MS_PER_DAY = Decimal("86400000")

# The executor tags both legs of one carry with the same id: `carry<hex>p` on
# the perp order and `carry<hex>s` on the spot order. That shared tag is the
# only exact join between a futures position and the spot balance hedging it -
# the alternative is matching on size and timestamp, which stops being unique
# the moment a second carry is opened on the same instrument.
TAG_PREFIX = "carry"

# How much of the notional may sit directional before it is worth saying.
#
# Deliberately NOT the plan's 0.001. That number is a tolerance on LOT
# ROUNDING, measured before fees, and the most common way a carry's delta
# actually breaks is a spot leg short by exactly its fee - which on this
# venue is 0.1% of the base, and lands at 0.0997% of notional. A threshold set
# at 0.001 therefore sits a rounding error away from the single failure it
# most needs to catch, and misses it. This sits below that failure instead:
# a hedge grossed up for its fee carries ~0.0001% and clears it by three
# orders of magnitude, while a hedge that forgot to trips it every time.
MAX_DELTA_SHARE = Decimal("0.0005")

# Below this share of the prediction, realised funding is not noise around the
# forecast, it is a different regime. Break-even moves and the report says by
# how much rather than waiting for the hold to end.
FUNDING_SHORTFALL_SHARE = Decimal("0.5")

# How far the derived and implied funding measures may sit apart before the
# disagreement is the headline rather than the funding.
FUNDING_DIVERGENCE_USD = Decimal("0.05")
FUNDING_DIVERGENCE_SHARE = Decimal("0.25")


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        if value is None or value == "":
            return None
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None


def _d(value: Any, default: str = "0") -> Decimal:
    parsed = _decimal(value)
    return Decimal(default) if parsed is None else parsed


def carry_tag(client_order_id: Optional[str]) -> Optional[str]:
    """The shared id joining a carry's two legs, from either leg's order id.

    `carry9f2cp` and `carry9f2cs` both return `9f2c`. Anything not written by
    the executor returns None rather than a guess, because a mis-join would
    pair a futures position with somebody else's spot fill.
    """
    if not client_order_id or not client_order_id.startswith(TAG_PREFIX):
        return None
    body = client_order_id[len(TAG_PREFIX):]
    if len(body) < 2 or body[-1] not in ("p", "s", "u"):
        return None
    return body[:-1] or None


# --------------------------------------------------------------------------
# what the carry was supposed to be
# --------------------------------------------------------------------------


@dataclass
class Baseline:
    """The position as opened, and the prediction it is being scored against.

    Reconstructable from the exchange rather than only written at open time,
    because a run that persisted nothing must still be scoreable. Everything
    here is observable after the fact: fills carry their price, size and fee,
    the position carries its entry and creation time, and the two legs are
    joined by the executor's shared client-order tag.
    """

    inst_id: str
    opened_at_ms: int
    position_id: str
    tag: Optional[str] = None

    perp_contracts: Decimal = ZERO          # signed; negative for the short
    perp_entry: Decimal = ZERO
    perp_fee_usd: Decimal = ZERO

    spot_ordered: Decimal = ZERO            # what was asked for
    spot_base: Decimal = ZERO               # what survived the fee
    spot_entry: Decimal = ZERO
    spot_cost_usd: Decimal = ZERO
    spot_fee_base: Decimal = ZERO

    notional_usd: Decimal = ZERO
    hold_days: Decimal = Decimal("30")
    planned_funding_per_day_bps: Decimal = ZERO
    planned_liquidation: Optional[Decimal] = None

    @property
    def entry_fee_usd(self) -> Decimal:
        """Both legs' entry fees, at the rate actually charged."""
        return self.perp_fee_usd + self.spot_fee_base * self.spot_entry

    @property
    def round_trip_bps(self) -> Decimal:
        """Entry fees, doubled for the exit. Spreads are not recoverable."""
        if self.notional_usd <= 0:
            return ZERO
        return self.entry_fee_usd * 2 / self.notional_usd * BPS

    @property
    def planned_breakeven_days(self) -> Optional[Decimal]:
        if self.planned_funding_per_day_bps <= 0:
            return None
        return self.round_trip_bps / self.planned_funding_per_day_bps


def reconstruct_baseline(
    position: Dict[str, Any],
    perp_orders: Sequence[Dict[str, Any]],
    spot_orders: Sequence[Dict[str, Any]],
    *,
    planned_funding_per_day_bps: Decimal,
    hold_days: Decimal = Decimal("30"),
    planned_liquidation: Optional[Decimal] = None,
) -> Optional[Baseline]:
    """Rebuild what the carry was, from what the exchange still remembers.

    The perp side comes from the position itself. The spot side is found by
    the executor's shared tag, and only by that: a spot order without a
    matching tag is somebody else's trade, and pairing it here would invent a
    hedge that does not exist.
    """
    position_id = str(position.get("positionId") or "")
    contracts = _decimal(position.get("positions"))
    entry = _decimal(position.get("averagePrice"))
    created = _decimal(position.get("createTime"))
    if contracts is None or entry is None or created is None or not contracts:
        return None

    baseline = Baseline(
        inst_id=str(position.get("instId") or ""),
        opened_at_ms=int(created),
        position_id=position_id,
        perp_contracts=contracts,
        perp_entry=entry,
    )

    opening = [
        order for order in perp_orders
        if str(order.get("positionId") or "") == position_id
        and str(order.get("reduceOnly", "false")).lower() != "true"
    ]
    for order in opening:
        baseline.perp_fee_usd += _d(order.get("fee"))
        tag = carry_tag(order.get("clientOrderId"))
        if tag and baseline.tag is None:
            baseline.tag = tag

    if baseline.tag:
        for order in spot_orders:
            if carry_tag(order.get("clientOrderId")) != baseline.tag:
                continue
            if str(order.get("side", "")).lower() != "buy":
                continue
            filled = _d(order.get("filledSize"))
            price = _d(order.get("averagePrice"))
            fee = _d(order.get("fee"))
            baseline.spot_ordered += filled
            baseline.spot_fee_base += fee
            baseline.spot_cost_usd += filled * price
            baseline.spot_entry = price

    baseline.spot_base = baseline.spot_ordered - baseline.spot_fee_base
    baseline.notional_usd = abs(contracts) * entry
    baseline.hold_days = hold_days
    baseline.planned_funding_per_day_bps = planned_funding_per_day_bps
    baseline.planned_liquidation = planned_liquidation
    return baseline


# --------------------------------------------------------------------------
# one reading
# --------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Both legs as they stand right now, plus the funding derivation's inputs.

    The raw inputs are kept alongside the derived number on purpose. Funding
    is inferred from an accounting identity, and if that identity ever turns
    out to be wrong, a history of snapshots holding only the conclusion could
    not be re-scored. A history holding `realized_pnl`, `fees_usd` and
    `fill_pnl_usd` can.
    """

    at_ms: int
    inst_id: str
    open: bool = True

    perp_contracts: Decimal = ZERO
    perp_mark: Decimal = ZERO
    perp_unrealized_usd: Decimal = ZERO
    liquidation: Optional[Decimal] = None
    margin_usd: Decimal = ZERO
    maintenance_margin_usd: Decimal = ZERO

    spot_base: Decimal = ZERO
    spot_mark: Decimal = ZERO

    realized_pnl_usd: Decimal = ZERO
    fees_usd: Decimal = ZERO
    fill_pnl_usd: Decimal = ZERO

    funding_implied_usd: Decimal = ZERO
    funding_periods: int = 0

    @property
    def funding_booked_usd(self) -> Decimal:
        """Funding credited, from the identity the module docstring derives."""
        return self.realized_pnl_usd + self.fees_usd - self.fill_pnl_usd

    @property
    def actual_mmr(self) -> Optional[Decimal]:
        notional = abs(self.perp_contracts) * self.perp_mark
        if notional <= 0 or self.maintenance_margin_usd <= 0:
            return None
        return self.maintenance_margin_usd / notional


def fills_totals(fills: Iterable[Dict[str, Any]],
                 position_id: str) -> Tuple[Decimal, Decimal]:
    """Fees and closed-trade PnL for one position, from its fills.

    Scoped by `positionId`: an earlier carry on the same instrument that was
    opened and unwound has its own id, and folding its fees in here would
    charge this position for a trade it never made.
    """
    fees = ZERO
    pnl = ZERO
    for fill in fills or []:
        if position_id and str(fill.get("positionId") or "") != position_id:
            continue
        fees += _d(fill.get("fee"))
        pnl += _d(fill.get("fillPnl"))
    return fees, pnl


def implied_funding(rates: Sequence[Dict[str, Any]], *,
                    contracts: Decimal, notional_usd: Decimal,
                    since_ms: int, until_ms: int) -> Tuple[Decimal, int]:
    """Funding the public rate history says a position this size collected.

    The control on the derived number, not a replacement for it. Every
    settlement is priced at today's notional because the notional at each past
    settlement was never recorded, so this drifts as the mark moves - which is
    exactly why it is reported beside the booked figure rather than instead of
    it.

    A short collects when the rate is positive; a long pays. Settlements at or
    before the open do not belong to this position.
    """
    if not contracts:
        return ZERO, 0
    direction = Decimal(-1) if contracts > 0 else Decimal(1)
    total = ZERO
    periods = 0
    for row in rates or []:
        when = _decimal(row.get("fundingTime"))
        rate = _decimal(row.get("fundingRate"))
        if when is None or rate is None:
            continue
        if not (since_ms < int(when) <= until_ms):
            continue
        total += rate * notional_usd * direction
        periods += 1
    return total, periods


def build_snapshot(
    *,
    at_ms: int,
    inst_id: str,
    position: Optional[Dict[str, Any]],
    spot_base: Decimal,
    spot_mark: Decimal,
    fills: Sequence[Dict[str, Any]],
    funding_rates: Sequence[Dict[str, Any]],
    baseline: Baseline,
) -> Snapshot:
    """One reading of both legs, with funding measured two ways."""
    snapshot = Snapshot(at_ms=at_ms, inst_id=inst_id, spot_base=spot_base,
                        spot_mark=spot_mark)

    fees, fill_pnl = fills_totals(fills, baseline.position_id)
    snapshot.fees_usd = fees
    snapshot.fill_pnl_usd = fill_pnl

    if position:
        snapshot.open = True
        snapshot.perp_contracts = _d(position.get("positions"))
        snapshot.perp_mark = _d(position.get("markPrice"))
        snapshot.perp_unrealized_usd = _d(position.get("unrealizedPnl"))
        snapshot.liquidation = _decimal(position.get("liquidationPrice"))
        snapshot.margin_usd = _d(position.get("margin"))
        snapshot.maintenance_margin_usd = _d(position.get("maintenanceMargin"))
        snapshot.realized_pnl_usd = _d(position.get("realizedPnl"))
    else:
        # The perp leg is gone. Its realised PnL went with it, so the funding
        # identity has nothing left to stand on and the report says the carry
        # is off rather than pricing a position that does not exist.
        snapshot.open = False

    notional = abs(snapshot.perp_contracts) * (snapshot.perp_mark or
                                               baseline.perp_entry)
    snapshot.funding_implied_usd, snapshot.funding_periods = implied_funding(
        funding_rates,
        contracts=snapshot.perp_contracts or baseline.perp_contracts,
        notional_usd=notional or baseline.notional_usd,
        since_ms=baseline.opened_at_ms, until_ms=at_ms)
    return snapshot


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


@dataclass
class Alert:
    level: str          # "info" | "warn" | "critical"
    code: str
    message: str


@dataclass
class MonitorReport:
    """Realised against predicted, and everything that would end the hold early."""

    inst_id: str
    days_elapsed: Decimal = ZERO
    open: bool = True

    funding_booked_usd: Decimal = ZERO
    funding_implied_usd: Decimal = ZERO
    funding_periods: int = 0
    funding_bps: Decimal = ZERO
    realised_funding_per_day_bps: Optional[Decimal] = None
    planned_funding_per_day_bps: Decimal = ZERO

    round_trip_bps: Decimal = ZERO
    planned_breakeven_days: Optional[Decimal] = None
    realised_breakeven_days: Optional[Decimal] = None
    projected_net_bps: Optional[Decimal] = None
    projected_net_usd: Optional[Decimal] = None

    net_delta_base: Decimal = ZERO
    net_delta_usd: Decimal = ZERO
    liquidation_distance: Optional[Decimal] = None

    perp_pnl_usd: Decimal = ZERO
    spot_pnl_usd: Decimal = ZERO
    total_pnl_usd: Decimal = ZERO

    alerts: List[Alert] = field(default_factory=list)

    @property
    def critical(self) -> bool:
        return any(alert.level == "critical" for alert in self.alerts)

    def add(self, level: str, code: str, message: str) -> None:
        self.alerts.append(Alert(level=level, code=code, message=message))


def compare(baseline: Baseline, snapshot: Snapshot, *,
            limits: Optional[RiskLimits] = None) -> MonitorReport:
    """Score the position against the plan, and raise what would end it early.

    Every number here is realised except the projection, and the projection is
    built on the realised funding rate rather than the planned one. That is
    the whole point: the plan's forecast has already been made, and repeating
    it back would be a report that cannot fail.
    """
    limits = limits or RiskLimits()
    report = MonitorReport(
        inst_id=baseline.inst_id,
        open=snapshot.open,
        planned_funding_per_day_bps=baseline.planned_funding_per_day_bps,
        round_trip_bps=baseline.round_trip_bps,
        planned_breakeven_days=baseline.planned_breakeven_days,
        funding_periods=snapshot.funding_periods,
    )

    elapsed_ms = Decimal(max(0, snapshot.at_ms - baseline.opened_at_ms))
    report.days_elapsed = elapsed_ms / MS_PER_DAY

    # ---- funding ---------------------------------------------------------
    report.funding_booked_usd = (snapshot.funding_booked_usd
                                 if snapshot.open else ZERO)
    report.funding_implied_usd = snapshot.funding_implied_usd
    if baseline.notional_usd > 0:
        report.funding_bps = report.funding_booked_usd / baseline.notional_usd * BPS
    if report.days_elapsed > 0:
        report.realised_funding_per_day_bps = (report.funding_bps
                                               / report.days_elapsed)

    daily = report.realised_funding_per_day_bps
    if daily is not None and daily > 0:
        report.realised_breakeven_days = report.round_trip_bps / daily
        report.projected_net_bps = daily * baseline.hold_days - report.round_trip_bps
        report.projected_net_usd = (report.projected_net_bps / BPS
                                    * baseline.notional_usd)

    # ---- delta -----------------------------------------------------------
    # `positions` is signed, so the sum of the two legs IS the net exposure.
    report.net_delta_base = snapshot.spot_base + snapshot.perp_contracts
    mark = snapshot.spot_mark or snapshot.perp_mark or baseline.perp_entry
    report.net_delta_usd = report.net_delta_base * mark

    # ---- mark to market --------------------------------------------------
    # The perp side carries its own fee inside `realizedPnl`; the spot side
    # carries its fee in the gap between what was paid and what is held. Added
    # this way each fee is counted once and neither is counted twice.
    report.perp_pnl_usd = snapshot.perp_unrealized_usd + snapshot.realized_pnl_usd
    report.spot_pnl_usd = (snapshot.spot_base * snapshot.spot_mark
                           - baseline.spot_cost_usd)
    report.total_pnl_usd = report.perp_pnl_usd + report.spot_pnl_usd

    if snapshot.liquidation and snapshot.perp_mark:
        report.liquidation_distance = (abs(snapshot.liquidation - snapshot.perp_mark)
                                       / snapshot.perp_mark)

    _raise_alerts(report, baseline, snapshot, limits)
    return report


def _raise_alerts(report: MonitorReport, baseline: Baseline,
                  snapshot: Snapshot, limits: RiskLimits) -> None:
    """Everything that would end the hold early, worst first."""

    # ---- is the position even still there --------------------------------
    if not snapshot.open and snapshot.spot_base > 0:
        report.add("critical", "naked-spot",
                   f"the perp leg is GONE and {snapshot.spot_base} "
                   f"{baseline.inst_id.split('-')[0]} of spot is still held. "
                   f"That is an unhedged long, which is the opposite of this "
                   f"trade. Close it or re-hedge it.")
        return
    if not snapshot.open:
        report.add("info", "closed",
                   "no perp position: this carry is closed.")
        return
    if snapshot.spot_base <= 0 and snapshot.perp_contracts:
        report.add("critical", "naked-perp",
                   f"a {snapshot.perp_contracts} contract perp short is open "
                   f"with no spot leg behind it. That is a naked directional "
                   f"short. Close it or re-hedge it.")

    # ---- liquidation ------------------------------------------------------
    distance = report.liquidation_distance
    if distance is not None and distance < limits.min_open_liquidation_buffer_pct:
        report.add("critical", "liquidation",
                   f"liquidation is {distance:.1%} away, inside the "
                   f"{limits.min_open_liquidation_buffer_pct:.1%} floor for an "
                   f"open position. The short leg going is not a flat pair - "
                   f"it leaves the spot leg unhedged.")

    # ---- delta ------------------------------------------------------------
    if baseline.notional_usd > 0:
        share = abs(report.net_delta_usd) / baseline.notional_usd
        if share > MAX_DELTA_SHARE:
            report.add("warn", "delta",
                       f"the legs are {report.net_delta_base:+} base apart "
                       f"(${report.net_delta_usd:+.2f}, {share:.2%} of "
                       f"notional). That much is directional, not carry.")

    # ---- funding ----------------------------------------------------------
    if snapshot.funding_periods == 0:
        report.add("info", "no-funding-yet",
                   "no funding settlement has passed since the open, so there "
                   "is nothing to score yet. Realised numbers below are "
                   "arithmetic on zero.")
        return

    booked = report.funding_booked_usd
    implied = report.funding_implied_usd
    gap = abs(booked - implied)
    tolerance = max(FUNDING_DIVERGENCE_USD,
                    abs(implied) * FUNDING_DIVERGENCE_SHARE)
    if gap > tolerance:
        report.add("warn", "funding-divergence",
                   f"the two funding measures disagree: ${booked:+.4f} booked "
                   f"against ${implied:+.4f} implied by the published rates "
                   f"(${gap:.4f} apart). One of them is wrong, so trust "
                   f"neither until it is understood.")

    if booked < 0:
        report.add("warn", "funding-negative",
                   f"funding has cost ${-booked:.4f} rather than paid. A carry "
                   f"that pays to exist is not a carry; if the rate stays "
                   f"here the position is a fee-burning delta-neutral hold.")
    elif (report.realised_funding_per_day_bps is not None
          and baseline.planned_funding_per_day_bps > 0
          and report.realised_funding_per_day_bps
          < baseline.planned_funding_per_day_bps * FUNDING_SHORTFALL_SHARE):
        share = (report.realised_funding_per_day_bps
                 / baseline.planned_funding_per_day_bps)
        report.add("warn", "funding-short",
                   f"realised funding is "
                   f"{report.realised_funding_per_day_bps:.2f} bps/day against "
                   f"{baseline.planned_funding_per_day_bps:.2f} predicted "
                   f"({share:.0%} of forecast). The median the plan used was "
                   f"drawn from a different regime.")

    if (report.realised_breakeven_days is not None
            and report.realised_breakeven_days > baseline.hold_days):
        report.add("warn", "breakeven-past-hold",
                   f"at the realised rate the round trip repays in "
                   f"{report.realised_breakeven_days:.1f} days, past the "
                   f"{baseline.hold_days}-day hold. Holding to plan loses "
                   f"money.")
