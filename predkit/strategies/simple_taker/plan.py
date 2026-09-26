"""The edge, net of the fee, or the reasons there is none. SENDS NOTHING.

`devig` turns a two-way price into probabilities that sum to one by
proportional normalisation (the multiplicative method). It is the simplest
of the standard methods and the one that makes the fewest assumptions; the
Shin and power methods sit closer to the truth on longshots and are a
one-function swap for a user who wants them.

Gates:
  time            at least `min_hours` to resolution; late in a market's
                  life the book is thin and the price moves on news faster
                  than a polling bot.
  fair            a model, a probability or two-way odds, or refuse.
  one position    already holding this market: refuse (hold to settlement).
  fee table       the tier must exist in `fees.py`.
  edge            fair - ask - taker fee per contract - builder fee >= min_edge
                  for YES; the mirror for NO at 1 - bid. The entry crosses
                  the spread by design, so it pays taker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Tuple

from predkit import fees
from predkit.backtest import Decision, WindowRow
from predkit.schema import ONE, ZERO, OrderIntent, Outcome, Side, to_decimal

MIN_EDGE = Decimal("0.03")
MIN_HOURS = 1.0


def devig(yes_odds, no_odds) -> Tuple[Decimal, Decimal]:
    """Decimal odds in, probabilities summing to one out."""
    yes_odds, no_odds = to_decimal(yes_odds), to_decimal(no_odds)
    if yes_odds <= ONE or no_odds <= ONE:
        raise ValueError(f"decimal odds must exceed 1.0; got {yes_odds} / {no_odds}")
    raw_yes, raw_no = ONE / yes_odds, ONE / no_odds
    total = raw_yes + raw_no
    return raw_yes / total, raw_no / total


@dataclass
class TakePlan:
    market_id: str
    fair: Optional[Decimal] = None
    edge_yes: Optional[Decimal] = None
    edge_no: Optional[Decimal] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    intents: List[OrderIntent] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def _fair_from(context) -> Tuple[Optional[Decimal], List[str]]:
    reference = context.reference or {}
    if context.fair is not None:
        return context.fair, []
    if reference.get("model_prob") is not None:
        return to_decimal(reference["model_prob"]), []
    if reference.get("sharp_prob") is not None:
        return to_decimal(reference["sharp_prob"]), []
    if reference.get("sharp_yes_odds") and reference.get("sharp_no_odds"):
        try:
            return devig(reference["sharp_yes_odds"], reference["sharp_no_odds"])[0], []
        except ValueError as exc:
            return None, [str(exc)]
    return None, ["no fair value: supply sharp_prob, sharp odds, or model_prob"]


def edges(venue: str, tier: str, fair: Decimal, ask: Decimal, bid: Decimal) -> Tuple[Decimal, Decimal]:
    """Net edge per contract for buying YES at the ask and NO at 1 - bid."""
    no_price = ONE - bid
    yes_edge = fair - ask - fees.total_fee(venue, tier, "taker", ask, ONE)
    no_edge = (ONE - fair) - no_price - fees.total_fee(venue, tier, "taker", no_price, ONE)
    return yes_edge, no_edge


def plan_take(context, *, min_edge: Decimal = MIN_EDGE, size: Decimal = Decimal("10"),
              min_hours: float = MIN_HOURS) -> TakePlan:
    contract, book = context.contract, context.book
    plan = TakePlan(contract.market_id)
    fair, problems = _fair_from(context)
    plan.reasons.extend(problems)
    plan.fair = fair
    if book.best_bid is None or book.best_ask is None:
        plan.reasons.append("book has no two-sided quote")
    hours = contract.seconds_to_resolution(context.now_ms) / 3600
    if hours < min_hours:
        plan.reasons.append(f"{hours:.1f}h to resolution is under {min_hours:g}h: too close to resolution")
    if context.position != ZERO:
        plan.reasons.append(f"already holding {context.position} in {contract.market_id}; "
                            "one position per market, held to settlement")
    try:
        fees.tier(contract.venue, contract.fee_tier)
    except KeyError as exc:
        plan.reasons.append(str(exc))
    if plan.reasons:
        return plan

    plan.edge_yes, plan.edge_no = edges(contract.venue, contract.fee_tier, fair, book.best_ask, book.best_bid)
    if plan.edge_yes >= min_edge and plan.edge_yes >= plan.edge_no:
        plan.intents.append(OrderIntent(contract, Side.BUY, book.best_ask, size, Outcome.YES, post_only=False))
    elif plan.edge_no >= min_edge:
        plan.intents.append(OrderIntent(contract, Side.BUY, ONE - book.best_bid, size, Outcome.NO,
                                        post_only=False))
    else:
        plan.warnings.append(f"no edge: YES {plan.edge_yes:+.4f}, NO {plan.edge_no:+.4f}, need {min_edge}")
    return plan


def decide_row(row: WindowRow, *, min_edge: Decimal = MIN_EDGE, size: Decimal = Decimal("10")) -> Optional[Decision]:
    if row.signal is None:
        return None
    try:
        yes_edge, no_edge = edges(row.venue, row.fee_tier, row.signal, row.entry_price, row.entry_bid)
    except KeyError:
        return None
    if yes_edge >= min_edge and yes_edge >= no_edge:
        return Decision(Outcome.YES, row.entry_price, size, role="taker")
    if no_edge >= min_edge:
        return Decision(Outcome.NO, ONE - row.entry_bid, size, role="taker")
    return None


__all__ = ["MIN_EDGE", "MIN_HOURS", "TakePlan", "decide_row", "devig", "edges", "plan_take"]
