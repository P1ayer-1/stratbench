"""Where the maker rests, and every reason not to. SENDS NOTHING.

Gates, each a refusal with the number that failed:

  fair value        none means no reference feed: refuse. The strategy IS
                    the difference between fair and the touch.
  feed lag          over 350 ms refuse, over 200 ms warn. A lead-lag quote
                    that pays at 150 ms of latency can lose at 500; the lag
                    measured on the host actually running is a gate.
  resolution        inside the window (60 s default) refuse; the settlement
                    TWAP makes the last minute a different market.
  alone at level    the bid at ask - tick must be above the best bid, or
                    the order joins a queue and the optimistic fill bound is
                    a lie. Warn and post nothing on that side.
  fee table         the contract's tier must exist in fees.py. A market
                    whose fee tier nobody has transcribed is refused until
                    someone does. That is the design working.
  edge              fair - price >= edge + builder maker fee per contract.

Sizes: `edge` and `stop` are in PRICE (probability points). On a 50-cent
contract one tick is 2% of price, so the defaults are two ticks of edge and
one of stop. They are placeholders, meant to be re-fitted from a replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional

from predkit import fees
from predkit.backtest import Decision, WindowRow
from predkit.schema import ONE, ZERO, OrderIntent, Outcome, Side

EDGE = Decimal("0.02")
STOP = Decimal("0.01")
LAG_WARN_MS = 200.0
LAG_REFUSE_MS = 350.0
RESOLUTION_WINDOW_S = 60.0


@dataclass
class QuotePlan:
    market_id: str
    fair: Optional[Decimal] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    intents: List[OrderIntent] = field(default_factory=list)
    exits: List[OrderIntent] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def plan_quote(context, *, edge: Decimal = EDGE, stop: Decimal = STOP, size: Decimal = Decimal("5"),
               lag_warn_ms: float = LAG_WARN_MS, lag_refuse_ms: float = LAG_REFUSE_MS,
               resolution_window_s: float = RESOLUTION_WINDOW_S) -> QuotePlan:
    contract, book = context.contract, context.book
    plan = QuotePlan(contract.market_id, fair=context.fair)
    tick = contract.tick

    if book.best_bid is None or book.best_ask is None:
        plan.reasons.append("book has no two-sided quote")
    elif book.is_crossed():
        plan.reasons.append(f"book is crossed (bid {book.best_bid} >= ask {book.best_ask})")
    if context.fair is None:
        plan.reasons.append("no fair value: the reference feed has not produced one")
    if contract.in_resolution_window(context.now_ms, resolution_window_s):
        plan.reasons.append(f"{contract.seconds_to_resolution(context.now_ms):.0f}s to resolution, "
                            f"inside the {resolution_window_s:.0f}s window")
    if context.feed_lag_ms is None:
        plan.warnings.append("feed lag not measured yet")
    elif context.feed_lag_ms > lag_refuse_ms:
        plan.reasons.append(f"feed lag {context.feed_lag_ms:.0f} ms is over {lag_refuse_ms:.0f}")
    elif context.feed_lag_ms > lag_warn_ms:
        plan.warnings.append(f"feed lag {context.feed_lag_ms:.0f} ms is over {lag_warn_ms:.0f}")
    try:
        maker_builder = fees.builder_fee("maker", Decimal("0.5"), ONE)   # per contract, at the peak
        fees.tier(contract.venue, contract.fee_tier)
    except KeyError as exc:
        plan.reasons.append(str(exc))
        maker_builder = ZERO
    if plan.reasons:
        return plan

    fair = context.fair
    bid, ask = book.best_bid, book.best_ask
    needed = edge + maker_builder

    # Bid side: buy YES one tick inside the ask, alone at the level.
    target = ask - tick
    if target <= bid:
        plan.warnings.append(f"bid at {target} would join the queue at the touch; no bid")
    elif fair - target >= needed:
        plan.intents.append(OrderIntent(contract, Side.BUY, target, size, Outcome.YES, post_only=True))
    # Ask side: sell YES one tick above the bid, which is a NO bid at 1 - that.
    target_yes = bid + tick
    if target_yes >= ask:
        plan.warnings.append(f"ask at {target_yes} would join the queue at the touch; no ask")
    elif target_yes - fair >= needed:
        plan.intents.append(OrderIntent(contract, Side.BUY, ONE - target_yes, size, Outcome.NO,
                                        post_only=True))

    # Stop: fair has moved through the position's cost by `stop`; exit at the touch.
    if context.position > ZERO and fair <= context.avg_cost - stop:
        plan.exits.append(OrderIntent(contract, Side.SELL, bid, abs(context.position), Outcome.YES,
                                      post_only=False))
        plan.warnings.append(f"stop: fair {fair} <= cost {context.avg_cost} - {stop}; exiting YES")
    elif context.position < ZERO and (ONE - fair) <= context.avg_cost - stop:
        plan.exits.append(OrderIntent(contract, Side.SELL, ONE - ask, abs(context.position), Outcome.NO,
                                      post_only=False))
        plan.warnings.append(f"stop: NO fair {ONE - fair} <= cost {context.avg_cost} - {stop}; exiting NO")
    return plan


def decide_row(row: WindowRow, *, edge: Decimal = EDGE, size: Decimal = Decimal("5")) -> Optional[Decision]:
    """The same thresholds on a labelled row. Assumes the resting order
    filled, which is the OPTIMISTIC bound; the replay fill model is the
    honest one and this exists so the two can be compared."""
    if row.signal is None:
        return None
    tick = Decimal("0.01")
    target = row.entry_price - tick
    if target > row.entry_bid and row.signal - target >= edge:
        return Decision(Outcome.YES, target, size, role="maker")
    target_yes = row.entry_bid + tick
    if target_yes < row.entry_price and target_yes - row.signal >= edge:
        return Decision(Outcome.NO, ONE - target_yes, size, role="maker")
    return None


__all__ = ["EDGE", "STOP", "QuotePlan", "decide_row", "plan_quote"]
