"""The veto. Imports nothing from predkit, uses Decimal, answers in full.

This module sits outside every strategy and can refuse any of them. Nothing
here imports the schema, the venues, the fee tables or the strategies: the
dependency only ever points inward, so the veto cannot be weakened by a
change to the thing it vetoes. That is why the order fields arrive as plain
strings and Decimals rather than as an `OrderIntent`.

Prediction markets make the veto simpler than a margin engine. A contract
costs what you pay for it and can lose no more, so there is no liquidation
price, no leverage and no funding. What is left:

  max_contracts_per_market   position size, in YES-equivalent contracts
  max_open_notional          dollars at risk across every open position
  resolution_window_s        no NEW risk this close to resolution, or after it
                             (Polymarket's 5-minute markets resolve on a
                             Chainlink 60-second TWAP, so the last minute is
                             a different game and needs its own analysis)
  max_daily_loss             realised loss today at which opening stops

Every failing gate is returned, not the first. And an order that strictly
shrinks a position takes the short path: a veto that stops you getting OUT
is a known way for a kill switch to deadlock a bot in a losing position, and
none of these gates is a reason to be unable to exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass
class Limits:
    """Hard limits. Not suggestions the strategy may negotiate with.

    Defaults are for a paper account finding its feet. Polymarket's minimum
    order is 5 contracts, so the defaults allow a few of those.
    """

    max_contracts_per_market: Decimal = Decimal("100")
    max_open_notional: Decimal = Decimal("500")
    max_open_orders_per_market: int = 2
    resolution_window_s: float = 60.0
    max_daily_loss: Decimal = Decimal("50")


@dataclass
class Veto:
    allowed: bool
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.allowed


def _signed(side: str, outcome: str, size: Decimal) -> Decimal:
    """YES-equivalent contracts an order adds: buy YES +, buy NO -, sells the
    reverse. Validated here because this module trusts nobody's enum."""
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', not {side!r}")
    if outcome not in ("yes", "no"):
        raise ValueError(f"outcome must be 'yes' or 'no', not {outcome!r}")
    sign = ONE if side == "buy" else -ONE
    if outcome == "no":
        sign = -sign
    return sign * size


@dataclass
class RiskEngine:
    limits: Limits = field(default_factory=Limits)
    # Signed YES-equivalent contracts and dollars at risk, per market.
    contracts: Dict[str, Decimal] = field(default_factory=dict)
    at_risk: Dict[str, Decimal] = field(default_factory=dict)
    open_orders: Dict[str, int] = field(default_factory=dict)
    realized_pnl_today: Decimal = ZERO
    day: str = ""
    kill_switch_active: bool = False
    kill_switch_reason: str = ""

    # ---- state -----------------------------------------------------------

    def trip(self, reason: str) -> None:
        self.kill_switch_active = True
        self.kill_switch_reason = reason

    def reset_kill_switch(self) -> None:
        self.kill_switch_active = False
        self.kill_switch_reason = ""

    def start_new_day(self, day: str) -> None:
        self.day = day
        self.realized_pnl_today = ZERO

    @property
    def open_notional(self) -> Decimal:
        return sum(self.at_risk.values(), ZERO)

    def on_fill(self, market_id: str, side: str, outcome: str, price: Decimal,
                size: Decimal, fee: Decimal = ZERO) -> None:
        """Book a fill. `price` is the price of the OUTCOME traded, so the
        dollars paid are `price * size` whichever outcome it is. Money at
        risk is what was paid; a fill that reduces the position (selling what
        is held, or buying the other outcome against it, which locks a $1
        payout per pair) realises the difference against average cost."""
        delta = _signed(side, outcome, size)
        cost = price * size
        held = self.contracts.get(market_id, ZERO)
        risk = self.at_risk.get(market_id, ZERO)
        reduces = held != ZERO and (held > ZERO) != (delta > ZERO)
        if reduces:
            closed = min(abs(delta), abs(held))
            avg_cost = risk / abs(held) if held else ZERO
            proceeds = price * closed if side == "sell" else (ONE - price) * closed
            self.realized_pnl_today += proceeds - avg_cost * closed - fee
            risk -= avg_cost * closed
            remaining = abs(delta) - closed
            held += delta
            if remaining > ZERO:
                risk += price * remaining
        else:
            held += delta
            risk += cost
            self.realized_pnl_today -= fee
        self.contracts[market_id] = held
        self.at_risk[market_id] = max(risk, ZERO)

    def on_resolution(self, market_id: str, resolved_yes: bool) -> None:
        """The market settled: held YES pays 1 each, held NO pays 1 each."""
        held = self.contracts.pop(market_id, ZERO)
        risk = self.at_risk.pop(market_id, ZERO)
        self.open_orders.pop(market_id, None)
        payout = abs(held) if (held > ZERO) == resolved_yes and held != ZERO else ZERO
        self.realized_pnl_today += payout - risk

    # ---- the gate --------------------------------------------------------

    def check_order(self, *, market_id: str, side: str, outcome: str, price: Decimal,
                    size: Decimal, resolves_at_ms: int, now_ms: int,
                    reduce_only: bool = False) -> Veto:
        """Approve or reject a proposed order, with every reason.

        `reduce_only` is verified, not trusted: an order so tagged that would
        not in fact shrink the position is rejected.
        """
        limits = self.limits
        reasons: List[str] = []
        if size <= ZERO:
            return Veto(False, [f"size {size} is not positive"])
        if price < ZERO or price > ONE:
            return Veto(False, [f"price {price} is outside [0, 1]"])
        delta = _signed(side, outcome, size)
        held = self.contracts.get(market_id, ZERO)

        if reduce_only:
            if held == ZERO or (held > ZERO) == (delta > ZERO):
                reasons.append(f"tagged reduce_only but {market_id} holds {held} and this "
                               f"order adds {delta}: it does not reduce")
            elif abs(delta) > abs(held):
                reasons.append(f"reduce_only order for {abs(delta)} exceeds the {abs(held)} "
                               f"held in {market_id}; it would flip the position")
            return Veto(not reasons, reasons)

        if self.kill_switch_active:
            reasons.append(f"kill switch active: {self.kill_switch_reason}")
        seconds_left = (resolves_at_ms - now_ms) / 1000.0
        if seconds_left <= limits.resolution_window_s:
            reasons.append(f"{market_id} resolves in {seconds_left:.0f}s, inside the "
                           f"{limits.resolution_window_s:.0f}s window: no new risk there")
        if self.realized_pnl_today <= -limits.max_daily_loss:
            reasons.append(f"daily loss {self.realized_pnl_today} has reached the "
                           f"{limits.max_daily_loss} stop")
        after = held + delta
        if abs(after) > limits.max_contracts_per_market:
            reasons.append(f"{market_id} would hold {after} contracts, over the "
                           f"{limits.max_contracts_per_market} limit")
        added_risk = price * size
        if self.open_notional + added_risk > limits.max_open_notional:
            reasons.append(f"open notional {self.open_notional} + {added_risk} exceeds "
                           f"{limits.max_open_notional}")
        if self.open_orders.get(market_id, 0) >= limits.max_open_orders_per_market:
            reasons.append(f"{market_id} already has {self.open_orders.get(market_id, 0)} "
                           f"open orders (limit {limits.max_open_orders_per_market})")
        return Veto(not reasons, reasons)

    def status(self) -> Dict[str, object]:
        return {
            "killSwitch": self.kill_switch_active,
            "killSwitchReason": self.kill_switch_reason,
            "day": self.day,
            "realizedPnlToday": str(self.realized_pnl_today),
            "openNotional": str(self.open_notional),
            "contracts": {k: str(v) for k, v in self.contracts.items()},
            "openOrders": dict(self.open_orders),
        }


__all__ = ["Limits", "RiskEngine", "Veto"]
