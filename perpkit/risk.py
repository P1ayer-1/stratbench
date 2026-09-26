"""Liquidation math, position sizing, and the hard-limit risk engine.

This module sits **outside** the strategy and can veto it. Nothing in here
imports the feature engine, the model, or the execution layer — the dependency
only ever points inward, so the risk engine cannot be accidentally weakened by
a change to a signal. That is the whole design point.

Everything here uses `Decimal`, not float. Elsewhere in this package floats are
correct (features are ratios). Here they are not: a liquidation price is money,
it gets compared against exchange-reported values, and 0.1 + 0.2 != 0.3 is not
an argument you want to have with a margin engine at 20x.

=============================================================================
READ THIS BEFORE TRADING REAL MONEY
=============================================================================
The liquidation formula below is the standard textbook model for a linear
(USDT-margined) perpetual with a flat maintenance-margin rate. Real exchanges
differ in ways that move the number:

  * MMR is **tiered** by position size — bigger positions have higher MMR and
    therefore liquidate sooner than this formula says.
  * Closing fees and accrued funding are debited from equity, pulling the real
    liquidation price closer to entry than the fee-free formula.
  * Cross margin liquidation depends on the whole account, not one position.
  * Some venues liquidate against the mark price, not last traded price.

`fee_buffer_bps` below biases the estimate in the safe direction, but the only
correct validation is to open a small demo position, read back the exchange's own
reported `liquidationPrice` from the positions endpoint, and compare. There is
a helper for exactly that: `compare_to_exchange()`. Do that before sizing
anything with leverage.
=============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, List, Optional

ZERO = Decimal("0")
ONE = Decimal("1")
BPS = Decimal("10000")


def to_decimal(value: object, default: str = "0") -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


# ---------------------------------------------------------------------------
# Liquidation
# ---------------------------------------------------------------------------


def liquidation_price(
    *,
    entry_price: Decimal,
    leverage: Decimal,
    side: Side,
    maintenance_margin_rate: Decimal = Decimal("0.005"),
    extra_margin: Decimal = ZERO,
    quantity_base: Decimal = ZERO,
    fee_buffer_bps: Decimal = ZERO,
) -> Optional[Decimal]:
    """Estimated liquidation price for a linear perpetual position.

    Derivation (long). Liquidation happens when account equity for the
    position falls to the maintenance requirement:

        equity  = initial_margin + extra_margin + qty * (P - entry)
        required = qty * P * mmr

    with initial_margin = qty * entry / L. Setting equity == required and
    dividing through by qty:

        entry/L + M/qty + P - entry = P * mmr
        P * (1 - mmr) = entry * (1 - 1/L) - M/qty

                        entry * (1 - 1/L) - M/qty
        P_liq(long)  =  --------------------------
                                (1 - mmr)

    and by the symmetric argument for a short:

                        entry * (1 + 1/L) + M/qty
        P_liq(short) =  --------------------------
                                (1 + mmr)

    Args:
        entry_price: average fill price of the position.
        leverage: position leverage, > 0.
        side: LONG or SHORT.
        maintenance_margin_rate: MMR as a fraction (0.005 = 0.5%). Use the
            *tier* that applies to your intended size, not the base tier.
        extra_margin: additional margin backing this position beyond the
            initial margin (cross margin, or a manual top-up). Requires
            `quantity_base` to be meaningful.
        quantity_base: position size in base currency (BTC, not contracts).
            Only needed when `extra_margin` is non-zero.
        fee_buffer_bps: conservatism pad. Moves the estimate *toward* entry by
            this many bps, approximating the closing fee and funding that the
            clean formula ignores. 10-20 bps is a reasonable starting pad.

    Returns:
        The estimated liquidation price, or None if the inputs are unusable.
        A long whose formula yields <= 0 cannot be liquidated by price alone
        (leverage <= 1 with margin to spare) and returns None.
    """
    if entry_price <= 0 or leverage <= 0:
        return None
    if maintenance_margin_rate < 0 or maintenance_margin_rate >= 1:
        return None

    margin_per_unit = ZERO
    if extra_margin != ZERO:
        if quantity_base <= 0:
            return None
        margin_per_unit = extra_margin / quantity_base

    if side is Side.LONG:
        numerator = entry_price * (ONE - ONE / leverage) - margin_per_unit
        denominator = ONE - maintenance_margin_rate
        price = numerator / denominator
        if price <= 0:
            return None
        # Safe direction for a long is *upward* (liquidates sooner).
        price *= ONE + fee_buffer_bps / BPS
        return min(price, entry_price)

    numerator = entry_price * (ONE + ONE / leverage) + margin_per_unit
    denominator = ONE + maintenance_margin_rate
    price = numerator / denominator
    # Safe direction for a short is *downward*.
    price *= ONE - fee_buffer_bps / BPS
    return max(price, entry_price)


def liquidation_distance_pct(
    *, mark_price: Decimal, liq_price: Optional[Decimal]
) -> Optional[Decimal]:
    """How far price must move, as a fraction of mark, before liquidation.

    Always non-negative. This is the single number to watch at high leverage:
    at 50x it is roughly 0.02, i.e. a 2% adverse move ends the position.
    """
    if liq_price is None or mark_price <= 0:
        return None
    return abs(mark_price - liq_price) / mark_price


def max_safe_leverage(
    *,
    buffer_pct: Decimal,
    maintenance_margin_rate: Decimal = Decimal("0.005"),
    side: Side = Side.LONG,
) -> Decimal:
    """Largest leverage whose liquidation sits at least `buffer_pct` away.

    Inverts the formula above. Use it to answer "I want to survive a 5% move,
    what leverage may I use?" rather than picking leverage first and hoping.
    """
    if buffer_pct <= 0 or buffer_pct >= 1:
        return ZERO
    mmr = maintenance_margin_rate
    if side is Side.LONG:
        # P_liq/entry = (1 - 1/L)/(1 - mmr) = 1 - buffer
        denominator = ONE - (ONE - buffer_pct) * (ONE - mmr)
    else:
        denominator = (ONE + buffer_pct) * (ONE + mmr) - ONE
    if denominator <= 0:
        return ZERO
    return ONE / denominator


def compare_to_exchange(
    *, estimated: Optional[Decimal], exchange_reported: Decimal
) -> Dict[str, object]:
    """Sanity-check our model against the exchange's own reported liquidation price.

    Run this against a small demo position before trusting any sizing logic.
    A relative error above ~1% means the MMR tier or fee assumptions are wrong
    and must be corrected before increasing leverage.
    """
    if estimated is None or exchange_reported <= 0:
        return {"ok": False, "reason": "missing or invalid inputs"}
    error = abs(estimated - exchange_reported) / exchange_reported
    return {
        "ok": error <= Decimal("0.01"),
        "estimated": estimated,
        "exchange": exchange_reported,
        "relativeError": error,
        "reason": "" if error <= Decimal("0.01") else "model disagrees with exchange",
    }


# ---------------------------------------------------------------------------
# Limits and state
# ---------------------------------------------------------------------------


@dataclass
class RiskLimits:
    """Hard limits. These are not suggestions the strategy may negotiate with.

    Defaults are deliberately conservative — they are set for a demo account
    finding its feet, not for a tuned production system.
    """

    max_position_base: Decimal = Decimal("0.05")
    max_notional: Decimal = Decimal("5000")
    max_leverage: Decimal = Decimal("5")
    max_order_base: Decimal = Decimal("0.01")
    max_open_orders: int = 4

    max_daily_loss: Decimal = Decimal("100")
    max_consecutive_losses: int = 4

    # Unrealized-loss limits. Every limit above keys on *realized* PnL, which
    # a strategy that never closes a loser never touches. These two are what
    # make an open drawdown visible to the kill switch.
    max_unrealized_loss: Decimal = Decimal("50")
    # Trip when an already-open position drifts this close to liquidation.
    # Looser than min_liquidation_buffer_pct (which gates *opening*): once you
    # are in, the question is no longer "is this a good entry" but "get out
    # before the margin engine does it for you".
    min_open_liquidation_buffer_pct: Decimal = Decimal("0.08")

    max_spread_bps: Decimal = Decimal("5")
    max_slippage_bps: Decimal = Decimal("5")

    # The liquidation guard: refuse any position whose estimated liquidation
    # sits closer than this fraction of price. 0.15 = need a 15% adverse move.
    min_liquidation_buffer_pct: Decimal = Decimal("0.15")
    maintenance_margin_rate: Decimal = Decimal("0.005")
    fee_buffer_bps: Decimal = Decimal("15")

    # Feed freshness. Stale data at high leverage is worse than no data.
    max_book_age_ms: int = 2000
    max_tape_staleness_s: float = 10.0

    # Round-trip cost floor used by the expected-edge gate, in bps. Pass
    # `perpkit.fees.ROUND_TRIP_COST_BPS` (derived from the transcribed BloFin
    # schedule); this default exists only so the engine is usable standalone,
    # since risk.py imports nothing. 10 bps = taker on both sides at 5 bps.
    round_trip_cost_bps: Decimal = Decimal("10")


@dataclass
class AccountState:
    equity: Decimal = Decimal("1000")
    position_base: Decimal = ZERO          # signed: + long, - short
    entry_price: Decimal = ZERO
    leverage: Decimal = ONE
    open_orders: int = 0
    realized_pnl_today: Decimal = ZERO
    consecutive_losses: int = 0

    @property
    def side(self) -> Optional[Side]:
        if self.position_base > 0:
            return Side.LONG
        if self.position_base < 0:
            return Side.SHORT
        return None

    def unrealized_pnl(self, mark_price: Decimal) -> Decimal:
        """Open PnL on the current position, in quote currency.

        `position_base` is signed, so one expression covers both sides: a
        short is a negative quantity, and a negative quantity multiplied by a
        price rise is a loss.
        """
        if self.position_base == 0 or self.entry_price <= 0 or mark_price <= 0:
            return ZERO
        return self.position_base * (mark_price - self.entry_price)

    def total_pnl_today(self, mark_price: Decimal) -> Decimal:
        """Realized PnL plus the open position marked to market.

        This, not `realized_pnl_today`, is what a daily loss limit means.
        """
        return self.realized_pnl_today + self.unrealized_pnl(mark_price)


@dataclass
class RiskDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)
    max_size_base: Decimal = ZERO
    liq_price: Optional[Decimal] = None
    liq_distance_pct: Optional[Decimal] = None

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class SizingResult:
    size_base: Decimal
    binding_constraint: str
    detail: Dict[str, Decimal] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class RiskEngine:
    """Vetoes orders. Owns the kill switch.

    The engine is intentionally boring and synchronous. Every check is a plain
    comparison that a human can read and verify, because this is the code that
    stands between a model bug and a liquidated account.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.connected = True
        self.last_event_time = time.time()

    # ---- kill switch -----------------------------------------------------

    def trip(self, reason: str) -> None:
        """Halt all new orders. Deliberately has no automatic un-trip: a human
        (or an explicit restart) decides when it is safe to resume."""
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.kill_switch_reason = reason

    def reset_kill_switch(self) -> None:
        self.kill_switch_active = False
        self.kill_switch_reason = ""

    def on_disconnect(self) -> List[str]:
        """The disconnect path: cancel everything, open nothing.

        Returns the ordered actions the execution layer must carry out. We
        return them rather than performing them so this module stays free of
        exchange dependencies and stays unit-testable.
        """
        self.connected = False
        self.trip("exchange disconnected")
        return ["cancel_all_orders", "stop_opening_positions", "alert_operator"]

    def on_reconnect(self) -> None:
        """Reconnection restores the feed but NOT trading. Positions and
        orders must be reconciled against the exchange first, then the kill
        switch cleared explicitly."""
        self.connected = True

    # ---- pnl tracking ----------------------------------------------------

    def on_trade_closed(self, realized_pnl: Decimal, account: AccountState) -> None:
        account.realized_pnl_today += realized_pnl
        if realized_pnl < 0:
            account.consecutive_losses += 1
        else:
            account.consecutive_losses = 0

        if account.realized_pnl_today <= -self.limits.max_daily_loss:
            self.trip(
                f"daily loss limit hit: {account.realized_pnl_today} "
                f"<= -{self.limits.max_daily_loss}"
            )
        if account.consecutive_losses >= self.limits.max_consecutive_losses:
            self.trip(
                f"consecutive loss limit hit: {account.consecutive_losses}"
            )

    def mark_to_market(
        self, account: AccountState, mark_price: Decimal
    ) -> Dict[str, object]:
        """Evaluate an OPEN position and trip the kill switch if it has gone
        too far wrong. Call this on every mark-price update, not only on fills.

        This is the counterpart to `on_trade_closed`, and it exists because of
        one specific failure mode. Every other breaker in this class is fed by
        `on_trade_closed`, so a strategy that simply declines to close a losing
        position - "the trend is up, we can afford to wait" - never trips any
        of them. Its drawdown stays invisible right up until the exchange
        closes the position on its behalf. An open loss is a loss.

        Returns what it saw, plus the ordered actions the execution layer must
        carry out if the switch tripped. As with `on_disconnect`, the actions
        are returned rather than performed, so this module keeps no exchange
        dependency.
        """
        limits = self.limits
        unrealized = account.unrealized_pnl(mark_price)
        total = account.realized_pnl_today + unrealized
        breaches: List[str] = []

        liq = None
        distance = None
        side = account.side
        if side is not None and account.entry_price > 0 and mark_price > 0:
            liq = liquidation_price(
                entry_price=account.entry_price,
                leverage=account.leverage,
                side=side,
                maintenance_margin_rate=limits.maintenance_margin_rate,
                fee_buffer_bps=limits.fee_buffer_bps,
            )
            distance = liquidation_distance_pct(
                mark_price=mark_price, liq_price=liq
            )
            if (
                distance is not None
                and distance < limits.min_open_liquidation_buffer_pct
            ):
                breaches.append(
                    f"open position {distance:.2%} from liquidation, need "
                    f"{limits.min_open_liquidation_buffer_pct:.2%}"
                )

        if unrealized <= -limits.max_unrealized_loss:
            breaches.append(
                f"unrealized loss {unrealized} <= -{limits.max_unrealized_loss}"
            )
        if total <= -limits.max_daily_loss:
            breaches.append(
                f"daily loss limit hit on a mark-to-market basis: {total} "
                f"<= -{limits.max_daily_loss}"
            )

        for reason in breaches:
            self.trip(reason)

        return {
            "unrealizedPnl": unrealized,
            "totalPnlToday": total,
            "liquidationPrice": liq,
            "liquidationDistancePct": distance,
            "breaches": breaches,
            "actions": (
                ["close_position", "cancel_all_orders", "alert_operator"]
                if breaches
                else []
            ),
        }

    def start_new_day(self, account: AccountState) -> None:
        account.realized_pnl_today = ZERO
        account.consecutive_losses = 0

    # ---- the gate --------------------------------------------------------

    def check_order(
        self,
        *,
        account: AccountState,
        side: Side,
        size_base: Decimal,
        price: Decimal,
        leverage: Decimal,
        spread_bps: Decimal,
        expected_edge_bps: Decimal = ZERO,
        book_age_ms: int = 0,
        tape_staleness_s: float = 0.0,
        features_valid: bool = True,
        reduce_only: bool = False,
    ) -> RiskDecision:
        """Approve, shrink, or reject a proposed order.

        Collects *every* failing reason rather than short-circuiting on the
        first — when something goes wrong at 3am you want the full list in the
        log, not just whichever check happened to run first.

        `reduce_only` marks an order that strictly shrinks the open position.
        Those take a much shorter path, because otherwise the engine deadlocks
        in precisely the situation it exists for: `mark_to_market` trips the
        kill switch and returns "close_position", and then the full gate below
        vetoes that closing order - kill switch active, no expected edge, daily
        loss reached. Every one of those is a reason to stop *opening* risk and
        none of them is a reason to be unable to get out. The claim is verified
        rather than trusted: an order tagged reduce_only that would not in fact
        reduce exposure is rejected.
        """
        limits = self.limits
        reasons: List[str] = []

        if reduce_only:
            return self._check_reduce_only(
                account=account, side=side, size_base=size_base, price=price
            )

        if self.kill_switch_active:
            reasons.append(f"kill switch active: {self.kill_switch_reason}")
        if not self.connected:
            reasons.append("not connected to exchange")
        if not features_valid:
            reasons.append("feature snapshot invalid (book stale or crossed)")
        if book_age_ms > limits.max_book_age_ms:
            reasons.append(f"book age {book_age_ms}ms > {limits.max_book_age_ms}ms")
        if tape_staleness_s > limits.max_tape_staleness_s:
            reasons.append(
                f"tape stale {tape_staleness_s:.1f}s > {limits.max_tape_staleness_s}s"
            )

        if price <= 0:
            reasons.append("invalid price")
        if size_base <= 0:
            reasons.append("invalid size")
        if leverage <= 0:
            reasons.append("invalid leverage")
        if reasons and price <= 0:
            return RiskDecision(False, reasons)

        if spread_bps > limits.max_spread_bps:
            reasons.append(
                f"spread {spread_bps:.2f}bps > max {limits.max_spread_bps}bps"
            )
        if leverage > limits.max_leverage:
            reasons.append(f"leverage {leverage} > max {limits.max_leverage}")
        if size_base > limits.max_order_base:
            reasons.append(
                f"order size {size_base} > max {limits.max_order_base}"
            )
        if account.open_orders >= limits.max_open_orders:
            reasons.append(
                f"open orders {account.open_orders} >= max {limits.max_open_orders}"
            )

        # The expected-edge > 0 gate, with fees as the floor.
        net_edge = expected_edge_bps - limits.round_trip_cost_bps
        if net_edge <= 0:
            reasons.append(
                f"expected edge {expected_edge_bps:.2f}bps does not clear "
                f"round-trip cost {limits.round_trip_cost_bps}bps"
            )

        # Resulting position after this fill.
        signed = size_base if side is Side.LONG else -size_base
        resulting = account.position_base + signed
        if abs(resulting) > limits.max_position_base:
            reasons.append(
                f"resulting position {abs(resulting)} > max {limits.max_position_base}"
            )
        if abs(resulting) * price > limits.max_notional:
            reasons.append(
                f"resulting notional {abs(resulting) * price:.2f} > "
                f"max {limits.max_notional}"
            )

        # Marked to market at the order price, not realized-only: an open
        # drawdown spends the day's risk budget just as surely as a booked one.
        total_pnl = account.total_pnl_today(price)
        if total_pnl <= -limits.max_daily_loss:
            reasons.append(
                f"daily loss limit reached (mark-to-market {total_pnl})"
            )
        unrealized = account.unrealized_pnl(price)
        if unrealized <= -limits.max_unrealized_loss:
            reasons.append(
                f"unrealized loss {unrealized} <= -{limits.max_unrealized_loss}"
            )
        if account.consecutive_losses >= limits.max_consecutive_losses:
            reasons.append("consecutive loss limit reached")

        # Liquidation guard on the *resulting* position, not the new slice.
        liq = None
        distance = None
        if resulting != 0 and price > 0 and leverage > 0:
            resulting_side = Side.LONG if resulting > 0 else Side.SHORT
            blended_entry = self._blended_entry(account, side, size_base, price)
            liq = liquidation_price(
                entry_price=blended_entry,
                leverage=leverage,
                side=resulting_side,
                maintenance_margin_rate=limits.maintenance_margin_rate,
                fee_buffer_bps=limits.fee_buffer_bps,
            )
            distance = liquidation_distance_pct(mark_price=price, liq_price=liq)
            if distance is None:
                reasons.append("could not compute liquidation price")
            elif distance < limits.min_liquidation_buffer_pct:
                reasons.append(
                    f"liquidation only {distance:.2%} away, need "
                    f"{limits.min_liquidation_buffer_pct:.2%}"
                )

        headroom = max(ZERO, limits.max_position_base - abs(account.position_base))
        allowance = min(limits.max_order_base, headroom)

        return RiskDecision(
            allowed=not reasons,
            reasons=reasons,
            max_size_base=allowance,
            liq_price=liq,
            liq_distance_pct=distance,
        )

    def _check_reduce_only(
        self,
        *,
        account: AccountState,
        side: Side,
        size_base: Decimal,
        price: Decimal,
    ) -> RiskDecision:
        """The narrow gate for orders that only shrink exposure.

        Deliberately checks almost nothing. A closing order is refused only if
        it is malformed, if there is nothing to close, if it is not in fact a
        reduction, or if there is no connection to send it over. A blown-out
        spread or a stale tape is a reason to close carefully, not a reason to
        stay in.
        """
        reasons: List[str] = []
        if not self.connected:
            reasons.append("not connected to exchange")
        if price <= 0:
            reasons.append("invalid price")
        if size_base <= 0:
            reasons.append("invalid size")

        existing = account.position_base
        if existing == 0:
            reasons.append("reduce_only order with no open position")
        else:
            closing_side = Side.SHORT if existing > 0 else Side.LONG
            if side is not closing_side:
                reasons.append(
                    f"reduce_only order is {side.value}, but closing a "
                    f"{account.side.value} position needs {closing_side.value}"
                )
            elif size_base > abs(existing):
                reasons.append(
                    f"reduce_only size {size_base} exceeds position "
                    f"{abs(existing)} and would flip it"
                )

        return RiskDecision(
            allowed=not reasons,
            reasons=reasons,
            max_size_base=abs(existing),
        )

    @staticmethod
    def _blended_entry(
        account: AccountState, side: Side, size_base: Decimal, price: Decimal
    ) -> Decimal:
        """Volume-weighted entry after adding this order.

        Only blends when the order increases exposure in the same direction;
        a reducing order leaves the entry price alone, which is how exchanges
        account for it.
        """
        existing = account.position_base
        signed = size_base if side is Side.LONG else -size_base
        if existing == 0 or account.entry_price <= 0:
            return price
        if (existing > 0) != (signed > 0):
            return account.entry_price
        total = abs(existing) + abs(signed)
        if total == 0:
            return price
        return (
            abs(existing) * account.entry_price + abs(signed) * price
        ) / total

    # ---- sizing ----------------------------------------------------------

    def size_position(
        self,
        *,
        account: AccountState,
        price: Decimal,
        expected_edge_bps: Decimal,
        volatility_bps: Decimal,
        win_probability: Decimal,
        leverage: Decimal,
        kelly_fraction: Decimal = Decimal("0.15"),
        risk_per_trade: Decimal = Decimal("0.005"),
    ) -> SizingResult:
        """Size a position from edge, volatility, and account equity.

        Three candidate sizes are computed and the **smallest wins**:

          1. Volatility target — risk a fixed fraction of equity per unit of
             expected move (expected return over expected volatility).
          2. Fractional Kelly — heavily discounted (default 15% of full Kelly)
             because at short horizons the estimate of `p` is mostly noise, and
             full Kelly on a mis-estimated edge is a reliable way to blow up.
          3. The hard limits — position cap, notional cap, order cap.

        Returning the binding constraint by name makes it obvious in the logs
        why a size came out where it did.
        """
        limits = self.limits
        detail: Dict[str, Decimal] = {}

        if price <= 0 or account.equity <= 0:
            return SizingResult(ZERO, "invalid inputs", detail)

        net_edge_bps = expected_edge_bps - limits.round_trip_cost_bps
        if net_edge_bps <= 0:
            return SizingResult(ZERO, "no edge after costs", detail)

        # 1. Volatility targeting.
        effective_vol = max(volatility_bps, Decimal("1"))
        risk_capital = account.equity * risk_per_trade
        vol_size = risk_capital / (price * effective_vol / BPS)
        detail["volTargetSize"] = vol_size

        # 2. Fractional Kelly. b = payoff ratio, approximated as the expected
        #    move over the expected adverse move (one volatility unit).
        payoff = net_edge_bps / effective_vol
        loss_probability = ONE - win_probability
        if payoff > 0:
            kelly = (win_probability * payoff - loss_probability) / payoff
        else:
            kelly = ZERO
        kelly = max(ZERO, min(kelly, ONE)) * kelly_fraction
        kelly_notional = account.equity * kelly * leverage
        kelly_size = kelly_notional / price
        detail["kellyFraction"] = kelly
        detail["kellySize"] = kelly_size

        # 3. Hard caps.
        headroom = max(ZERO, limits.max_position_base - abs(account.position_base))
        notional_cap = limits.max_notional / price
        cap_size = min(limits.max_order_base, headroom, notional_cap)
        detail["capSize"] = cap_size

        candidates = [
            ("volatility_target", vol_size),
            ("fractional_kelly", kelly_size),
            ("hard_limit", cap_size),
        ]
        name, size = min(candidates, key=lambda item: item[1])
        return SizingResult(max(ZERO, size), name, detail)

    # ---- reporting -------------------------------------------------------

    def status(self, account: AccountState, mark_price: Decimal) -> Dict[str, object]:
        """Compact snapshot for the dashboard / logs."""
        liq = None
        distance = None
        side = account.side
        if side is not None and account.entry_price > 0:
            liq = liquidation_price(
                entry_price=account.entry_price,
                leverage=account.leverage,
                side=side,
                maintenance_margin_rate=self.limits.maintenance_margin_rate,
                fee_buffer_bps=self.limits.fee_buffer_bps,
            )
            distance = liquidation_distance_pct(mark_price=mark_price, liq_price=liq)
        unrealized = account.unrealized_pnl(mark_price)
        return {
            "killSwitch": self.kill_switch_active,
            "killSwitchReason": self.kill_switch_reason,
            "connected": self.connected,
            "positionBase": float(account.position_base),
            "entryPrice": float(account.entry_price),
            "leverage": float(account.leverage),
            "liquidationPrice": float(liq) if liq is not None else None,
            "liquidationDistancePct": float(distance) if distance is not None else None,
            "realizedPnlToday": float(account.realized_pnl_today),
            "unrealizedPnl": float(unrealized),
            "totalPnlToday": float(account.realized_pnl_today + unrealized),
            "consecutiveLosses": account.consecutive_losses,
        }
