"""Plans a delta-neutral funding carry. Computes orders; never sends them.

    LONG spot  +  SHORT perp  ->  collects funding while price-neutral

`perpkit/analysis/funding_carry.py` screens which instruments might be worth
carrying and `perpkit/analysis/carry_backtest.py` says what one would have
earned. Neither says
what to actually *do*: how many contracts, how much margin, where the short
leg gets liquidated, whether the account even holds the right currency in the
right wallet. This does, and stops there.

Nothing here places an order. `plan_carry.py` prints what this returns. That
separation is deliberate - every sizing and margin question gets answered and
reviewed while the answer is still only text.

Why the plan is the hard part
-----------------------------
**Delta neutrality survives rounding or it does not exist.** The spot leg
trades in base units on one lot size; the perp leg trades in contracts on
another, each worth `contractValue` of base. Round the two legs independently
and the position is quietly directional by the difference. So the perp leg is
sized first in whole lots, the spot leg is derived from it, and whatever
mismatch the spot lot size forces is reported in base and in dollars rather
than left for the market to reveal.

**The short leg can be liquidated on its own.** Delta-neutral is not
risk-neutral: the two legs margin separately, and a rally that leaves the pair
flat can still take the perp leg out. Once that happens the hedge is gone and
what remains is an unhedged long spot position - the opposite of the trade.
The liquidation price here comes from `risk.liquidation_price`, which matched
BloFin's own number to 0.057% on a small position, with the MMR and fee buffer
that comparison measured. Re-check with `risk.compare_to_exchange` at your
size.

**Isolated, not cross.** Cross margin would back the perp leg with the whole
account including the spot leg's cash, which reads as a wider liquidation
buffer and is really just a larger blast radius. Isolated keeps the failure
contained to the margin posted for it.

What it deliberately does not do
--------------------------------
No order type, no execution schedule, no leg sequencing. Which leg to send
first is a real decision - whichever fills first leaves you directional until
the other does - and it belongs to the executor that sends them, not to the
arithmetic that sizes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional

from ...risk import (
    ONE,
    ZERO,
    RiskLimits,
    Side,
    liquidation_distance_pct,
    liquidation_price,
)

BPS = Decimal("10000")


def round_down(value: Decimal, step: Decimal) -> Decimal:
    """Largest multiple of `step` not exceeding `value`.

    Down, never nearest: rounding a leg UP means buying size the plan did not
    budget capital for, and on the perp leg it means margin the account may
    not have.
    """
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding="ROUND_FLOOR") * step


@dataclass
class CarryPlan:
    """Everything needed to open one carry, and every reason not to."""

    inst_id: str
    ok: bool = False
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Legs
    perp_contracts: Decimal = ZERO
    perp_base: Decimal = ZERO
    spot_base: Decimal = ZERO
    residual_base: Decimal = ZERO
    residual_usd: Decimal = ZERO

    # Prices used
    spot_ask: Decimal = ZERO
    perp_bid: Decimal = ZERO
    notional_usd: Decimal = ZERO

    # Capital
    spot_cost_usd: Decimal = ZERO
    perp_margin_usd: Decimal = ZERO
    total_capital_usd: Decimal = ZERO
    spot_transfer_usd: Decimal = ZERO
    futures_transfer_usd: Decimal = ZERO

    # Risk
    leverage: Decimal = ZERO
    liquidation_price: Optional[Decimal] = None
    liquidation_distance: Optional[Decimal] = None

    spot_fee_rate: Decimal = ZERO
    spot_after_fee: Decimal = ZERO

    # Economics
    round_trip_bps: Decimal = ZERO
    funding_per_day_bps: Decimal = ZERO
    breakeven_days: Optional[Decimal] = None
    expected_net_bps: Decimal = ZERO
    expected_net_usd: Decimal = ZERO


@dataclass
class Market:
    """Live prices and the exchange's own size rules for one instrument."""

    inst_id: str
    spot_bid: Decimal
    spot_ask: Decimal
    perp_bid: Decimal
    perp_ask: Decimal
    contract_value: Decimal          # base units per perp contract
    perp_lot_size: Decimal           # contracts
    perp_min_size: Decimal           # contracts
    spot_lot_size: Decimal           # base units
    spot_min_size: Decimal           # base units

    @property
    def spot_mid(self) -> Decimal:
        return (self.spot_bid + self.spot_ask) / 2

    @property
    def perp_mid(self) -> Decimal:
        return (self.perp_bid + self.perp_ask) / 2

    @property
    def spot_spread_bps(self) -> Decimal:
        return (self.spot_ask - self.spot_bid) / self.spot_mid * BPS

    @property
    def perp_spread_bps(self) -> Decimal:
        return (self.perp_ask - self.perp_bid) / self.perp_mid * BPS


@dataclass
class Wallets:
    """USDT actually available, per wallet. Carry needs both funded."""

    spot_usdt: Decimal
    futures_usdt: Decimal


def plan_carry(
    *,
    market: Market,
    wallets: Wallets,
    target_notional_usd: Decimal,
    leverage: Decimal,
    funding_per_day_bps: Decimal,
    hold_days: Decimal,
    spot_taker_bps: Decimal,
    perp_taker_bps: Decimal,
    limits: Optional[RiskLimits] = None,
    maintenance_margin_rate: Decimal = Decimal("0.005"),
    fee_buffer_bps: Decimal = Decimal("6"),
) -> CarryPlan:
    """Size both legs, price the risk, and say whether to do it.

    `maintenance_margin_rate` and `fee_buffer_bps` default to what a comparison
    of `risk.liquidation_price` with the exchange's own number measured on a
    small real position. MMR is tiered by size, so a plan much
    larger than the position that measurement came from should re-measure.
    """
    limits = limits or RiskLimits()
    plan = CarryPlan(inst_id=market.inst_id, leverage=leverage,
                     funding_per_day_bps=funding_per_day_bps)

    if market.spot_ask <= 0 or market.perp_bid <= 0:
        plan.reasons.append("no usable quote on one of the legs")
        return plan
    if target_notional_usd <= 0:
        plan.reasons.append("target notional must be positive")
        return plan
    if leverage <= 0:
        plan.reasons.append("leverage must be positive")
        return plan

    plan.spot_ask = market.spot_ask
    plan.perp_bid = market.perp_bid

    # ---- sizing, perp first ------------------------------------------------
    # The perp leg has the coarser unit (a contract is `contract_value` of
    # base), so it sets the achievable granularity. Deriving spot from it and
    # rounding second keeps the mismatch to one spot lot instead of one
    # contract.
    wanted_base = target_notional_usd / market.perp_mid
    contracts = round_down(wanted_base / market.contract_value,
                           market.perp_lot_size)
    if contracts < market.perp_min_size or contracts <= 0:
        plan.reasons.append(
            f"target notional buys {contracts} contracts, below the "
            f"{market.perp_min_size} minimum")
        return plan

    plan.perp_contracts = contracts
    plan.perp_base = contracts * market.contract_value
    # The spot taker fee is charged in the BASE currency, so buying exactly
    # the perp's base leaves the hedge short by the fee: a 254-unit buy at
    # 0.1% delivered 253.746 against a 254 short. Measured on a real fill, not
    # inferred. So buy the amount that NETS to the hedge after the fee.
    plan.spot_fee_rate = spot_taker_bps / BPS
    gross = plan.perp_base / (ONE - plan.spot_fee_rate)
    plan.spot_base = round_down(gross, market.spot_lot_size)
    plan.spot_after_fee = plan.spot_base * (ONE - plan.spot_fee_rate)
    if plan.spot_base < market.spot_min_size:
        plan.reasons.append(
            f"spot leg {plan.spot_base} is below the {market.spot_min_size} "
            f"minimum")
        return plan

    # Whatever the two lot sizes could not reconcile is naked directional
    # exposure, so it is measured rather than assumed away.
    plan.residual_base = plan.perp_base - plan.spot_after_fee
    plan.residual_usd = abs(plan.residual_base) * market.spot_mid
    plan.notional_usd = plan.perp_base * market.perp_mid

    if plan.residual_usd > plan.notional_usd * Decimal("0.001"):
        plan.warnings.append(
            f"legs differ by {plan.residual_base} base "
            f"(${plan.residual_usd:.2f}) after rounding - that much is "
            f"directional")

    # ---- capital -----------------------------------------------------------
    plan.spot_cost_usd = plan.spot_base * market.spot_ask
    plan.perp_margin_usd = plan.notional_usd / leverage
    plan.total_capital_usd = plan.spot_cost_usd + plan.perp_margin_usd

    if wallets.spot_usdt < plan.spot_cost_usd:
        plan.spot_transfer_usd = plan.spot_cost_usd - wallets.spot_usdt
    if wallets.futures_usdt < plan.perp_margin_usd:
        plan.futures_transfer_usd = plan.perp_margin_usd - wallets.futures_usdt

    shortfall = (plan.spot_transfer_usd + plan.futures_transfer_usd
                 - max(ZERO, wallets.spot_usdt - plan.spot_cost_usd)
                 - max(ZERO, wallets.futures_usdt - plan.perp_margin_usd))
    if shortfall > 0:
        plan.reasons.append(
            f"not enough USDT: need ${plan.total_capital_usd:.2f} across both "
            f"wallets, hold ${wallets.spot_usdt + wallets.futures_usdt:.2f}")

    # ---- risk on the short leg --------------------------------------------
    # Isolated: the margin posted for the perp leg is all that backs it, which
    # is the point. Cross would read as a safer buffer by putting the spot
    # leg's cash behind it too.
    plan.liquidation_price = liquidation_price(
        entry_price=market.perp_bid,
        leverage=leverage,
        side=Side.SHORT,
        maintenance_margin_rate=maintenance_margin_rate,
        fee_buffer_bps=fee_buffer_bps,
    )
    plan.liquidation_distance = liquidation_distance_pct(
        mark_price=market.perp_mid, liq_price=plan.liquidation_price)

    if plan.liquidation_distance is None:
        plan.reasons.append("liquidation price could not be estimated")
    elif plan.liquidation_distance < limits.min_liquidation_buffer_pct:
        plan.reasons.append(
            f"liquidation only {plan.liquidation_distance:.1%} away, needs "
            f"{limits.min_liquidation_buffer_pct:.1%} - lower the leverage")

    if leverage > limits.max_leverage:
        plan.reasons.append(
            f"leverage {leverage} exceeds the {limits.max_leverage} limit")
    if plan.notional_usd > limits.max_notional:
        plan.reasons.append(
            f"notional ${plan.notional_usd:.2f} exceeds the "
            f"${limits.max_notional} limit")

    # ---- economics ---------------------------------------------------------
    plan.round_trip_bps = (market.spot_spread_bps + market.perp_spread_bps
                           + 2 * (spot_taker_bps + perp_taker_bps))
    if funding_per_day_bps > 0:
        plan.breakeven_days = plan.round_trip_bps / funding_per_day_bps
    else:
        plan.reasons.append(
            "funding is not positive, so there is nothing to collect - the "
            "spot leg cannot be shorted to harvest the other sign")
    plan.expected_net_bps = (funding_per_day_bps * hold_days
                             - plan.round_trip_bps)
    plan.expected_net_usd = plan.expected_net_bps / BPS * plan.notional_usd

    if plan.expected_net_bps <= 0:
        plan.reasons.append(
            f"expected net over {hold_days} days is "
            f"{plan.expected_net_bps:+.1f} bps - the round trip is not repaid")

    if plan.breakeven_days is not None and plan.breakeven_days > hold_days:
        plan.warnings.append(
            f"break-even is {plan.breakeven_days:.1f} days against a "
            f"{hold_days} day hold")

    plan.ok = not plan.reasons
    return plan
