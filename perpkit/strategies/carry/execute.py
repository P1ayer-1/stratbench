"""Puts a carry on and takes it off. Both directions, and the minutes between.

`plan.py` decides what the position should be. This sends it, and its real job
is the stretch in the middle where only one leg exists.

Both directions size the legs so that the moment of maximum exposure is as
short and as cheap as possible, but they reach opposite orderings, because
opening and closing fail differently:

    open()   perp first  - a failed second leg leaves a short that closes
                           instantly, and it is unwound automatically
    close()  spot first  - a failed FIRST leg leaves nothing at all, because
                           the position is still fully hedged

`close()` carries its own reasoning; the rest of this docstring is about the
open.

Order of operations, and why
----------------------------
**Perp first.** Between sending the two legs the position is directional, so
the question is which exposure you would rather be holding when something goes
wrong. Sending the perp first means a failed spot leg leaves you SHORT the
perp - a position that closes instantly, on the venue's deepest book, with a
`reduce_only` order the risk engine will always approve. Sending spot first
means a failed perp leg leaves you LONG spot, unwindable only by selling into
the wider book, and holding an asset rather than a contract.

The perp leg also carries the liquidation risk and the tighter spread, so it
is the leg whose fill price matters more and the one worth getting on while
the plan's prices are freshest.

**Unwind, never hope.** If the spot leg fails after the perp filled, this
closes the perp immediately rather than retrying the spot. A carry that is
half on is not a carry, it is a naked directional position nobody decided to
take, and the longer it lives the more it stops being an execution problem and
starts being a trading one.

**Verify against the exchange, not against the plan.** After both legs are on,
the actual position is read back and its liquidation price compared to the
planned one. That check exists because maintenance margin is TIERED and
instrument-specific - measured at 0.500% on SOL-USDT and 0.300% on BTC-USDT -
so the rate a plan assumed is a guess until the exchange prices the position
you actually opened. A real liquidation closer than planned is a reason to
know immediately, not at 3am.

Dry run by default
------------------
`dry_run=True` unless a caller explicitly says otherwise. Every step reports
what it would send and returns the same result shape, so the difference
between a rehearsal and the real thing is one flag and nothing else.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol

from .plan import CarryPlan, round_down

ZERO = Decimal("0")


class Broker(Protocol):
    """The exchange operations an open needs. Implemented for real in
    `run_carry.py`, and faked in the tests, so none of the logic below has
    ever needed a network to be exercised."""

    def transfer(self, *, currency: str, amount: Decimal,
                 from_account: str, to_account: str) -> Dict[str, Any]: ...

    def margin_mode(self) -> str: ...

    def set_leverage(self, inst_id: str, leverage: Decimal) -> Dict[str, Any]: ...

    def place_perp(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str,
                   reduce_only: bool = False) -> Dict[str, Any]: ...

    def place_spot(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str) -> Dict[str, Any]: ...

    def perp_position(self, inst_id: str) -> Optional[Dict[str, Any]]: ...

    def spot_balance(self, currency: str) -> Decimal: ...


@dataclass
class Step:
    """One thing done or one thing that would have been done."""

    name: str
    detail: str
    sent: bool = False
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ExecutionResult:
    inst_id: str
    dry_run: bool
    steps: List[Step] = field(default_factory=list)
    opened: bool = False
    unwound: bool = False
    problems: List[str] = field(default_factory=list)

    perp_position: Optional[Dict[str, Any]] = None
    planned_liquidation: Optional[Decimal] = None
    actual_liquidation: Optional[Decimal] = None
    actual_mmr: Optional[Decimal] = None
    spot_acquired: Optional[Decimal] = None

    # Closing
    closed: bool = False
    already_flat: bool = False
    spot_sold: Optional[Decimal] = None
    perp_closed: Optional[Decimal] = None
    dust_base: Decimal = ZERO

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    @property
    def ok(self) -> bool:
        return not self.problems


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None


def _filled(response: Optional[Dict[str, Any]]) -> bool:
    """Did an order actually result in a position or balance change?

    BloFin returns a per-order `code` inside `data`; a transport-level success
    with a rejected order is the failure mode worth catching, because it looks
    like success at the top level.
    """
    if not isinstance(response, dict):
        return False
    if str(response.get("code")) not in ("0", "None", ""):
        return False
    rows = response.get("data")
    if isinstance(rows, list) and rows:
        first = rows[0]
        if isinstance(first, dict) and str(first.get("code", "0")) not in ("0", ""):
            return False
    return True


class CarryExecutor:
    """Turns a `CarryPlan` into orders. Dry by default."""

    def __init__(self, broker: Broker, *, dry_run: bool = True,
                 on_log: Optional[Callable[[str], Any]] = None,
                 settle_seconds: float = 1.0,
                 sleep: Callable[[float], Any] = time.sleep):
        self.broker = broker
        self.dry_run = dry_run
        self.on_log = on_log or (lambda message: None)
        self.settle_seconds = settle_seconds
        self.sleep = sleep

    def log(self, message: str) -> None:
        self.on_log(message)

    def _do(self, result: ExecutionResult, name: str, detail: str,
            action: Callable[[], Dict[str, Any]]) -> Step:
        step = Step(name=name, detail=detail)
        self.log(f"  {'[dry] ' if self.dry_run else ''}{name}: {detail}")
        if self.dry_run:
            return result.add(step)
        try:
            step.response = action()
            step.sent = True
            if not _filled(step.response):
                step.error = f"rejected: {step.response}"
        except Exception as exc:  # noqa: BLE001 - reported, never raised through
            step.error = f"{type(exc).__name__}: {exc}"
        return result.add(step)

    def open(self, plan: CarryPlan, *, base_currency: str) -> ExecutionResult:
        """Put the planned carry on, or leave nothing behind trying."""
        result = ExecutionResult(inst_id=plan.inst_id, dry_run=self.dry_run,
                                 planned_liquidation=plan.liquidation_price)

        if not plan.ok:
            result.problems.append(
                "plan was refused; nothing to execute: "
                + "; ".join(plan.reasons))
            return result

        tag = uuid.uuid4().hex[:16]
        # Recorded before anything is sent, so the spot leg can be verified by
        # what the BALANCE did rather than by what the order response said.
        # `size` on a spot market order means base units or quote depending on
        # a parameter the venue lets you omit, so a fill that reports success
        # is not yet evidence the hedge is the right size.
        base_before = (ZERO if self.dry_run
                       else self.broker.spot_balance(base_currency))

        # ---- funding the wallets ------------------------------------------
        if plan.spot_transfer_usd > 0:
            self._do(result, "transfer",
                     f"${plan.spot_transfer_usd:.2f} USDT futures -> spot",
                     lambda: self.broker.transfer(
                         currency="USDT", amount=plan.spot_transfer_usd,
                         from_account="futures", to_account="spot"))
        if plan.futures_transfer_usd > 0:
            self._do(result, "transfer",
                     f"${plan.futures_transfer_usd:.2f} USDT spot -> futures",
                     lambda: self.broker.transfer(
                         currency="USDT", amount=plan.futures_transfer_usd,
                         from_account="spot", to_account="futures"))

        if any(not step.ok for step in result.steps):
            result.problems.append("wallet transfer failed; no orders sent")
            return result

        # ---- margin configuration -----------------------------------------
        # Margin mode is checked, never set. On BloFin it is an ACCOUNT-wide
        # setting, so flipping it here to suit one carry would silently
        # re-margin every other open position - the sort of side effect an
        # execution path must not have. If it is wrong, say so and stop.
        mode = self.broker.margin_mode()
        if mode != "isolated":
            result.problems.append(
                f"account margin mode is '{mode}', and the plan priced "
                f"isolated. Changing it is an ACCOUNT-wide switch that would "
                f"re-margin every other open position, so it is not done "
                f"here - switch it deliberately, then re-run.")
            return result
        self.log(f"  margin mode: {mode} (checked, not changed)")

        self._do(result, "leverage", f"{plan.inst_id} -> {plan.leverage}x",
                 lambda: self.broker.set_leverage(plan.inst_id, plan.leverage))

        if any(not step.ok for step in result.steps):
            result.problems.append(
                "could not set leverage; no orders sent")
            return result

        # ---- leg 1: the perp short ----------------------------------------
        perp_step = self._do(
            result, "perp",
            f"SELL {plan.perp_contracts} contracts, isolated {plan.leverage}x",
            lambda: self.broker.place_perp(
                inst_id=plan.inst_id, side="sell",
                size=plan.perp_contracts,
                client_order_id=f"carry{tag}p"))

        if not perp_step.ok:
            # Nothing is on. This is the cheap failure and the reason the perp
            # goes first.
            result.problems.append(f"perp leg failed: {perp_step.error}")
            return result

        # ---- leg 2: the spot buy ------------------------------------------
        spot_step = self._do(
            result, "spot", f"BUY {plan.spot_base} {base_currency}",
            lambda: self.broker.place_spot(
                inst_id=plan.inst_id, side="buy", size=plan.spot_base,
                client_order_id=f"carry{tag}s"))

        if not spot_step.ok:
            result.problems.append(f"spot leg failed: {spot_step.error}")
            self._unwind(result, plan)
            return result

        result.opened = True

        # ---- verification against the exchange -----------------------------
        if not self.dry_run:
            self.sleep(self.settle_seconds)
            self._verify(result, plan)
            self._verify_spot(result, plan, base_currency, base_before)
        return result

    def close(self, inst_id: str, *, base_currency: str,
              spot_lot_size: Decimal = ZERO,
              retries: int = 2) -> ExecutionResult:
        """Take the carry off. Spot leg first, then the perp.

        Sized from the EXCHANGE, never from the plan
        --------------------------------------------
        The plan that opened this is weeks old and both legs have moved since:
        fees came out of the spot balance, funding moved the margin, and a
        partial fill anywhere would have moved the rest. So the close reads
        what is actually there and closes that.

        The useful consequence is that this is idempotent. Interrupted after
        one leg, run it again - it sizes from whatever is left and finishes
        the job, with no memory of the attempt that failed.

        Spot first, and why it is not the open's order
        ----------------------------------------------
        The open sends the perp first so a failed spot leg leaves a short that
        closes instantly. The close sends spot first for a stronger reason:
        **order the legs so a first-leg failure is a no-op rather than a
        position.**

        The spot sell is the leg that can actually be refused. It has no
        `reduce_only` protection, it needs a real balance, and the fee
        convention on a sell is not yet measured - the BUY was charged in base
        currency, and if a sell is too, then selling the whole balance is
        short of its own fee. If that order bounces, nothing has happened:
        still hedged, still collecting funding, retry costs nothing.

        Once the spot is gone, what remains is a `reduce_only` perp buy, which
        is the most reliable order this system can send - deepest book, always
        approved by the risk engine, and it cannot overshoot into a long.

        Never unwind a close
        --------------------
        `open()` refuses to leave half a carry on, because half a carry is a
        directional bet nobody decided to take. This must NOT inherit that:
        undoing a close means re-opening the position you just decided to
        exit. So a failed spot leg aborts having sent nothing, and a failed
        perp leg retries and then screams. Re-buying spot to re-hedge is
        deliberately not done - an executor that opens risk during a close is
        a surprise, and if the venue is rejecting orders the re-hedge is as
        likely to fail as the retry it replaced.
        """
        result = ExecutionResult(inst_id=inst_id, dry_run=self.dry_run)
        tag = uuid.uuid4().hex[:16]

        position = self.broker.perp_position(inst_id)
        result.perp_position = position
        contracts = _decimal(position.get("positions")) if position else ZERO
        contracts = contracts or ZERO
        balance = self.broker.spot_balance(base_currency)

        sellable = (round_down(balance, spot_lot_size)
                    if spot_lot_size > 0 else balance)
        result.dust_base = balance - sellable

        if contracts == 0 and sellable <= 0:
            result.already_flat = True
            result.closed = True
            self.log("  nothing open: no perp position and no spot balance")
            return result

        self.log(f"  closing {inst_id}: {contracts} contracts, "
                 f"{sellable} {base_currency}")

        # ---- leg 1: sell the spot, while the pair is still hedged ---------
        if sellable > 0:
            spot_step = self._do(
                result, "spot", f"SELL {sellable} {base_currency}",
                lambda: self.broker.place_spot(
                    inst_id=inst_id, side="sell", size=sellable,
                    client_order_id=f"carry{tag}x"))
            if not spot_step.ok:
                # Nothing was sent that changed anything. The carry is intact.
                result.problems.append(
                    f"spot leg failed: {spot_step.error}. Nothing else was "
                    f"sent - the position is untouched and still hedged, so "
                    f"this is safe to retry.")
                return result
            result.spot_sold = sellable

        # ---- leg 2: buy the perp back, reduce_only ------------------------
        if contracts != 0:
            size = abs(contracts)
            side = "buy" if contracts < 0 else "sell"
            step = None
            for attempt in range(1, max(1, retries) + 1):
                step = self._do(
                    result, "perp",
                    f"{side.upper()} {size} contracts, reduce_only"
                    + (f" (attempt {attempt})" if attempt > 1 else ""),
                    lambda: self.broker.place_perp(
                        inst_id=inst_id, side=side, size=size,
                        client_order_id=f"carry{uuid.uuid4().hex[:12]}x",
                        reduce_only=True))
                if step.ok:
                    result.perp_closed = size
                    break
                self.log(f"  perp close rejected, retrying ({attempt})")
            if step is not None and not step.ok:
                result.problems.append(
                    f"PERP CLOSE FAILED after {retries} attempts and the spot "
                    f"leg is already sold. A naked {size} contract short is "
                    f"open on {inst_id} with nothing hedging it. Close it by "
                    f"hand now: {side} {size} contracts, reduce_only.")
                return result

        if self.dry_run:
            result.closed = True
            return result

        self.sleep(self.settle_seconds)
        self._verify_flat(result, inst_id, base_currency)
        return result

    def _verify_flat(self, result: ExecutionResult, inst_id: str,
                     base_currency: str) -> None:
        """Both legs gone, checked against the exchange rather than assumed.

        An accepted order is not a filled one, and a market order that fills
        partially leaves exactly the exposure this whole exercise was meant to
        remove.
        """
        position = self.broker.perp_position(inst_id)
        remaining = _decimal(position.get("positions")) if position else ZERO
        remaining = remaining or ZERO
        if remaining != 0:
            result.problems.append(
                f"perp close was accepted but {remaining} contracts are still "
                f"open. Re-run to finish, or close by hand.")

        left = self.broker.spot_balance(base_currency)
        if result.spot_sold is not None and left > result.dust_base:
            result.problems.append(
                f"spot sell was accepted but {left} {base_currency} is still "
                f"held against {result.dust_base} of expected dust - the "
                f"order filled partially. Re-run to finish.")

        result.closed = not result.problems

    def _unwind(self, result: ExecutionResult, plan: CarryPlan) -> None:
        """Close the perp leg, because half a carry is a directional bet.

        `reduce_only`, so the risk engine approves it even with the kill
        switch tripped - getting out is never the thing to block.
        """
        self.log("  spot leg failed - unwinding the perp leg")
        step = self._do(
            result, "unwind",
            f"BUY {plan.perp_contracts} contracts, reduce_only",
            lambda: self.broker.place_perp(
                inst_id=plan.inst_id, side="buy", size=plan.perp_contracts,
                client_order_id=f"carry{uuid.uuid4().hex[:12]}u",
                reduce_only=True))
        if step.ok:
            result.unwound = True
        else:
            result.problems.append(
                "UNWIND FAILED - a naked short perp is open on "
                f"{plan.inst_id}. Close it by hand now.")

    def _verify_spot(self, result: ExecutionResult, plan: CarryPlan,
                     base_currency: str, before: Decimal) -> None:
        """Did the spot leg actually buy the amount the hedge needs?

        Checked against the balance rather than the order response, because
        the failure this guards is a units mismatch: `targetCurrency` decides
        whether a market order's `size` is base or quote, and the wrong one
        fills successfully at the wrong size. An order response cannot show
        that; a balance can.
        """
        after = self.broker.spot_balance(base_currency)
        acquired = after - before
        result.spot_acquired = acquired

        if plan.spot_base <= 0:
            return
        drift = abs(acquired - plan.spot_base) / plan.spot_base
        if drift > Decimal("0.02"):
            result.problems.append(
                f"spot leg bought {acquired} {base_currency}, plan needed "
                f"{plan.spot_base} ({drift:.1%} off). The hedge is the wrong "
                f"size - check targetCurrency on the order.")

    def _verify(self, result: ExecutionResult, plan: CarryPlan) -> None:
        """Read the position back and price it the way the exchange does.

        The plan assumed a maintenance margin rate. MMR is tiered and
        instrument-specific, so that assumption is only a guess until a real
        position exists - at which point the exchange publishes both its own
        liquidation price and the maintenance margin it is charging.
        """
        position = self.broker.perp_position(plan.inst_id)
        result.perp_position = position
        if not position:
            result.problems.append(
                "perp order was accepted but no position is reported")
            return

        contracts = _decimal(position.get("positions")) or ZERO
        if contracts >= 0:
            result.problems.append(
                f"expected a short perp position, found {contracts}")

        actual = _decimal(position.get("liquidationPrice"))
        mark = _decimal(position.get("markPrice"))
        maintenance = _decimal(position.get("maintenanceMargin"))
        result.actual_liquidation = actual
        if maintenance and mark and contracts:
            notional = abs(contracts) * mark
            if notional > 0:
                result.actual_mmr = maintenance / notional

        if actual and plan.liquidation_price and mark:
            planned_distance = abs(plan.liquidation_price - mark) / mark
            actual_distance = abs(actual - mark) / mark
            if actual_distance < planned_distance * Decimal("0.95"):
                result.problems.append(
                    f"liquidation is CLOSER than planned: {actual_distance:.2%} "
                    f"actual vs {planned_distance:.2%} planned. The plan's "
                    f"maintenance margin rate was wrong for this instrument.")
