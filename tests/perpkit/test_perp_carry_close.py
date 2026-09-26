"""Taking a carry off, and the stretch where one leg is already gone.

The close is not the open run backwards. It sizes from the exchange rather
than from the plan, it sends the spot leg FIRST so that a refusal costs
nothing, and it must never "unwind" - undoing a close means re-opening the
position somebody just decided to exit.

No network. The broker is a protocol so every failure can be ordered on
demand.
"""

from decimal import Decimal

import pytest

from perpkit.strategies.carry import CarryExecutor

OK = {"code": "0", "data": [{"code": "0", "orderId": "1"}]}
REJECTED = {"code": "0", "data": [{"code": "103", "msg": "insufficient"}]}


def position(contracts="-254", mark="0.78"):
    return {"instId": "SUI-USDT", "positions": contracts, "markPrice": mark,
            "averagePrice": "0.7858", "positionId": "p1"}


class FakeBroker:
    """Holds a position and a balance, and lets them be moved or refused."""

    def __init__(self, *, contracts="-254", balance="253.746", spot=OK,
                 perp=OK, perp_failures=0, spot_fills=True,
                 perp_leaves=Decimal("0")):
        self.contracts = Decimal(contracts)
        self.balance = Decimal(balance)
        self.spot_response = spot
        self.perp_response = perp
        # How many perp attempts fail before one succeeds.
        self.perp_failures = perp_failures
        self.spot_fills = spot_fills
        # What the position is left at after a "successful" perp close, so a
        # partial fill can be simulated.
        self.perp_leaves = perp_leaves
        self.calls = []

    # -- reads ----------------------------------------------------------
    def perp_position(self, inst_id):
        self.calls.append("perp_position")
        if self.contracts == 0:
            return None
        return position(str(self.contracts))

    def spot_balance(self, currency):
        self.calls.append("spot_balance")
        return self.balance

    # -- writes ---------------------------------------------------------
    def place_spot(self, *, inst_id, side, size, client_order_id):
        self.calls.append(f"spot:{side}:{size}")
        if self.spot_response is OK and self.spot_fills:
            self.balance -= size
        return self.spot_response

    def place_perp(self, *, inst_id, side, size, client_order_id,
                   reduce_only=False):
        self.calls.append(
            f"perp:{side}:{size}:{'reduce' if reduce_only else 'open'}")
        if self.perp_failures > 0:
            self.perp_failures -= 1
            return REJECTED
        if self.perp_response is OK:
            self.contracts = self.perp_leaves
        return self.perp_response

    # -- unused by close, present for the protocol ----------------------
    def transfer(self, **kwargs):
        return OK

    def margin_mode(self):
        return "isolated"

    def set_leverage(self, inst_id, leverage):
        return OK


def close(broker, *, dry_run=False, lot=Decimal("0.0001"), retries=2):
    executor = CarryExecutor(broker, dry_run=dry_run, sleep=lambda _s: None)
    return executor.close("SUI-USDT", base_currency="SUI",
                          spot_lot_size=lot, retries=retries)


# ---------------------------------------------------------------------------
# The happy path, and the ordering that defines it
# ---------------------------------------------------------------------------


def test_a_clean_close_sells_spot_then_buys_the_perp_back():
    broker = FakeBroker()
    result = close(broker)

    orders = [call for call in broker.calls if ":" in call]
    assert orders == ["spot:sell:253.746", "perp:buy:254:reduce"]
    assert result.closed and result.ok
    assert result.spot_sold == Decimal("253.746")
    assert result.perp_closed == Decimal("254")


def test_the_perp_leg_is_always_reduce_only():
    """It cannot overshoot into a long, and the risk engine always approves
    it - getting out is never the thing to block."""
    broker = FakeBroker()
    close(broker)
    assert any(call.endswith(":reduce") for call in broker.calls)
    assert not any(call.endswith(":open") for call in broker.calls)


def test_a_long_carry_closes_by_selling_the_perp():
    """Nothing here assumes the perp leg is short. A position of the other
    sign closes in the other direction."""
    broker = FakeBroker(contracts="254")
    close(broker)
    assert "perp:sell:254:reduce" in broker.calls


# ---------------------------------------------------------------------------
# Sizing comes from the exchange, never from the plan
# ---------------------------------------------------------------------------


def test_the_close_sizes_from_the_live_balance_not_from_the_open():
    """The plan is weeks old by now. Fees came out of the spot balance and
    funding moved the margin, so what to sell is a question only the exchange
    can answer."""
    broker = FakeBroker(contracts="-100", balance="99.5")
    result = close(broker)
    assert result.spot_sold == Decimal("99.5")
    assert result.perp_closed == Decimal("100")


def test_lot_rounding_strands_dust_and_says_so():
    broker = FakeBroker(contracts="-254", balance="253.74678")
    result = close(broker)
    assert result.spot_sold == Decimal("253.7467")
    assert result.dust_base == Decimal("0.00008")


def test_a_balance_that_is_an_exact_multiple_leaves_no_dust():
    result = close(FakeBroker(balance="253.746"))
    assert result.dust_base == Decimal("0")


def test_a_carry_already_flat_sends_nothing():
    broker = FakeBroker(contracts="0", balance="0")
    result = close(broker)
    assert result.already_flat and result.closed
    assert not [call for call in broker.calls if ":" in call]


# ---------------------------------------------------------------------------
# Idempotency: a half-closed carry finishes on the next run
# ---------------------------------------------------------------------------


def test_a_carry_with_only_a_perp_left_closes_just_the_perp():
    broker = FakeBroker(contracts="-254", balance="0")
    result = close(broker)
    assert broker.calls.count("spot:sell:0") == 0
    assert result.spot_sold is None
    assert result.perp_closed == Decimal("254")
    assert result.closed


def test_a_carry_with_only_spot_left_sells_just_the_spot():
    broker = FakeBroker(contracts="0", balance="253.746")
    result = close(broker)
    assert result.spot_sold == Decimal("253.746")
    assert result.perp_closed is None
    assert result.closed


def test_rerunning_after_an_interrupted_close_finishes_the_job():
    """The close keeps no memory of the attempt that failed, so finishing is
    just running it again."""
    broker = FakeBroker(perp_failures=99)          # perp will never go
    first = close(broker, retries=1)
    assert not first.ok and first.spot_sold is not None

    broker.perp_failures = 0                        # venue recovers
    second = close(broker)
    assert second.closed and second.ok
    assert second.spot_sold is None                 # nothing left to sell
    assert second.perp_closed == Decimal("254")


# ---------------------------------------------------------------------------
# Failure, and the rule that a close is never unwound
# ---------------------------------------------------------------------------


def test_a_refused_spot_leg_touches_nothing_else():
    """The reason spot goes first. If it bounces, the carry is still fully
    hedged and nothing has been lost."""
    broker = FakeBroker(spot=REJECTED)
    result = close(broker)

    assert not result.ok
    assert not any(call.startswith("perp:") for call in broker.calls)
    assert result.perp_closed is None
    assert broker.contracts == Decimal("-254")      # position intact
    assert "still hedged" in result.problems[0]


def test_a_failed_close_is_never_unwound_by_rebuying_spot():
    """Undoing a close means re-opening the carry somebody just decided to
    exit. Whatever happens, no BUY is ever sent."""
    broker = FakeBroker(perp_failures=99)
    close(broker, retries=2)
    assert not any(call.startswith("spot:buy") for call in broker.calls)


def test_a_perp_close_that_fails_after_the_spot_sold_screams():
    broker = FakeBroker(perp_failures=99)
    result = close(broker, retries=2)

    assert not result.ok
    problem = result.problems[0]
    assert "naked" in problem
    assert "by hand" in problem
    assert "reduce_only" in problem


def test_the_perp_close_is_retried_before_it_is_given_up_on():
    broker = FakeBroker(perp_failures=1)            # first attempt bounces
    result = close(broker, retries=2)

    assert result.closed and result.ok
    assert len([c for c in broker.calls if c.startswith("perp:")]) == 2


def test_a_broker_that_raises_is_caught_not_propagated():
    broker = FakeBroker()

    def explode(**kwargs):
        raise ConnectionError("gateway")

    broker.place_spot = explode
    result = close(broker)
    assert not result.ok
    assert "ConnectionError" in result.problems[0]


# ---------------------------------------------------------------------------
# Verification against the exchange
# ---------------------------------------------------------------------------


def test_a_perp_that_is_still_open_afterwards_is_a_problem():
    """An accepted order is not a filled one."""
    broker = FakeBroker(perp_leaves=Decimal("-54"))
    result = close(broker)

    assert not result.ok
    assert not result.closed
    assert "54 contracts are still open" in result.problems[0]


def test_a_spot_sell_that_only_partly_filled_is_a_problem():
    broker = FakeBroker(spot_fills=False)           # balance does not move
    result = close(broker)

    assert not result.ok
    assert "filled partially" in " ".join(result.problems)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_a_dry_run_sends_nothing_at_all():
    broker = FakeBroker()
    result = close(broker, dry_run=True)

    assert not [call for call in broker.calls if ":" in call]
    assert broker.contracts == Decimal("-254")
    assert broker.balance == Decimal("253.746")
    assert result.dry_run


def test_a_dry_run_still_reads_live_state_and_sizes_from_it():
    """The whole value of rehearsing a close is seeing the real sizes."""
    broker = FakeBroker(contracts="-100", balance="99.5")
    result = close(broker, dry_run=True)

    assert "perp_position" in broker.calls
    assert "spot_balance" in broker.calls
    assert [step.detail for step in result.steps] == [
        "SELL 99.5 SUI", "BUY 100 contracts, reduce_only"]
