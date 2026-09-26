"""Opening a carry, and the two minutes where only one leg exists.

The happy path is the easy part. What these pin down is the failure ordering:
that a rejected spot leg unwinds the perp rather than leaving a naked short,
that a rejected perp leg leaves nothing behind, that margin mode is never
silently changed, and that the exchange's own liquidation price is checked
against the plan's rather than trusted to match.

No network. The broker is a protocol precisely so the sequencing logic can be
exercised against a fake that fails on command.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest

from perpkit.strategies.carry import CarryExecutor, CarryPlan

OK = {"code": "0", "data": [{"code": "0", "orderId": "1"}]}
REJECTED = {"code": "0", "data": [{"code": "103", "msg": "insufficient"}]}
TRANSPORT_ERROR = {"code": "500", "msg": "gateway"}


class FakeBroker:
    def __init__(self, *, perp=OK, spot=OK, transfer=OK, leverage=OK,
                 unwind=OK, mode="isolated", position=None,
                 spot_fills=Decimal("2534")):
        self.responses = {"perp": perp, "spot": spot, "transfer": transfer,
                          "leverage": leverage, "unwind": unwind}
        self.mode = mode
        self.position = position
        # What the spot leg actually adds to the balance, which is what the
        # executor checks rather than the order response.
        self.spot_fills = spot_fills
        self.base_held = Decimal("0")
        self.calls: List[str] = []

    def transfer(self, **kwargs) -> Dict[str, Any]:
        self.calls.append("transfer")
        return self.responses["transfer"]

    def margin_mode(self) -> str:
        self.calls.append("margin_mode")
        return self.mode

    def set_leverage(self, inst_id, leverage) -> Dict[str, Any]:
        self.calls.append("set_leverage")
        return self.responses["leverage"]

    def place_perp(self, *, inst_id, side, size, client_order_id,
                   reduce_only=False) -> Dict[str, Any]:
        self.calls.append(f"perp:{side}:{'reduce' if reduce_only else 'open'}")
        return self.responses["unwind" if reduce_only else "perp"]

    def place_spot(self, *, inst_id, side, size, client_order_id) -> Dict[str, Any]:
        self.calls.append(f"spot:{side}")
        response = self.responses["spot"]
        if response is OK:
            self.base_held += self.spot_fills
        return response

    def perp_position(self, inst_id) -> Optional[Dict[str, Any]]:
        self.calls.append("perp_position")
        return self.position

    def spot_balance(self, currency) -> Decimal:
        return self.base_held


def good_plan(**overrides) -> CarryPlan:
    defaults = dict(
        inst_id="SUI-USDT", ok=True,
        perp_contracts=Decimal("2534"), perp_base=Decimal("2534"),
        spot_base=Decimal("2534"), leverage=Decimal("3"),
        liquidation_price=Decimal("1.0461"),
    )
    defaults.update(overrides)
    return CarryPlan(**defaults)


def position(liquidation="1.0461", contracts="-2534", mark="0.789",
             maintenance="10.0"):
    return {"instId": "SUI-USDT", "positions": contracts,
            "liquidationPrice": liquidation, "markPrice": mark,
            "maintenanceMargin": maintenance}


def execute(broker, plan=None, *, dry_run=False):
    executor = CarryExecutor(broker, dry_run=dry_run, sleep=lambda _: None,
                             settle_seconds=0)
    return executor.open(plan or good_plan(), base_currency="SUI")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_the_perp_leg_goes_first():
    """The whole reason for the order: a failed second leg is recoverable."""
    broker = FakeBroker(position=position())
    execute(broker)

    perp_index = broker.calls.index("perp:sell:open")
    spot_index = broker.calls.index("spot:buy")
    assert perp_index < spot_index


def test_a_happy_open_reports_both_legs_and_verifies():
    broker = FakeBroker(position=position())
    result = execute(broker)

    assert result.opened
    assert result.ok, result.problems
    assert result.actual_liquidation == Decimal("1.0461")


# ---------------------------------------------------------------------------
# The failure that matters
# ---------------------------------------------------------------------------


def test_a_rejected_spot_leg_unwinds_the_perp():
    """Half a carry is a directional bet nobody decided to take."""
    broker = FakeBroker(spot=REJECTED, position=position())
    result = execute(broker)

    assert not result.opened
    assert result.unwound
    assert "perp:buy:reduce" in broker.calls


def test_the_unwind_is_reduce_only():
    """So the risk engine approves it even with the kill switch tripped."""
    broker = FakeBroker(spot=REJECTED)
    execute(broker)
    assert any(call.endswith(":reduce") for call in broker.calls)


def test_a_failed_unwind_screams():
    """The one state that needs a human immediately."""
    broker = FakeBroker(spot=REJECTED, unwind=REJECTED)
    result = execute(broker)

    assert not result.unwound
    assert any("UNWIND FAILED" in problem for problem in result.problems)
    assert any("naked short" in problem for problem in result.problems)


def test_a_rejected_perp_leg_sends_no_spot_order():
    """Nothing is on, which is the cheapest possible failure."""
    broker = FakeBroker(perp=REJECTED)
    result = execute(broker)

    assert not result.opened
    assert not result.unwound
    assert not any(call.startswith("spot:") for call in broker.calls)


def test_an_order_rejected_inside_a_200_response_is_still_a_rejection():
    """Transport success with a rejected order is the failure mode that
    looks most like success."""
    broker = FakeBroker(perp=REJECTED)
    assert not execute(broker).opened

    broker = FakeBroker(perp=TRANSPORT_ERROR)
    assert not execute(broker).opened


def test_a_broker_that_raises_is_caught_not_propagated():
    class Exploding(FakeBroker):
        def place_perp(self, **kwargs):
            raise ConnectionError("socket died")

    result = execute(Exploding())
    assert not result.opened
    assert any("ConnectionError" in problem for problem in result.problems)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_a_refused_plan_sends_nothing():
    plan = CarryPlan(inst_id="SUI-USDT", ok=False,
                     reasons=["liquidation too close"])
    broker = FakeBroker()
    result = execute(broker, plan)

    assert broker.calls == []
    assert "liquidation too close" in result.problems[0]


def test_cross_margin_stops_the_open_rather_than_being_switched():
    """Margin mode is account-wide on BloFin.

    Flipping it to suit one carry would silently re-margin every other open
    position, so it is checked and never set.
    """
    broker = FakeBroker(mode="cross")
    result = execute(broker)

    assert not result.opened
    assert not any(call.startswith("perp:") for call in broker.calls)
    assert any("ACCOUNT-wide" in problem for problem in result.problems)


def test_a_failed_transfer_stops_before_any_order():
    broker = FakeBroker(transfer=REJECTED)
    result = execute(broker, good_plan(spot_transfer_usd=Decimal("500")))

    assert not any(call.startswith(("perp:", "spot:")) for call in broker.calls)
    assert any("transfer failed" in problem for problem in result.problems)


def test_a_failed_leverage_set_stops_before_any_order():
    broker = FakeBroker(leverage=REJECTED)
    result = execute(broker)

    assert not any(call.startswith(("perp:", "spot:")) for call in broker.calls)
    assert any("leverage" in problem for problem in result.problems)


# ---------------------------------------------------------------------------
# Verification against the exchange
# ---------------------------------------------------------------------------


def test_a_liquidation_closer_than_planned_is_a_problem():
    """MMR is tiered and instrument-specific, so the plan's rate is a guess
    until a real position exists to price."""
    broker = FakeBroker(position=position(liquidation="0.85"))
    result = execute(broker)

    assert any("CLOSER than planned" in problem for problem in result.problems)


def test_a_liquidation_further_than_planned_is_fine():
    broker = FakeBroker(position=position(liquidation="1.30"))
    result = execute(broker)
    assert result.ok, result.problems


def test_the_actual_mmr_is_read_back():
    # 10 maintenance on 2534 * 0.789 notional
    broker = FakeBroker(position=position(maintenance="10.0"))
    result = execute(broker)

    expected = Decimal("10.0") / (Decimal("2534") * Decimal("0.789"))
    assert result.actual_mmr == pytest.approx(expected)


def test_an_accepted_order_with_no_resulting_position_is_a_problem():
    broker = FakeBroker(position=None)
    result = execute(broker)
    assert any("no position is reported" in problem for problem in result.problems)


def test_a_long_position_where_a_short_was_expected_is_a_problem():
    broker = FakeBroker(position=position(contracts="2534"))
    result = execute(broker)
    assert any("expected a short" in problem for problem in result.problems)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_a_dry_run_sends_nothing_at_all():
    broker = FakeBroker(position=position())
    result = execute(broker, dry_run=True)

    assert result.dry_run
    assert not any(call.startswith(("perp:", "spot:", "transfer"))
                   for call in broker.calls)
    assert all(not step.sent for step in result.steps)


def test_a_dry_run_still_walks_every_step():
    broker = FakeBroker(position=position())
    result = execute(broker, good_plan(spot_transfer_usd=Decimal("500")),
                     dry_run=True)

    names = [step.name for step in result.steps]
    assert "transfer" in names
    assert "leverage" in names
    assert "perp" in names
    assert "spot" in names


def test_a_dry_run_still_refuses_a_bad_margin_mode():
    """A rehearsal that ignores preconditions rehearses the wrong thing."""
    broker = FakeBroker(mode="cross", position=position())
    result = execute(broker, dry_run=True)
    assert any("ACCOUNT-wide" in problem for problem in result.problems)


# ---------------------------------------------------------------------------
# The hedge is the right SIZE, not merely filled
# ---------------------------------------------------------------------------


def test_a_spot_fill_of_the_wrong_size_is_caught():
    """The failure `targetCurrency` causes, and the reason the check reads the
    balance rather than the order response.

    Quoting size in the quote currency instead of the base fills cleanly and
    reports success, while buying roughly 1/price of the intended hedge.
    """
    broker = FakeBroker(position=position(),
                        spot_fills=Decimal("323"))   # 254 USDT of SUI, not 254 SUI
    result = execute(broker)

    assert any("wrong size" in problem for problem in result.problems)
    assert any("targetCurrency" in problem for problem in result.problems)


def test_a_spot_fill_of_the_right_size_passes():
    broker = FakeBroker(position=position(), spot_fills=Decimal("2534"))
    result = execute(broker)

    assert result.ok, result.problems
    assert result.spot_acquired == Decimal("2534")


def test_a_small_fill_difference_is_tolerated():
    """Market orders do not fill to the microlot; 2% is the bar."""
    broker = FakeBroker(position=position(), spot_fills=Decimal("2520"))
    assert execute(broker).ok


def test_the_balance_delta_is_measured_not_the_absolute():
    """An account already holding the base currency must not read as a fill."""
    broker = FakeBroker(position=position(), spot_fills=Decimal("2534"))
    broker.base_held = Decimal("10000")
    result = execute(broker)

    assert result.spot_acquired == Decimal("2534")
    assert result.ok, result.problems
