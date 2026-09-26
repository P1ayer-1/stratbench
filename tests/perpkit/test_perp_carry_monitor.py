"""Scoring a carry that is already on, against the plan that predicted it.

The fixtures are not invented. They are the rows the demo account actually
returned for the first live carry - the same position ids, fees, prices and
the `realizedPnl` that made the funding derivation checkable in the first
place. A test built on made-up shapes would have passed against the wrong
field names, which is the failure this whole module exists downstream of.

No network. Every function here takes rows and returns numbers.
"""

from decimal import Decimal

import pytest

from perpkit.strategies.carry import (
    Baseline,
    build_snapshot,
    carry_tag,
    compare,
    fills_totals,
    implied_funding,
    reconstruct_baseline,
)
from perpkit.risk import RiskLimits

# A demo position's payload shape (ids replaced). No funding period had passed.
POSITION = {
    "positionId": "1000000000272",
    "instId": "SUI-USDT",
    "marginMode": "isolated",
    "positions": "-254",
    "averagePrice": "0.785800000000000000",
    "margin": "66.531066666666666666",
    "markPrice": "0.7817012236666667",
    "liquidationPrice": "1.040346870552411213",
    "unrealizedPnl": "1.0410891886666582",
    "maintenanceMargin": "1.2905887202736667217",
    "createTime": "1788984364565",
    "leverage": "3",
    "realizedPnl": "-0.11975592",
}

# Three perp orders: the open that made this position, and the earlier
# attempt that was opened and immediately unwound under a different id.
PERP_ORDERS = [
    {"orderId": "1000000550101", "clientOrderId": "carry0a1b2c3d4e5f6a7bp",
     "positionId": "1000000000272", "side": "sell", "reduceOnly": "false",
     "state": "filled", "filledSize": "254", "averagePrice": "0.7858",
     "fee": "0.11975592"},
    {"orderId": "1000000549717", "clientOrderId": "carry0f0e0d0c0b0au",
     "positionId": "1000000000228", "side": "buy", "reduceOnly": "true",
     "state": "filled", "filledSize": "254", "averagePrice": "0.7861",
     "fee": "0.11980164"},
    {"orderId": "1000000549713", "clientOrderId": "carry1122334455667788p",
     "positionId": "1000000000228", "side": "sell", "reduceOnly": "false",
     "state": "filled", "filledSize": "254", "averagePrice": "0.786",
     "fee": "0.1197864"},
]

# The spot leg. Fee charged in the BASE currency, which is why 254 ordered
# left 253.746 held.
SPOT_ORDERS = [
    {"orderId": "4000000000436", "clientOrderId": "carry0a1b2c3d4e5f6a7bs",
     "instId": "SUI-USDT", "side": "buy", "state": "filled",
     "filledSize": "254.000000000000000000",
     "averagePrice": "0.786300000000000000", "fee": "0.254000000000000000"},
]

FILLS = [
    {"positionId": "1000000000272", "fee": "0.11975592", "fillPnl": "0"},
    {"positionId": "1000000000228", "fee": "0.11980164", "fillPnl": "-0.0254"},
    {"positionId": "1000000000228", "fee": "0.1197864", "fillPnl": "0"},
]

OPENED = 1788984364565
HOUR = 3600 * 1000


def baseline(**overrides) -> Baseline:
    base = reconstruct_baseline(
        POSITION, PERP_ORDERS, SPOT_ORDERS,
        planned_funding_per_day_bps=Decimal("6.54"),
        hold_days=Decimal("30"))
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def rates(*pairs):
    return [{"fundingTime": str(when), "fundingRate": str(rate)}
            for when, rate in pairs]


# --------------------------------------------------------------------------
# joining the two legs
# --------------------------------------------------------------------------


def test_both_legs_of_one_carry_share_a_tag():
    assert (carry_tag("carry0a1b2c3d4e5f6a7bp")
            == carry_tag("carry0a1b2c3d4e5f6a7bs")
            == "0a1b2c3d4e5f6a7b")


def test_an_unwind_order_carries_the_same_tag_as_its_open():
    assert carry_tag("carry0f0e0d0c0b0au") == "0f0e0d0c0b0a"


def test_an_order_this_project_did_not_write_has_no_tag():
    assert carry_tag("web_order_991") is None
    assert carry_tag("carry") is None
    assert carry_tag(None) is None


# --------------------------------------------------------------------------
# rebuilding the baseline from the exchange
# --------------------------------------------------------------------------


def test_the_baseline_comes_back_off_the_exchange():
    base = baseline()
    assert base.position_id == "1000000000272"
    assert base.perp_contracts == Decimal("-254")
    assert base.perp_entry == Decimal("0.7858")
    assert base.opened_at_ms == OPENED
    assert base.tag == "0a1b2c3d4e5f6a7b"


def test_the_spot_leg_is_found_by_the_tag_not_by_the_size():
    base = baseline()
    assert base.spot_ordered == Decimal("254")
    assert base.spot_fee_base == Decimal("0.254")
    # What survived the fee is the hedge, and it is short of the perp leg.
    assert base.spot_base == Decimal("253.746")
    assert base.spot_cost_usd == Decimal("254") * Decimal("0.7863")


def test_a_spot_order_from_another_carry_is_not_adopted():
    stranger = dict(SPOT_ORDERS[0], clientOrderId="carrydeadbeefdeadbeefs")
    base = reconstruct_baseline(
        POSITION, PERP_ORDERS, [stranger],
        planned_funding_per_day_bps=Decimal("6.54"))
    assert base.spot_ordered == Decimal("0")


def test_the_earlier_unwound_attempt_does_not_pay_this_positions_fees():
    base = baseline()
    # Only the fill that made THIS position, not the two from the aborted one.
    assert base.perp_fee_usd == Decimal("0.11975592")


def test_a_position_with_no_size_cannot_be_a_baseline():
    assert reconstruct_baseline(dict(POSITION, positions="0"), [], [],
                                planned_funding_per_day_bps=Decimal("1")) is None


def test_the_round_trip_uses_the_fee_actually_charged():
    base = baseline()
    # perp 0.11975592 + spot 0.254 SUI * 0.7863, doubled, over notional.
    assert base.notional_usd == Decimal("254") * Decimal("0.7858")
    assert base.round_trip_bps == pytest.approx(Decimal("32.0"), abs=0.2)
    # The plan priced 25.82 bps off a VIP tier the account was not on.
    assert base.round_trip_bps > Decimal("25.82")


def test_break_even_moves_with_the_real_fee():
    base = baseline()
    assert base.planned_breakeven_days == pytest.approx(Decimal("4.9"), abs=0.1)


def test_a_prediction_of_zero_has_no_break_even():
    assert baseline(planned_funding_per_day_bps=Decimal("0")
                    ).planned_breakeven_days is None


# --------------------------------------------------------------------------
# funding, derived
# --------------------------------------------------------------------------


def test_fills_are_scoped_to_the_position_that_paid_them():
    fees, pnl = fills_totals(FILLS, "1000000000272")
    assert fees == Decimal("0.11975592")
    assert pnl == Decimal("0")


def test_the_funding_identity_returns_zero_when_no_funding_has_been_paid():
    """The case that made the derivation checkable: realizedPnl was exactly
    the entry fee, so funding must come out at exactly zero."""
    snapshot = build_snapshot(
        at_ms=OPENED + HOUR, inst_id="SUI-USDT", position=POSITION,
        spot_base=Decimal("253.746"), spot_mark=Decimal("0.7817"),
        fills=FILLS, funding_rates=[], baseline=baseline())
    assert snapshot.funding_booked_usd == Decimal("0")


def test_funding_credited_shows_up_as_the_gap_from_the_fees():
    paid = dict(POSITION, realizedPnl="0.13024408")   # -0.11975592 + 0.25
    snapshot = build_snapshot(
        at_ms=OPENED + 9 * HOUR, inst_id="SUI-USDT", position=paid,
        spot_base=Decimal("253.746"), spot_mark=Decimal("0.7817"),
        fills=FILLS, funding_rates=[], baseline=baseline())
    assert snapshot.funding_booked_usd == Decimal("0.25")


def test_a_maker_rebate_cancels_in_the_same_direction():
    """A negative fee is a credit, and `abs()` would have broken this."""
    rebate = [{"positionId": "1000000000272", "fee": "-0.05", "fillPnl": "0"}]
    snapshot = build_snapshot(
        at_ms=OPENED + HOUR, inst_id="SUI-USDT",
        position=dict(POSITION, realizedPnl="0.05"),
        spot_base=Decimal("253.746"), spot_mark=Decimal("0.7817"),
        fills=rebate, funding_rates=[], baseline=baseline())
    assert snapshot.funding_booked_usd == Decimal("0")


def test_a_closed_trades_pnl_is_not_counted_as_funding():
    partial = [{"positionId": "1000000000272", "fee": "0.12", "fillPnl": "4.00"}]
    snapshot = build_snapshot(
        at_ms=OPENED + HOUR, inst_id="SUI-USDT",
        position=dict(POSITION, realizedPnl="3.88"),
        spot_base=Decimal("253.746"), spot_mark=Decimal("0.7817"),
        fills=partial, funding_rates=[], baseline=baseline())
    assert snapshot.funding_booked_usd == Decimal("0")


# --------------------------------------------------------------------------
# funding, implied - the control
# --------------------------------------------------------------------------


def test_a_short_collects_when_the_rate_is_positive():
    total, periods = implied_funding(
        rates((OPENED + HOUR, "0.0001")), contracts=Decimal("-254"),
        notional_usd=Decimal("200"), since_ms=OPENED, until_ms=OPENED + 9 * HOUR)
    assert periods == 1
    assert total == Decimal("0.02")


def test_a_long_pays_what_the_short_collects():
    total, _ = implied_funding(
        rates((OPENED + HOUR, "0.0001")), contracts=Decimal("254"),
        notional_usd=Decimal("200"), since_ms=OPENED, until_ms=OPENED + 9 * HOUR)
    assert total == Decimal("-0.02")


def test_settlements_before_the_open_belong_to_someone_else():
    total, periods = implied_funding(
        rates((OPENED - HOUR, "0.001"), (OPENED, "0.001")),
        contracts=Decimal("-254"), notional_usd=Decimal("200"),
        since_ms=OPENED, until_ms=OPENED + 9 * HOUR)
    assert (total, periods) == (Decimal("0"), 0)


def test_settlements_after_the_reading_are_not_counted_early():
    total, periods = implied_funding(
        rates((OPENED + 20 * HOUR, "0.001")), contracts=Decimal("-254"),
        notional_usd=Decimal("200"), since_ms=OPENED, until_ms=OPENED + 9 * HOUR)
    assert (total, periods) == (Decimal("0"), 0)


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


def snapshot_at(hours, *, position=POSITION, spot=Decimal("253.746"),
                mark=Decimal("0.7817"), fills=FILLS, funding=()):
    return build_snapshot(
        at_ms=OPENED + int(hours * HOUR), inst_id="SUI-USDT",
        position=position, spot_base=spot, spot_mark=mark, fills=fills,
        funding_rates=list(funding), baseline=baseline())


def test_before_the_first_settlement_nothing_is_scored():
    result = compare(baseline(), snapshot_at(1))
    codes = [alert.code for alert in result.alerts]
    assert "no-funding-yet" in codes
    # And crucially not a shortfall warning, which would fire on every carry
    # in its first hours purely because zero is less than the forecast.
    assert "funding-short" not in codes
    assert result.realised_breakeven_days is None


def test_the_delta_the_fee_left_behind_is_reported():
    result = compare(baseline(), snapshot_at(1))
    assert result.net_delta_base == Decimal("-0.254")
    assert result.net_delta_usd == pytest.approx(Decimal("-0.199"), abs=0.01)
    assert "delta" in [alert.code for alert in result.alerts]


def test_a_hedge_that_survives_its_fee_raises_nothing():
    result = compare(baseline(), snapshot_at(1, spot=Decimal("254")))
    assert "delta" not in [alert.code for alert in result.alerts]


def test_funding_below_forecast_moves_break_even_and_says_so():
    paid = dict(POSITION, realizedPnl="-0.11975592")
    # One settlement, and it paid almost nothing.
    result = compare(
        baseline(),
        snapshot_at(9, position=dict(paid, realizedPnl="-0.10975592"),
                    funding=rates((OPENED + 2 * HOUR, "0.000005"))))
    codes = [alert.code for alert in result.alerts]
    assert "funding-short" in codes
    assert result.realised_breakeven_days > result.planned_breakeven_days


def test_funding_that_costs_money_is_not_a_carry():
    result = compare(
        baseline(),
        snapshot_at(9, position=dict(POSITION, realizedPnl="-0.20975592"),
                    funding=rates((OPENED + 2 * HOUR, "-0.00004"))))
    assert "funding-negative" in [alert.code for alert in result.alerts]
    assert result.funding_booked_usd < 0


def test_break_even_past_the_hold_is_worth_saying_out_loud():
    result = compare(
        baseline(hold_days=Decimal("2")),
        snapshot_at(9, position=dict(POSITION, realizedPnl="-0.11875592"),
                    funding=rates((OPENED + 2 * HOUR, "0.000005"))))
    assert "breakeven-past-hold" in [alert.code for alert in result.alerts]


def test_the_two_funding_measures_are_checked_against_each_other():
    # Booked says +$1.00; the published rates say roughly nothing.
    result = compare(
        baseline(),
        snapshot_at(9, position=dict(POSITION, realizedPnl="0.88024408"),
                    funding=rates((OPENED + 2 * HOUR, "0.000005"))))
    assert "funding-divergence" in [alert.code for alert in result.alerts]


def test_measures_that_agree_raise_nothing():
    # 0.0001 * 199.59 = 0.019959 collected; booked matches.
    booked = Decimal("-0.11975592") + Decimal("0.019959")
    result = compare(
        baseline(),
        snapshot_at(9, position=dict(POSITION, realizedPnl=str(booked)),
                    funding=rates((OPENED + 2 * HOUR, "0.0001"))))
    assert "funding-divergence" not in [alert.code for alert in result.alerts]


# --------------------------------------------------------------------------
# the things that end a hold early
# --------------------------------------------------------------------------


def test_a_liquidation_inside_the_open_position_floor_is_critical():
    close = dict(POSITION, liquidationPrice="0.80", markPrice="0.7817")
    result = compare(baseline(), snapshot_at(1, position=close))
    assert result.critical
    assert "liquidation" in [alert.code for alert in result.alerts]


def test_the_liquidation_this_position_actually_has_is_not_critical():
    result = compare(baseline(), snapshot_at(1))
    assert not result.critical
    assert result.liquidation_distance == pytest.approx(Decimal("0.331"),
                                                        abs=0.005)


def test_a_perp_leg_that_vanished_leaves_an_unhedged_long():
    result = compare(baseline(), snapshot_at(1, position=None))
    assert result.critical
    assert "naked-spot" in [alert.code for alert in result.alerts]


def test_a_spot_leg_that_vanished_leaves_a_naked_short():
    result = compare(baseline(), snapshot_at(1, spot=Decimal("0")))
    assert result.critical
    assert "naked-perp" in [alert.code for alert in result.alerts]


def test_a_carry_closed_cleanly_is_reported_closed_not_naked():
    result = compare(baseline(),
                     snapshot_at(1, position=None, spot=Decimal("0")))
    assert not result.critical
    assert "closed" in [alert.code for alert in result.alerts]


def test_a_custom_liquidation_floor_is_honoured():
    close = dict(POSITION, liquidationPrice="0.95", markPrice="0.7817")
    limits = RiskLimits(min_open_liquidation_buffer_pct=Decimal("0.30"))
    result = compare(baseline(), snapshot_at(1, position=close), limits=limits)
    assert result.critical


# --------------------------------------------------------------------------
# mark to market
# --------------------------------------------------------------------------


def test_each_leg_pays_its_own_fee_exactly_once():
    result = compare(baseline(), snapshot_at(1))
    # perp: unrealised 1.0411 plus realised -0.1198 (its entry fee)
    assert result.perp_pnl_usd == pytest.approx(Decimal("0.921"), abs=0.001)
    # spot: 253.746 held at 0.7817, against 254 * 0.7863 paid. The gap
    # includes the 0.254 SUI the fee took.
    assert result.spot_pnl_usd == pytest.approx(Decimal("-1.367"), abs=0.001)
    assert result.total_pnl_usd == pytest.approx(
        result.perp_pnl_usd + result.spot_pnl_usd)


def test_the_projection_runs_on_the_realised_rate_not_the_forecast():
    result = compare(
        baseline(),
        snapshot_at(24, position=dict(POSITION, realizedPnl="-0.09975592"),
                    funding=rates((OPENED + 2 * HOUR, "0.0001"),
                                  (OPENED + 10 * HOUR, "0.0001"),
                                  (OPENED + 18 * HOUR, "0.0001"))))
    # 0.02 USDT a day on 199.59 notional is ~1.00 bps/day, not the 6.54
    # the plan predicted, and the projection has to say so.
    assert result.realised_funding_per_day_bps == pytest.approx(
        Decimal("1.00"), abs=0.05)
    assert result.projected_net_bps < Decimal("170.4")
