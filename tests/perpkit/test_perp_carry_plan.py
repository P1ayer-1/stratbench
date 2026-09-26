"""The carry planner: sizing, capital, and every reason to refuse.

Pure arithmetic, no network. The things worth pinning down are the ones a
live executor would inherit silently - that the two legs actually cancel after
the exchange's lot sizes have had their way, that rounding never buys size the
plan did not budget for, and that each risk gate refuses on its own rather
than the first one masking the rest.
"""

from decimal import Decimal

import pytest

from perpkit.strategies.carry import Market, Wallets, plan_carry, round_down
from perpkit.risk import RiskLimits


def market(**overrides) -> Market:
    """A round instrument: $1 base, 1 unit per contract, integer lots."""
    defaults = dict(
        inst_id="X-USDT",
        spot_bid=Decimal("0.999"), spot_ask=Decimal("1.001"),
        perp_bid=Decimal("0.999"), perp_ask=Decimal("1.001"),
        contract_value=Decimal("1"),
        perp_lot_size=Decimal("1"), perp_min_size=Decimal("1"),
        spot_lot_size=Decimal("1"), spot_min_size=Decimal("1"),
    )
    defaults.update(overrides)
    return Market(**defaults)


def plan(**overrides):
    defaults = dict(
        market=market(),
        wallets=Wallets(spot_usdt=Decimal("50000"),
                        futures_usdt=Decimal("20000")),
        target_notional_usd=Decimal("1000"),
        leverage=Decimal("3"),
        funding_per_day_bps=Decimal("6"),
        hold_days=Decimal("30"),
        spot_taker_bps=Decimal("6"),
        perp_taker_bps=Decimal("5"),
    )
    defaults.update(overrides)
    return plan_carry(**defaults)


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------


def test_rounding_is_always_down():
    """Rounding a leg UP buys size the plan did not budget capital for."""
    assert round_down(Decimal("7.9"), Decimal("1")) == Decimal("7")
    assert round_down(Decimal("7.1"), Decimal("1")) == Decimal("7")
    assert round_down(Decimal("0.9"), Decimal("1")) == Decimal("0")
    assert round_down(Decimal("2.55"), Decimal("0.1")) == Decimal("2.5")


def test_a_zero_step_is_a_no_op():
    assert round_down(Decimal("7.9"), Decimal("0")) == Decimal("7.9")


# ---------------------------------------------------------------------------
# Delta neutrality, which is the whole trade
# ---------------------------------------------------------------------------


def test_a_coarse_lot_size_cannot_absorb_the_fee():
    """A 1-unit spot lot cannot buy the 1000.6 that would net to 1000.

    So the residual is the fee itself - unavoidable at this granularity, and
    reported rather than hidden. A finer lot size removes it; see
    `test_the_spot_leg_is_grossed_up_for_a_fee_paid_in_base`.
    """
    result = plan()
    assert result.spot_base == result.perp_base, "rounded back down to the lot"
    assert result.residual_base == pytest.approx(
        result.perp_base * result.spot_fee_rate)


def test_a_fine_lot_size_leaves_the_legs_matched():
    result = plan(market=market(spot_lot_size=Decimal("0.000001")))
    assert result.spot_after_fee == pytest.approx(result.perp_base,
                                                  rel=Decimal("0.00001"))


def test_the_perp_leg_is_sized_first_and_spot_follows():
    """The perp contract is the coarser unit, so it sets the granularity.

    Sizing spot first and deriving contracts would leave a mismatch of up to
    a whole contract instead of a whole spot lot.
    """
    result = plan(market=market(contract_value=Decimal("10"),
                                spot_lot_size=Decimal("1")))
    assert result.perp_base == result.perp_contracts * Decimal("10")
    assert result.spot_base == result.perp_base


def test_a_rounding_mismatch_is_measured_not_hidden():
    """Whatever the lot sizes cannot reconcile is naked directional risk."""
    result = plan(
        market=market(contract_value=Decimal("10"), spot_lot_size=Decimal("3")),
        target_notional_usd=Decimal("1000"),
    )
    assert result.residual_base != 0
    assert result.residual_usd > 0
    assert result.spot_base < result.perp_base, "spot rounds down, so it lags"


def test_a_material_mismatch_warns():
    result = plan(
        market=market(contract_value=Decimal("100"),
                      spot_lot_size=Decimal("70")),
        target_notional_usd=Decimal("1000"),
    )
    assert any("directional" in w for w in result.warnings)


def test_a_negligible_mismatch_does_not_warn():
    result = plan()
    assert not any("directional" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------


def test_capital_is_the_spot_cost_plus_the_perp_margin():
    result = plan(leverage=Decimal("4"))
    assert result.perp_margin_usd == pytest.approx(
        result.notional_usd / Decimal("4"))
    assert result.total_capital_usd == pytest.approx(
        result.spot_cost_usd + result.perp_margin_usd)


def test_a_transfer_is_reported_when_a_wallet_is_short():
    """Carry needs BOTH wallets funded; USDT in the wrong one is not usable."""
    result = plan(wallets=Wallets(spot_usdt=Decimal("100"),
                                  futures_usdt=Decimal("20000")))
    assert result.spot_transfer_usd > 0
    assert result.futures_transfer_usd == 0


def test_not_enough_total_usdt_is_refused():
    result = plan(wallets=Wallets(spot_usdt=Decimal("10"),
                                  futures_usdt=Decimal("10")))
    assert not result.ok
    assert any("not enough USDT" in reason for reason in result.reasons)


def test_enough_usdt_in_the_wrong_wallet_is_a_transfer_not_a_refusal():
    result = plan(wallets=Wallets(spot_usdt=Decimal("0"),
                                  futures_usdt=Decimal("50000")))
    assert result.ok, result.reasons
    assert result.spot_transfer_usd > 0


# ---------------------------------------------------------------------------
# The risk gates
# ---------------------------------------------------------------------------


def test_leverage_that_puts_liquidation_inside_the_buffer_is_refused():
    result = plan(leverage=Decimal("10"))
    assert not result.ok
    assert any("liquidation only" in reason for reason in result.reasons)


def test_every_failing_gate_is_reported_not_just_the_first():
    """At 3am you want the whole list, not whichever check ran first."""
    result = plan(leverage=Decimal("10"), target_notional_usd=Decimal("50000"),
                  wallets=Wallets(spot_usdt=Decimal("1"),
                                  futures_usdt=Decimal("1")))
    assert len(result.reasons) >= 3


def test_the_liquidation_estimate_ignores_the_wallet_balance():
    """Isolated margin, deliberately.

    Cross would back the short leg with the spot leg's cash too, which reads
    as a wider buffer and is really a larger blast radius.
    """
    rich = plan(wallets=Wallets(spot_usdt=Decimal("1000000"),
                                futures_usdt=Decimal("1000000")))
    lean = plan(wallets=Wallets(spot_usdt=Decimal("50000"),
                                futures_usdt=Decimal("20000")))
    assert rich.liquidation_price == lean.liquidation_price


def test_a_short_liquidates_above_the_entry():
    result = plan()
    assert result.liquidation_price > result.perp_bid


def test_the_notional_cap_is_enforced():
    result = plan(target_notional_usd=Decimal("9000"))
    assert not result.ok
    assert any("exceeds the" in reason and "limit" in reason
               for reason in result.reasons)


def test_a_looser_limit_set_admits_a_bigger_plan():
    limits = RiskLimits(max_notional=Decimal("50000"), max_leverage=Decimal("5"))
    result = plan(target_notional_usd=Decimal("9000"), limits=limits)
    assert result.ok, result.reasons


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------


def test_negative_funding_is_refused_because_spot_cannot_be_shorted():
    result = plan(funding_per_day_bps=Decimal("-1"))
    assert not result.ok
    assert any("nothing to collect" in reason for reason in result.reasons)


def test_a_hold_that_does_not_repay_the_round_trip_is_refused():
    result = plan(funding_per_day_bps=Decimal("0.1"),
                  hold_days=Decimal("5"))
    assert not result.ok
    assert any("not repaid" in reason for reason in result.reasons)


def test_break_even_beyond_the_hold_warns_when_the_total_still_clears():
    """A warning, not a refusal: it is a fact about the plan, not a veto."""
    result = plan(funding_per_day_bps=Decimal("6"), hold_days=Decimal("30"))
    assert result.ok
    assert result.breakeven_days < Decimal("30")

    slow = plan(funding_per_day_bps=Decimal("1.2"), hold_days=Decimal("40"))
    assert any("break-even" in w for w in slow.warnings)


def test_the_round_trip_counts_four_legs_and_both_spreads():
    result = plan(market=market(spot_bid=Decimal("0.99"),
                                spot_ask=Decimal("1.01")))
    # 200 bps of spot spread + ~20 of perp + 2 * (6 + 5) of fees
    assert result.round_trip_bps > Decimal("200")


def test_expected_usd_scales_with_notional():
    small = plan(target_notional_usd=Decimal("1000"))
    large = plan(target_notional_usd=Decimal("4000"),
                 limits=RiskLimits(max_notional=Decimal("50000")))
    assert large.expected_net_usd == pytest.approx(
        small.expected_net_usd * 4, rel=Decimal("0.01"))


# ---------------------------------------------------------------------------
# Sizing floors
# ---------------------------------------------------------------------------


def test_a_notional_below_the_contract_minimum_is_refused():
    result = plan(target_notional_usd=Decimal("0.10"),
                  market=market(perp_min_size=Decimal("10")))
    assert not result.ok
    assert any("minimum" in reason for reason in result.reasons)


def test_an_unusable_quote_is_refused():
    assert not plan(market=market(spot_ask=Decimal("0"))).ok
    assert not plan(market=market(perp_bid=Decimal("0"))).ok


def test_nonsense_inputs_are_refused():
    assert not plan(target_notional_usd=Decimal("0")).ok
    assert not plan(leverage=Decimal("0")).ok


# ---------------------------------------------------------------------------
# The spot fee is charged in the BASE currency
# ---------------------------------------------------------------------------
#
# Measured on a real fill: buying 254 SUI at 0.1% delivered 253.746, because
# the fee comes out of what you receive rather than out of the USDT. Buying
# exactly the perp's base therefore leaves the hedge short by the fee, every
# single time, in the same direction.


def test_the_spot_leg_is_grossed_up_for_a_fee_paid_in_base():
    result = plan(spot_taker_bps=Decimal("10"),
                  market=market(spot_lot_size=Decimal("0.001")))

    assert result.spot_base > result.perp_base, "must buy more than the hedge"
    assert result.spot_after_fee == pytest.approx(result.perp_base,
                                                  rel=Decimal("0.0001"))


def test_grossing_up_shrinks_the_residual_by_orders_of_magnitude():
    """Without it the hedge is short by the whole fee."""
    result = plan(spot_taker_bps=Decimal("10"),
                  market=market(spot_lot_size=Decimal("0.001")))
    naive_residual = result.perp_base * Decimal("0.001")

    assert abs(result.residual_base) < naive_residual / 100


def test_a_zero_fee_needs_no_gross_up():
    result = plan(spot_taker_bps=Decimal("0"),
                  market=market(spot_lot_size=Decimal("0.001")))
    assert result.spot_base == result.perp_base


def test_the_residual_is_measured_after_the_fee_not_before():
    """What matters is what survives to hedge with."""
    result = plan(spot_taker_bps=Decimal("10"),
                  market=market(spot_lot_size=Decimal("0.001")))
    assert result.residual_base == pytest.approx(
        result.perp_base - result.spot_after_fee)
