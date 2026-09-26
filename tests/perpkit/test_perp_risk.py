"""Tests for the liquidation math and the risk gate.

The liquidation expectations here are hand-computed from the formula in the
module docstring, not copied from the implementation's output. That is the
point — a test that just records whatever the code currently returns would
pass just as happily if the formula were wrong.
"""

from decimal import Decimal

import pytest

from perpkit.risk import (
    AccountState,
    RiskEngine,
    RiskLimits,
    Side,
    liquidation_distance_pct,
    liquidation_price,
    max_safe_leverage,
)

D = Decimal


# ---------------------------------------------------------------------------
# Liquidation price
# ---------------------------------------------------------------------------


def test_long_liquidation_matches_hand_calculation():
    # entry 100000, 10x, mmr 0.5%, no fee pad
    #   P = 100000 * (1 - 1/10) / (1 - 0.005)
    #     = 90000 / 0.995
    #     = 90452.2613...
    price = liquidation_price(
        entry_price=D("100000"),
        leverage=D("10"),
        side=Side.LONG,
        maintenance_margin_rate=D("0.005"),
    )
    assert price is not None
    assert price == pytest.approx(Decimal("90452.2613"), abs=Decimal("0.01"))


def test_short_liquidation_matches_hand_calculation():
    #   P = 100000 * (1 + 1/10) / (1 + 0.005)
    #     = 110000 / 1.005
    #     = 109452.7363...
    price = liquidation_price(
        entry_price=D("100000"),
        leverage=D("10"),
        side=Side.SHORT,
        maintenance_margin_rate=D("0.005"),
    )
    assert price is not None
    assert price == pytest.approx(Decimal("109452.7363"), abs=Decimal("0.01"))


def test_long_liquidation_is_below_entry_and_short_above():
    long_price = liquidation_price(
        entry_price=D("50000"), leverage=D("20"), side=Side.LONG
    )
    short_price = liquidation_price(
        entry_price=D("50000"), leverage=D("20"), side=Side.SHORT
    )
    assert long_price < D("50000")
    assert short_price > D("50000")


def test_higher_leverage_moves_liquidation_closer():
    """The core high-leverage risk, stated as a test."""
    entry = D("100000")
    distances = []
    for leverage in (D("2"), D("5"), D("10"), D("25"), D("50")):
        price = liquidation_price(
            entry_price=entry, leverage=leverage, side=Side.LONG
        )
        distances.append(liquidation_distance_pct(mark_price=entry, liq_price=price))
    # Strictly decreasing distance to liquidation as leverage rises.
    assert all(a > b for a, b in zip(distances, distances[1:]))
    # 50x leaves under 2.5% of room.
    assert distances[-1] < D("0.025")


def test_fee_buffer_is_conservative_in_the_right_direction():
    """The pad must make liquidation look *closer*, never further away."""
    clean_long = liquidation_price(
        entry_price=D("100000"), leverage=D("10"), side=Side.LONG
    )
    padded_long = liquidation_price(
        entry_price=D("100000"),
        leverage=D("10"),
        side=Side.LONG,
        fee_buffer_bps=D("20"),
    )
    assert padded_long > clean_long  # higher = closer to entry for a long

    clean_short = liquidation_price(
        entry_price=D("100000"), leverage=D("10"), side=Side.SHORT
    )
    padded_short = liquidation_price(
        entry_price=D("100000"),
        leverage=D("10"),
        side=Side.SHORT,
        fee_buffer_bps=D("20"),
    )
    assert padded_short < clean_short  # lower = closer to entry for a short


def test_extra_margin_pushes_liquidation_away():
    without = liquidation_price(
        entry_price=D("100000"), leverage=D("10"), side=Side.LONG
    )
    with_margin = liquidation_price(
        entry_price=D("100000"),
        leverage=D("10"),
        side=Side.LONG,
        extra_margin=D("1000"),
        quantity_base=D("1"),
    )
    assert with_margin < without  # further below entry = safer


def test_invalid_inputs_return_none_not_garbage():
    assert liquidation_price(entry_price=D("0"), leverage=D("10"), side=Side.LONG) is None
    assert liquidation_price(entry_price=D("100"), leverage=D("0"), side=Side.LONG) is None
    assert (
        liquidation_price(
            entry_price=D("100"), leverage=D("10"), side=Side.LONG,
            maintenance_margin_rate=D("1.5"),
        )
        is None
    )
    # extra_margin without quantity is undefined, not silently ignored
    assert (
        liquidation_price(
            entry_price=D("100"), leverage=D("10"), side=Side.LONG,
            extra_margin=D("50"),
        )
        is None
    )


def test_max_safe_leverage_round_trips():
    """Leverage from a buffer, fed back in, should reproduce the buffer."""
    buffer_pct = D("0.10")
    leverage = max_safe_leverage(buffer_pct=buffer_pct, side=Side.LONG)
    price = liquidation_price(
        entry_price=D("100000"), leverage=leverage, side=Side.LONG
    )
    distance = liquidation_distance_pct(mark_price=D("100000"), liq_price=price)
    assert distance == pytest.approx(buffer_pct, abs=D("0.0001"))


# ---------------------------------------------------------------------------
# The risk gate
# ---------------------------------------------------------------------------


def _clean_order(**overrides):
    order = dict(
        account=AccountState(equity=D("1000")),
        side=Side.LONG,
        size_base=D("0.001"),
        price=D("100000"),
        leverage=D("3"),
        spread_bps=D("1"),
        expected_edge_bps=D("20"),
        book_age_ms=100,
        tape_staleness_s=0.5,
        features_valid=True,
    )
    order.update(overrides)
    return order


def test_clean_order_is_allowed():
    engine = RiskEngine()
    decision = engine.check_order(**_clean_order())
    assert decision.allowed, decision.reasons


def test_kill_switch_blocks_everything():
    engine = RiskEngine()
    engine.trip("test")
    decision = engine.check_order(**_clean_order())
    assert not decision.allowed
    assert any("kill switch" in reason for reason in decision.reasons)


def test_disconnect_trips_kill_switch_and_returns_actions():
    engine = RiskEngine()
    actions = engine.on_disconnect()
    assert actions[0] == "cancel_all_orders"
    assert "stop_opening_positions" in actions
    assert engine.kill_switch_active
    # Reconnecting must NOT silently re-enable trading.
    engine.on_reconnect()
    assert engine.kill_switch_active
    assert not engine.check_order(**_clean_order()).allowed


def test_edge_must_clear_round_trip_cost():
    engine = RiskEngine(RiskLimits(round_trip_cost_bps=D("6")))
    decision = engine.check_order(**_clean_order(expected_edge_bps=D("5")))
    assert not decision.allowed
    assert any("round-trip cost" in reason for reason in decision.reasons)


def test_wide_spread_blocks_order():
    engine = RiskEngine(RiskLimits(max_spread_bps=D("5")))
    decision = engine.check_order(**_clean_order(spread_bps=D("12")))
    assert not decision.allowed
    assert any("spread" in reason for reason in decision.reasons)


def test_stale_book_blocks_order():
    engine = RiskEngine()
    decision = engine.check_order(**_clean_order(book_age_ms=9999))
    assert not decision.allowed
    assert any("book age" in reason for reason in decision.reasons)


def test_invalid_features_block_order():
    engine = RiskEngine()
    decision = engine.check_order(**_clean_order(features_valid=False))
    assert not decision.allowed
    assert any("invalid" in reason for reason in decision.reasons)


def test_liquidation_buffer_blocks_excessive_leverage():
    # 25x leaves ~4% of room; the default guard demands 15%.
    engine = RiskEngine(RiskLimits(max_leverage=D("50")))
    decision = engine.check_order(**_clean_order(leverage=D("25")))
    assert not decision.allowed
    assert any("liquidation" in reason for reason in decision.reasons)
    assert decision.liq_distance_pct is not None
    assert decision.liq_distance_pct < D("0.15")


def test_position_cap_blocks_accumulation():
    engine = RiskEngine(RiskLimits(max_position_base=D("0.002")))
    account = AccountState(equity=D("1000"), position_base=D("0.0019"))
    decision = engine.check_order(**_clean_order(account=account, size_base=D("0.001")))
    assert not decision.allowed
    assert any("resulting position" in reason for reason in decision.reasons)


def test_all_failures_are_reported_not_just_the_first():
    engine = RiskEngine()
    decision = engine.check_order(
        **_clean_order(
            spread_bps=D("50"), expected_edge_bps=D("0"), book_age_ms=9999
        )
    )
    assert not decision.allowed
    assert len(decision.reasons) >= 3


def test_daily_loss_limit_trips_kill_switch():
    engine = RiskEngine(RiskLimits(max_daily_loss=D("50")))
    account = AccountState(equity=D("1000"))
    engine.on_trade_closed(D("-30"), account)
    assert not engine.kill_switch_active
    engine.on_trade_closed(D("-25"), account)
    assert engine.kill_switch_active


def test_consecutive_losses_trip_then_reset_on_a_win():
    engine = RiskEngine(RiskLimits(max_consecutive_losses=3))
    account = AccountState(equity=D("1000"))
    engine.on_trade_closed(D("-1"), account)
    engine.on_trade_closed(D("-1"), account)
    engine.on_trade_closed(D("5"), account)
    assert account.consecutive_losses == 0
    assert not engine.kill_switch_active
    for _ in range(3):
        engine.on_trade_closed(D("-1"), account)
    assert engine.kill_switch_active


def test_blended_entry_only_blends_when_adding_exposure():
    account = AccountState(position_base=D("1"), entry_price=D("100"))
    added = RiskEngine._blended_entry(account, Side.LONG, D("1"), D("200"))
    assert added == D("150")
    # Reducing a long must not move the entry price.
    reduced = RiskEngine._blended_entry(account, Side.SHORT, D("1"), D("200"))
    assert reduced == D("100")


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_no_edge_means_no_position():
    engine = RiskEngine()
    result = engine.size_position(
        account=AccountState(equity=D("1000")),
        price=D("100000"),
        expected_edge_bps=D("2"),
        volatility_bps=D("10"),
        win_probability=D("0.55"),
        leverage=D("3"),
    )
    assert result.size_base == 0
    assert result.binding_constraint == "no edge after costs"


def test_sizing_never_exceeds_hard_caps():
    engine = RiskEngine(RiskLimits(max_order_base=D("0.001")))
    result = engine.size_position(
        account=AccountState(equity=D("1000000")),
        price=D("100000"),
        expected_edge_bps=D("500"),
        volatility_bps=D("5"),
        win_probability=D("0.99"),
        leverage=D("5"),
    )
    assert result.size_base <= D("0.001")


def test_higher_volatility_reduces_size():
    engine = RiskEngine(RiskLimits(max_order_base=D("100"), max_notional=D("10000000")))
    common = dict(
        account=AccountState(equity=D("100000")),
        price=D("100000"),
        expected_edge_bps=D("50"),
        win_probability=D("0.6"),
        leverage=D("3"),
    )
    calm = engine.size_position(volatility_bps=D("10"), **common)
    wild = engine.size_position(volatility_bps=D("100"), **common)
    assert wild.size_base < calm.size_base


def test_kelly_is_fractional_not_full():
    """A 60% edge should not produce anything close to full-Kelly sizing."""
    engine = RiskEngine(RiskLimits(max_order_base=D("1000"), max_notional=D("100000000")))
    result = engine.size_position(
        account=AccountState(equity=D("100000")),
        price=D("100"),
        expected_edge_bps=D("100"),
        volatility_bps=D("100"),
        win_probability=D("0.60"),
        leverage=D("3"),
        kelly_fraction=D("0.15"),
    )
    assert result.detail["kellyFraction"] <= D("0.15")


# ---------------------------------------------------------------------------
# Unrealized drawdown
#
# The gap these cover: every breaker fed by `on_trade_closed` measures
# *realized* PnL, so a strategy that holds a loser indefinitely - "the trend is
# up, we can wait" - never trips one. These assert that an open loss counts.
# ---------------------------------------------------------------------------


def test_unrealized_pnl_signs_are_right_for_both_sides():
    long_account = AccountState(position_base=D("0.05"), entry_price=D("100000"))
    assert long_account.unrealized_pnl(D("99000")) == D("-50")
    assert long_account.unrealized_pnl(D("101000")) == D("50")

    # A short is a negative quantity, so a price rise must be a loss.
    short_account = AccountState(position_base=D("-0.05"), entry_price=D("100000"))
    assert short_account.unrealized_pnl(D("101000")) == D("-50")
    assert short_account.unrealized_pnl(D("99000")) == D("50")


def test_flat_account_has_no_unrealized_pnl():
    assert AccountState().unrealized_pnl(D("100000")) == D("0")


def test_holding_a_loser_trips_the_kill_switch_without_ever_closing_it():
    # The whole point. No call to on_trade_closed anywhere in this test.
    engine = RiskEngine(RiskLimits(max_unrealized_loss=D("50")))
    account = AccountState(
        equity=D("1000"), position_base=D("0.05"), entry_price=D("100000")
    )

    engine.mark_to_market(account, D("99400"))  # -30, inside the limit
    assert not engine.kill_switch_active
    assert account.realized_pnl_today == D("0")

    result = engine.mark_to_market(account, D("98000"))  # -100
    assert engine.kill_switch_active
    assert result["unrealizedPnl"] == D("-100")
    assert any("unrealized loss" in reason for reason in result["breaches"])
    assert result["actions"][0] == "close_position"


def test_daily_loss_limit_counts_realized_and_open_together():
    engine = RiskEngine(
        RiskLimits(max_daily_loss=D("100"), max_unrealized_loss=D("1000000"))
    )
    account = AccountState(
        equity=D("1000"),
        position_base=D("0.05"),
        entry_price=D("100000"),
        realized_pnl_today=D("-60"),
    )
    # -60 booked, -50 open: neither alone breaches, the sum does.
    result = engine.mark_to_market(account, D("99000"))
    assert result["totalPnlToday"] == D("-110")
    assert engine.kill_switch_active


def test_open_position_drifting_toward_liquidation_trips():
    # entry 100000, 5x, mmr 0.5%, 15bps fee pad:
    #   P_liq = 100000 * (1 - 1/5) / (1 - 0.005) * 1.0015 = 80522.6
    # At mark 86000 that is (86000 - 80522.6) / 86000 = 6.37% away, inside the
    # 8% floor. The loss limits are lifted so only the liquidation guard fires.
    engine = RiskEngine(
        RiskLimits(
            min_open_liquidation_buffer_pct=D("0.08"),
            max_unrealized_loss=D("1000000"),
            max_daily_loss=D("1000000"),
        )
    )
    account = AccountState(
        equity=D("1000"),
        position_base=D("0.05"),
        entry_price=D("100000"),
        leverage=D("5"),
    )
    result = engine.mark_to_market(account, D("86000"))
    assert engine.kill_switch_active
    assert len(result["breaches"]) == 1
    assert "liquidation" in result["breaches"][0]


def test_mark_to_market_on_a_flat_account_does_nothing():
    engine = RiskEngine()
    result = engine.mark_to_market(AccountState(equity=D("1000")), D("100000"))
    assert not engine.kill_switch_active
    assert result["breaches"] == []
    assert result["actions"] == []


def test_check_order_refuses_to_add_to_an_open_drawdown():
    engine = RiskEngine()
    account = AccountState(
        equity=D("1000"), position_base=D("0.001"), entry_price=D("200000")
    )
    # Marked at 100000 the open position is down 100, past both the 50
    # unrealized limit and the 100 daily limit, with nothing yet realized.
    decision = engine.check_order(**_clean_order(account=account))
    assert not decision.allowed
    assert any("unrealized loss" in reason for reason in decision.reasons)
    assert any("mark-to-market" in reason for reason in decision.reasons)


def test_status_reports_open_pnl_alongside_realized():
    engine = RiskEngine()
    account = AccountState(
        position_base=D("0.05"),
        entry_price=D("100000"),
        leverage=D("5"),
        realized_pnl_today=D("-20"),
    )
    status = engine.status(account, D("99000"))
    assert status["unrealizedPnl"] == pytest.approx(-50.0)
    assert status["totalPnlToday"] == pytest.approx(-70.0)


# ---------------------------------------------------------------------------
# reduce_only
#
# Without this path the engine deadlocks in the situation it exists for: it
# trips the switch, says "close_position", then vetoes the closing order.
# ---------------------------------------------------------------------------


def _reduce_order(**overrides):
    order = dict(
        account=AccountState(
            equity=D("1000"), position_base=D("0.01"), entry_price=D("100000")
        ),
        side=Side.SHORT,
        size_base=D("0.01"),
        price=D("95000"),
        leverage=D("3"),
        spread_bps=D("1"),
        reduce_only=True,
    )
    order.update(overrides)
    return order


def test_reduce_only_closes_through_a_tripped_kill_switch():
    engine = RiskEngine()
    engine.trip("daily loss limit hit")
    decision = engine.check_order(**_reduce_order())
    assert decision.allowed, decision.reasons


def test_reduce_only_ignores_the_edge_gate_and_a_blown_out_spread():
    engine = RiskEngine()
    decision = engine.check_order(
        **_reduce_order(spread_bps=D("500"), expected_edge_bps=D("0"))
    )
    assert decision.allowed, decision.reasons


def test_reduce_only_still_needs_a_connection():
    engine = RiskEngine()
    engine.on_disconnect()
    decision = engine.check_order(**_reduce_order())
    assert not decision.allowed
    assert any("not connected" in reason for reason in decision.reasons)


def test_reduce_only_rejects_an_order_that_would_add_exposure():
    engine = RiskEngine()
    # Long position, and a LONG order claiming to be reduce_only.
    decision = engine.check_order(**_reduce_order(side=Side.LONG))
    assert not decision.allowed
    assert any("reduce_only" in reason for reason in decision.reasons)


def test_reduce_only_rejects_a_size_that_would_flip_the_position():
    engine = RiskEngine()
    decision = engine.check_order(**_reduce_order(size_base=D("0.02")))
    assert not decision.allowed
    assert any("would flip it" in reason for reason in decision.reasons)


def test_reduce_only_rejects_when_there_is_nothing_to_close():
    engine = RiskEngine()
    decision = engine.check_order(
        **_reduce_order(account=AccountState(equity=D("1000")))
    )
    assert not decision.allowed
    assert any("no open position" in reason for reason in decision.reasons)


def test_reduce_only_partial_close_is_allowed():
    engine = RiskEngine()
    engine.trip("test")
    decision = engine.check_order(**_reduce_order(size_base=D("0.004")))
    assert decision.allowed, decision.reasons
