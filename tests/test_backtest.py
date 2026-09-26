"""Labels never see the future; holds never overlap; the control is the mean."""

from decimal import Decimal

import pytest

from predkit.backtest import Decision, WindowRow, label_row, read_rows, run, write_rows
from predkit.schema import Contract, Outcome, utc

D = Decimal


def contract(minute=5):
    return Contract("kalshi", f"KX-{minute}", "q", utc(2026, 9, 12, 14, minute), "kalshi:KX", "default")


def test_a_row_cannot_be_written_before_resolution():
    c = contract()
    with pytest.raises(ValueError):
        label_row(c, resolved_yes=True, resolution_source="kalshi:KX", now_ms=c.resolves_at_ms - 1,
                  opens_at_ms=0, entry_price="0.5", entry_bid="0.48")


def test_a_spot_exchange_is_not_a_resolution_source():
    c = Contract("polymarket", "0xc", "q", utc(2026, 9, 12, 14, 5), "binance:BTCUSDT", "economics_fees")
    with pytest.raises(ValueError):
        label_row(c, resolved_yes=True, resolution_source="binance:BTCUSDT", now_ms=c.resolves_at_ms + 1,
                  opens_at_ms=0, entry_price="0.5", entry_bid="0.48")


def test_the_label_source_must_be_the_contracts_own():
    c = contract()
    with pytest.raises(ValueError):
        label_row(c, resolved_yes=True, resolution_source="my-spreadsheet", now_ms=c.resolves_at_ms + 1,
                  opens_at_ms=0, entry_price="0.5", entry_bid="0.48")


def rows():
    out = []
    for i, (minute, yes) in enumerate([(5, True), (10, False), (15, True), (20, True)]):
        c = contract(minute)
        out.append(label_row(c, resolved_yes=yes, resolution_source="kalshi:KX", now_ms=c.resolves_at_ms + 1,
                             opens_at_ms=c.resolves_at_ms - 240_000, entry_price="0.50", entry_bid="0.48",
                             signal="0.60"))
    return out


def always_yes(row):
    return Decision(Outcome.YES, D("0.50"), D("10"), role="taker")


def test_pnl_is_hand_computed_with_the_kalshi_fee():
    # 4 trades of 10 @ 0.50: cost 5.00 each. Wins pay 10.00: 3 wins.
    # gross = 3*(10-5) + 1*(0-5) = 10.00
    # fee per order = ceil_cent(0.07*0.5*0.5*10 = 0.175) = 0.18; x4 = 0.72
    result = run(rows(), always_yes, shuffles=0)
    assert (result.trades, result.wins) == (4, 3)
    assert result.gross == D("10.00")
    assert result.fees == D("0.72")
    assert result.net == D("9.28")


def test_overlapping_holds_on_one_market_are_skipped():
    a, b = rows()[:2]
    same_market = WindowRow(**{**b.__dict__, "market_id": a.market_id, "opens_at_ms": a.opens_at_ms + 1})
    result = run([a, same_market], always_yes, shuffles=0)
    assert result.trades == 1 and result.skipped_overlap == 1


def test_the_control_mean_lands_near_minus_the_cost():
    """Three of four labels are wins, so shuffling labels keeps the same
    3:1 mix: every shuffle nets exactly 9.28 here. Skewed decisions vs
    labels is what a control is for; a strategy that buys where labels are
    balanced sees a control near -fees."""
    result = run(rows(), always_yes, shuffles=50)
    assert result.control_mean == pytest.approx(9.28)
    assert result.control_percentile == 0.0          # real == every shuffle


def test_rows_round_trip_through_jsonl(tmp_path):
    original = rows()
    write_rows(tmp_path / "rows.jsonl", original)
    assert read_rows(tmp_path / "rows.jsonl") == original
