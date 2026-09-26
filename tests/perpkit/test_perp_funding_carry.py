"""The funding-carry arithmetic, and the asymmetries that constrain it.

No network: the two API shapes are dicts, so a stub is enough. What matters
here is the arithmetic and the direction of every sign, because a sign error
in a carry calculation turns a cost into a return and looks plausible either
way.
"""

import io
from contextlib import redirect_stdout

import pytest

from perpkit.analysis.blofin_spot import top_of_book
from perpkit.analysis.funding_carry import (
    PERIODS_PER_DAY,
    Carry,
    build,
    funding_stats,
    report,
)

CROSS = {"spot_fee": 5.0, "perp_fee": 5.0, "cross": True}
QUOTE = {"spot_fee": 0.6, "perp_fee": 0.6, "cross": False}


def history(*rates_bps):
    """Newest first, the way the endpoint returns it."""
    return [{"fundingRate": str(rate / 10_000.0), "fundingTime": str(i)}
            for i, rate in enumerate(reversed(rates_bps))]


def ticker(bid, ask, volume="1000000"):
    return {"bidPrice": str(bid), "askPrice": str(ask),
            "volCurrency24h": volume}


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def test_a_crossed_or_one_sided_spot_quote_is_rejected():
    """Returning a negative spread would flatter the cost of trading."""
    assert top_of_book(ticker(1.001, 1.000)) is None      # crossed
    assert top_of_book(ticker(1.000, 1.000)) is None      # locked
    assert top_of_book(ticker(0, 1.0)) is None            # no bid
    assert top_of_book(None) is None

    book = top_of_book(ticker(0.9995, 1.0005))
    assert book is not None
    assert book[3] == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Funding statistics
# ---------------------------------------------------------------------------


def test_funding_stats_summarise_the_distribution_not_just_the_middle():
    stats = funding_stats(history(2.0, 2.0, -1.0, 2.0, 2.0))

    assert stats["median"] == pytest.approx(2.0)
    assert stats["positive_share"] == pytest.approx(0.8)
    assert stats["worst"] == pytest.approx(-1.0)
    assert stats["periods"] == 5


def test_the_drawdown_is_over_cumulative_funding():
    """A carry position experiences the running total, not single periods.

    Three good periods then two bad ones is a 3 bps give-back from the peak,
    and that is the number a position actually lives through.
    """
    stats = funding_stats(history(1.0, 1.0, 1.0, -2.0, -1.0))
    assert stats["drawdown"] == pytest.approx(-3.0)


def test_a_monotonically_rising_curve_has_no_drawdown():
    stats = funding_stats(history(1.0, 1.0, 1.0))
    assert stats["drawdown"] == pytest.approx(0.0)


def test_history_order_does_not_change_the_drawdown_sign():
    """The endpoint returns newest first and carry accrues oldest to newest.

    Reading it in the wrong direction turns a recovery into a drawdown.
    """
    losses_last = funding_stats(history(-2.0, -2.0, 5.0))   # newest first
    assert losses_last["drawdown"] < 0


def test_empty_history_measures_nothing():
    assert funding_stats([]) == {}
    assert funding_stats([{"fundingRate": ""}]) == {}


# ---------------------------------------------------------------------------
# The carry arithmetic
# ---------------------------------------------------------------------------


def carry(funding_bps=2.0, spot_spread=4.0, perp_spread=2.0, basis=-5.0,
          positive_share=1.0):
    return Carry(inst_id="X-USDT", funding_median_bps=funding_bps,
                 funding_mean_bps=funding_bps, positive_share=positive_share,
                 worst_period_bps=0.0, max_drawdown_bps=0.0, periods=100,
                 spot_spread_bps=spot_spread, perp_spread_bps=perp_spread,
                 basis_bps=basis)


def test_funding_accrues_three_times_a_day():
    assert PERIODS_PER_DAY == 3
    assert carry(funding_bps=2.0).funding_per_day_bps == pytest.approx(6.0)


def test_crossing_pays_both_spreads_and_four_fees():
    row = carry(spot_spread=4.0, perp_spread=2.0)
    # 4 + 2 spread, plus 2 x (5 + 5) of fees
    assert row.round_trip_bps(**CROSS) == pytest.approx(26.0)


def test_quoting_pays_no_spread():
    row = carry(spot_spread=40.0, perp_spread=20.0)
    assert row.round_trip_bps(**QUOTE) == pytest.approx(2.4)


def test_breakeven_is_the_round_trip_divided_by_daily_funding():
    row = carry(funding_bps=2.0, spot_spread=4.0, perp_spread=2.0)
    assert row.breakeven_days(**CROSS) == pytest.approx(26.0 / 6.0)


def test_negative_funding_never_breaks_even():
    """You cannot short the spot leg, so negative funding is not carryable."""
    row = carry(funding_bps=-1.0)
    assert row.breakeven_days(**CROSS) == float("inf")
    assert not row.harvestable


def test_convergence_costs_money_when_the_perp_is_below_spot():
    """The measured case on BloFin, and the unfavourable direction.

    The hedge buys spot and shorts the perp, so a gap that closes takes money
    out of the position.
    """
    row = carry(basis=-5.0)     # perp 5 bps below spot
    assert row.convergence_cost_bps == pytest.approx(5.0)
    assert row.net_after_convergence(30, **CROSS) == pytest.approx(
        row.net_bps_over(30, **CROSS) - 5.0)


def test_a_favourable_basis_is_never_counted_as_return():
    """A widening gap would pay. An uncontrolled exposure is not income."""
    row = carry(basis=+5.0)     # perp above spot
    assert row.convergence_cost_bps == pytest.approx(0.0)
    assert row.net_after_convergence(30, **CROSS) == pytest.approx(
        row.net_bps_over(30, **CROSS))


def test_net_grows_with_the_holding_period():
    row = carry(funding_bps=2.0)
    assert row.net_bps_over(60, **CROSS) - row.net_bps_over(30, **CROSS) == (
        pytest.approx(6.0 * 30))


# ---------------------------------------------------------------------------
# Assembly and reporting
# ---------------------------------------------------------------------------


def test_build_requires_both_legs():
    assert build("X-USDT", history(1.0), None, ticker(1, 1.1)).unavailable
    assert build("X-USDT", history(1.0), ticker(1, 1.1), None).unavailable
    assert build("X-USDT", [], ticker(1, 1.1), ticker(1, 1.1)).unavailable


def test_build_signs_the_basis_from_the_perp_side():
    """Negative basis means the perp is cheaper than spot."""
    row = build("X-USDT", history(1.0),
                spot_ticker=ticker(99.9, 100.1),      # spot mid 100
                perp_ticker=ticker(99.4, 99.6))       # perp mid 99.5
    assert row.basis_bps == pytest.approx(-50.0, abs=0.1)
    assert row.convergence_cost_bps == pytest.approx(50.0, abs=0.1)


def run_report(rows, **kwargs):
    options = dict(spot_maker=0.6, spot_taker=5.0, hold_days=30.0, top=10,
                   spot_fees_confirmed=False)
    options.update(kwargs)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(rows, **options)
    return buffer.getvalue()


def test_unconfirmed_spot_fees_are_flagged():
    """They default to the futures schedule, which flatters the result."""
    text = run_report([carry()])
    assert "NOT CONFIRMED" in text

    text = run_report([carry()], spot_fees_confirmed=True)
    assert "NOT CONFIRMED" not in text


def test_an_instrument_paying_longs_is_called_out_as_uncarryable():
    text = run_report([carry(funding_bps=-1.0)])
    assert "pay longs on median" in text


def test_a_wide_spot_spread_sinks_crossing_but_not_quoting():
    """The finding the tool exists to surface: funding is not the constraint.

    And the fallback must stay honest. Quoting sidesteps the spread on paper,
    so the tool falls to that branch - but quoting the spot leg of a hedge
    means sitting delta-exposed until it fills, which is a fill-rate
    assumption rather than a result.
    """
    text = run_report([carry(funding_bps=2.0, spot_spread=400.0)])

    assert "Nothing clears its cost crossing" in text
    assert "fill-rate assumption, not a result" in text
    assert "delta-exposed until it" in text


def test_a_spread_wide_enough_to_beat_quoting_too_clears_nothing():
    """Fees alone can exceed the carry once funding is small enough."""
    text = run_report([carry(funding_bps=0.01, spot_spread=400.0)])

    assert "NOTHING CLEARS ITS OWN COST" in text
    assert "spot spread is what does the damage" in text


def test_an_unreliable_funding_history_is_surfaced_beside_the_profit():
    row = carry(funding_bps=3.0, positive_share=0.6)
    row.max_drawdown_bps = -40.0
    text = run_report([row])
    assert "Read the distribution" in text
    assert "60%" in text
