"""The carry backtest: every entry point, actual funding, actual basis.

`funding_carry` screens on a median rate. This runs the position. The tests
that matter are the ones separating those two claims - that funding accrues
over the hold rather than at the median, that the basis contributes its real
move, and that a distribution drawn from overlapping windows is not mistaken
for that many observations.
"""

import io
import statistics
from contextlib import redirect_stdout

import pytest

from perpkit.analysis.carry_backtest import Backtest, align, report, run

BUCKET_MS = 8 * 3600 * 1000


def funding_rows(rates_bps, start=1_700_000_000_000):
    """Oldest first here; `align` sorts by timestamp anyway."""
    return [{"fundingTime": str(start + i * BUCKET_MS),
             "fundingRate": str(rate / 10_000.0)}
            for i, rate in enumerate(rates_bps)]


def candles(prices, start=1_700_000_000_000):
    return {start + i * BUCKET_MS: price for i, price in enumerate(prices)}


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_align_pairs_funding_with_both_legs():
    ts, funding, gap = align(funding_rows([1.0, 2.0, 3.0]),
                             candles([100.0, 100.0, 100.0]),
                             candles([100.0, 100.0, 100.0]))
    assert len(ts) == 3
    assert funding == pytest.approx([1.0, 2.0, 3.0])
    assert gap == pytest.approx([0.0, 0.0, 0.0])


def test_gap_is_spot_minus_perp_in_bps():
    """Positive gap means spot is above perp, the case measured on BloFin."""
    _, _, gap = align(funding_rows([1.0]), candles([100.0]), candles([99.5]))
    assert gap[0] == pytest.approx(50.0, abs=0.01)


def test_a_period_missing_a_leg_is_dropped():
    """A window cannot be priced without both legs, and guessing one would
    invent the very quantity the backtest exists to measure."""
    spot = candles([100.0, 100.0, 100.0])
    perp = candles([100.0, 100.0, 100.0])
    del perp[max(perp)]

    ts, funding, gap = align(funding_rows([1.0, 2.0, 3.0]), spot, perp)
    assert len(ts) == 2


def test_funding_stamps_snap_to_the_candle_bucket():
    """The two grids are both 8h but need not share an instant."""
    offset = 90_000   # funding printed 90s after the candle opens
    rows = [{"fundingTime": str(1_700_000_000_000 + offset),
             "fundingRate": "0.0001"}]
    ts, funding, gap = align(rows, candles([100.0]), candles([99.9]))

    assert len(ts) == 1
    assert funding[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The position
# ---------------------------------------------------------------------------


def flat_test(rates, spot_prices, perp_prices, *, hold_days, round_trip=0.0):
    return run("X-USDT", funding_rows(rates), candles(spot_prices),
               candles(perp_prices), hold_days=hold_days,
               round_trip_bps=round_trip)


def test_funding_accrues_over_the_hold_not_at_the_median():
    """The whole reason this exists beside the screen.

    Four periods of 1 bps held for one day (3 periods) collects 3 bps, not the
    median rate times anything.
    """
    result = flat_test([1.0] * 5, [100.0] * 5, [100.0] * 5, hold_days=1.0)
    assert result.median_funding_bps == pytest.approx(3.0)
    assert result.median_bps == pytest.approx(3.0)


def test_the_entry_period_funding_is_not_collected():
    """You receive funding for periods you hold THROUGH."""
    # 99 at entry, then 1s. A hold of one period must collect 1, not 99.
    result = flat_test([99.0, 1.0, 1.0, 1.0], [100.0] * 4, [100.0] * 4,
                       hold_days=1 / 3)
    assert result.best_bps == pytest.approx(1.0)


def test_the_basis_move_lands_in_the_result():
    """P&L is gap_exit - gap_entry, so a gap that CLOSES costs money."""
    # Spot flat; perp rises to meet it, so the gap closes from +100bps to 0.
    result = flat_test([0.0] * 4, [100.0] * 4, [99.0, 99.5, 100.0, 100.0],
                       hold_days=2 / 3)

    # Entry gap +100.0 bps, exit gap +50.0 -> -50 for the first window.
    assert result.worst_bps == pytest.approx(-100.0, abs=1.0)
    assert result.median_basis_bps < 0


def test_a_widening_gap_pays():
    result = flat_test([0.0] * 4, [100.0] * 4, [100.0, 100.0, 99.5, 99.0],
                       hold_days=2 / 3)
    assert result.best_bps > 0
    assert result.median_basis_bps > 0


def test_the_round_trip_is_subtracted_once_per_window():
    without = flat_test([1.0] * 5, [100.0] * 5, [100.0] * 5, hold_days=1.0)
    with_cost = flat_test([1.0] * 5, [100.0] * 5, [100.0] * 5, hold_days=1.0,
                          round_trip=10.0)
    assert with_cost.median_bps == pytest.approx(without.median_bps - 10.0)


def test_too_little_history_refuses_rather_than_extrapolating():
    result = flat_test([1.0] * 3, [100.0] * 3, [100.0] * 3, hold_days=30.0)
    assert not result.usable
    assert "aligned periods" in result.unavailable


def test_overlapping_windows_are_not_counted_as_observations():
    """510 windows over 200 days of 30-day holds is about 7 real ones."""
    rates = [2.0] * 600
    result = run("X-USDT", funding_rows(rates), candles([100.0] * 600),
                 candles([100.0] * 600), hold_days=30.0, round_trip_bps=0.0)

    assert result.windows == 510
    assert result.effective_n < 20, "the window count is not the sample size"
    assert result.effective_n >= 1


# ---------------------------------------------------------------------------
# The verdict, which is where a screen would mislead
# ---------------------------------------------------------------------------


def backtest(inst_id, *, median, p5, worst, profitable=1.0):
    return Backtest(inst_id=inst_id, windows=510, effective_n=6,
                    span_days=200.0, median_bps=median, p5_bps=p5,
                    worst_bps=worst, best_bps=median + 50,
                    profitable_share=profitable, median_funding_bps=median + 30,
                    median_basis_bps=0.0, round_trip_bps=30.0)


def run_report(rows, hold_days=30.0):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(rows, hold_days=hold_days, top=10, spot_taker=6.0)
    return buffer.getvalue()


def test_profitable_from_every_entry_is_stated_as_the_strongest_claim():
    text = run_report([backtest("SUI-USDT", median=150, p5=64, worst=48)])
    assert "profitable from EVERY entry" in text


def test_a_median_that_hides_a_bad_tail_is_called_out():
    """DOT-USDT: +92 median, -158 at p5. The screen would have sold you this."""
    text = run_report([
        backtest("SUI-USDT", median=150, p5=64, worst=48),
        backtest("DOT-USDT", median=92, p5=-158, worst=-195, profitable=0.70),
    ])
    assert "Profitable on median but NOT at the 5th percentile: DOT-USDT" in text
    assert "the screen would have sold you" in text


def test_nothing_surviving_the_tail_overrides_the_screen():
    text = run_report([backtest("DOT-USDT", median=92, p5=-158, worst=-195,
                                profitable=0.70)])
    assert "NOTHING IS PROFITABLE AT THE 5TH PERCENTILE" in text
    assert "Trust this one" in text


def test_a_thin_independent_sample_is_flagged():
    text = run_report([backtest("SUI-USDT", median=150, p5=64, worst=48)])
    assert "fewer than 10 independent holds" in text


def test_results_are_ranked_by_the_tail_not_the_median():
    text = run_report([
        backtest("HIGH-MEDIAN", median=300, p5=-50, worst=-90, profitable=0.8),
        backtest("SAFE-USDT", median=100, p5=80, worst=70),
    ])
    assert text.index("SAFE-USDT") < text.index("HIGH-MEDIAN")
