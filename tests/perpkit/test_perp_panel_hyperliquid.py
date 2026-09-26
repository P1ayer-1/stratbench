"""The Hyperliquid panel: pre-listing candles, and the hourly cadence.

The pre-listing candle is the bug this file exists for. `candleSnapshot`
returns rows for days BEFORE a coin listed, carrying an OHLC from somewhere
with volume and trade count both zero. Measured on a 60-coin panel: ZEC and XMR
had 999 such rows each and 13.3% of the panel was one. They are not thin days - they
are days this venue did not trade the coin - so a return across one is a price
move that could not have been captured, and a liquidity filter reading their
volume as zero behaves erratically instead of excluding them.
"""

import math
from datetime import datetime, timezone

import pytest

from perpkit.analysis.panel_hyperliquid import coin_rows, funding_by_day


def at(date_string, hour=0, millisecond=0):
    base = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(base.timestamp() * 1000) + hour * 3_600_000 + millisecond


def candle(date, *, close=100.0, volume=1000.0, trades=50.0, high=101.0,
           low=99.0):
    return (at(date), 100.0, high, low, close, volume, trades)


def full_day(date):
    """24 hourly settlements, which is what a complete day looks like here."""
    return {date: (24.0, 24)}


# ---------------------------------------------------------------------------
# Pre-listing candles
# ---------------------------------------------------------------------------


def test_a_pre_listing_candle_is_marked_untraded():
    """Zero volume AND zero trades means the venue did not trade this coin.

    `minutes = 0` routes it through the same guard the shared schema in
    perpkit/analysis/panel.py uses for an exchange outage: excluded from the universe, and the history run-length
    counter resets so a coin cannot appear to have years of history it did not
    have. Reporting 1440 minutes would put a fabricated day in the book.
    """
    rows = coin_rows("ZEC", [candle("2026-03-04", volume=0.0, trades=0.0)],
                     full_day("2026-03-04"))
    assert rows[0]["minutes"] == 0


def test_a_thin_but_real_day_is_kept():
    """One trade is not no trades. The guard must key on whether the venue
    traded the coin, not on whether it traded much of it."""
    rows = coin_rows("X", [candle("2026-03-04", volume=0.0, trades=1.0)],
                     full_day("2026-03-04"))
    assert rows[0]["minutes"] == 1440
    rows = coin_rows("X", [candle("2026-03-04", volume=0.01, trades=0.0)],
                     full_day("2026-03-04"))
    assert rows[0]["minutes"] == 1440


def test_the_row_still_exists_so_the_gap_stays_visible():
    """Dropping the row would hide the gap, and the next return would silently
    span more than a day - the same reason the shared schema in
    perpkit/analysis/panel.py keeps short days."""
    rows = coin_rows("ZEC", [candle("2026-03-04", volume=0.0, trades=0.0),
                             candle("2026-03-05")], full_day("2026-03-05"))
    assert [row["date"] for row in rows] == ["2026-03-04", "2026-03-05"]


# ---------------------------------------------------------------------------
# Hourly funding
# ---------------------------------------------------------------------------


def test_a_day_needs_most_of_its_24_settlements_to_count():
    """A partial day looks like cheap funding and is really missing data, and
    on an hourly venue the difference is 24 settlements wide."""
    partial = {"2026-03-04": (2.0, 3)}
    rows = coin_rows("X", [candle("2026-03-04")], partial)
    assert rows[0]["funding_periods"] == 0
    assert rows[0]["funding_bps"] != rows[0]["funding_bps"]


def test_hourly_settlements_bucket_into_the_day_they_accrued_in():
    """Same rule as every other panel: a settlement stamped T paid for the
    interval ending at T, so 00:00 on D+1 belongs to day D. Getting this wrong
    on one panel and right on the others makes every cross-venue comparison
    carry a constant offset."""
    rates = {at("2026-03-04", hour): 1.0 for hour in range(1, 24)}
    rates[at("2026-03-05", 0, 40)] = 1.0          # prints 40ms late
    table = funding_by_day(rates)
    assert table["2026-03-04"] == (24.0, 24)
    assert "2026-03-05" not in table


def test_parkinson_estimator_is_used_for_intraday_vol():
    """No 1m bars here, so `rv_bps` is a different estimator of the same thing
    and is documented as such rather than passed off as comparable."""
    rows = coin_rows("X", [candle("2026-03-04", high=110.0, low=90.0)],
                     full_day("2026-03-04"))
    assert rows[0]["rv_bps"] == pytest.approx(
        math.log(110.0 / 90.0) / (2 * math.sqrt(math.log(2))) * 10_000.0)
