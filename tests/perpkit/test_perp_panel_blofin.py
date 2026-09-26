"""The BloFin panel: same schema, same alignment rule, different venue.

The point of this file is that `factor_panel.py` reads every venue's panel
unchanged, so panels are only comparable if they agree on the things the harness cannot
see: what a row's timestamp means, when funding accrued, and whether the last
row is a whole day. Each test names the specific way a mismatch would show up
as a venue difference that is really a parsing difference.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from perpkit.analysis.panel_blofin import (
    Instrument,
    fetch_candles,
    fetch_funding_history,
    funding_by_day,
    instrument_rows,
    liquid_instruments,
    write_spreads,
)


def at(date_string, hour=0, millisecond=0):
    base = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(base.timestamp() * 1000) + hour * 3_600_000 + millisecond


class FakeApi:
    """Only the four endpoints this module calls. No socket, no credentials."""

    def __init__(self, *, tickers=(), rates=(), candles=(), funding=()):
        self._tickers, self._rates = list(tickers), list(rates)
        self._candles, self._funding = list(candles), list(funding)
        self.funding_cursors = []

    def getTickers(self):
        return {"data": self._tickers}

    def getFundingRate(self, instId=None):
        return {"data": self._rates}

    def getCandlesticks(self, instId, bar=None, limit=None):
        return {"data": self._candles}

    def getFundingRateHistory(self, instId, after=None, limit=None):
        self.funding_cursors.append(after)
        page = [row for row in self._funding
                if after is None or int(row["fundingTime"]) < int(after)]
        return {"data": page[:int(limit)]}


# ---------------------------------------------------------------------------
# Funding alignment, which must match the shared schema in perpkit/analysis/panel.py
# ---------------------------------------------------------------------------


def test_funding_buckets_match_the_shared_panel_rule():
    """A settlement pays for the interval ENDING at its stamp, on every venue.

    If two panels disagreed by one settlement, every cross-venue comparison
    would carry a constant offset of a third of a day's funding - large next to
    carry differences of a few bps a day.
    """
    rates = {
        at("2026-03-04", 8): 3.0,
        at("2026-03-04", 16): 2.0,
        at("2026-03-05", 0, 2): 1.0,       # prints two milliseconds late
    }
    assert funding_by_day(rates) == {"2026-03-04": (6.0, 3)}


def test_four_hourly_instruments_sum_six_settlements_into_one_day():
    """BloFin runs some instruments on 4h and some on 8h.

    A daily total is the right unit precisely because it is cadence-independent
    - it is what the position actually paid that day. Comparing per-settlement
    rates across instruments with different intervals would say a 4h contract
    is half as expensive as it is.
    """
    rates = {at("2026-03-04", h): 1.0 for h in (4, 8, 12, 16, 20)}
    rates[at("2026-03-05", 0)] = 1.0
    assert funding_by_day(rates) == {"2026-03-04": (6.0, 6)}


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------


def candle(ts, open_, high, low, close, quote, confirm="1"):
    """BloFin's array shape: ts, o, h, l, c, vol, volCurrency, volQuote, confirm."""
    return [str(ts), str(open_), str(high), str(low), str(close),
            "1", "1", str(quote), confirm]


def test_the_day_in_progress_is_dropped(tmp_path):
    """BloFin returns today with `confirm = "0"` and a close that is just "now".

    Keeping it gives the newest row a return over an unknown fraction of a day,
    at the end of the sample - the one place a wrong row is least likely to be
    noticed and most likely to be the row a live signal reads.
    """
    api = FakeApi(candles=[
        candle(at("2026-03-04"), 10, 11, 9, 10.5, 100.0),
        candle(at("2026-03-05"), 10.5, 12, 10, 11.0, 50.0, confirm="0"),
    ])
    rows = fetch_candles(api, "X-USDT", cache=tmp_path, refresh=True)
    assert len(rows) == 1
    assert rows[0][0] == at("2026-03-04")


def test_candles_come_back_oldest_first(tmp_path):
    """The endpoint returns newest first and every trailing window here assumes
    the opposite. A reversed series computes each feature from the future."""
    api = FakeApi(candles=[
        candle(at("2026-03-06"), 12, 12, 12, 12, 1.0),
        candle(at("2026-03-04"), 10, 10, 10, 10, 1.0),
        candle(at("2026-03-05"), 11, 11, 11, 11, 1.0),
    ])
    rows = fetch_candles(api, "X-USDT", cache=tmp_path, refresh=True)
    assert [row[0] for row in rows] == sorted(row[0] for row in rows)
    assert [row[4] for row in rows] == [10.0, 11.0, 12.0]


def test_candles_are_cached_and_not_refetched(tmp_path):
    api = FakeApi(candles=[candle(at("2026-03-04"), 10, 10, 10, 10, 1.0)])
    fetch_candles(api, "X-USDT", cache=tmp_path, refresh=True)
    empty = FakeApi(candles=[])
    rows = fetch_candles(empty, "X-USDT", cache=tmp_path, refresh=False)
    assert len(rows) == 1, "the cache was ignored"


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def test_rows_carry_the_schema_the_shared_harness_expects():
    rows = instrument_rows(
        "SUI-USDT",
        [(at("2026-03-04"), 10.0, 11.0, 9.0, 10.5, 1234.0)],
        {"2026-03-04": (6.0, 3)})
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "SUIUSDT", "symbols must use the shared BASEQUOTE form"
    assert row["funding_bps"] == 6.0 and row["funding_periods"] == 3
    assert row["minutes"] == 1440
    # Parkinson: log(H/L) / (2 sqrt(ln 2)), in bps.
    import math
    assert row["rv_bps"] == pytest.approx(
        math.log(11.0 / 9.0) / (2 * math.sqrt(math.log(2))) * 10_000.0)


def test_a_day_with_no_settlement_is_nan_not_zero():
    """Zero funding and unknown funding are different claims, and the harness
    treats NaN as "cannot speak for this day" while zero is a real rate."""
    rows = instrument_rows("X-USDT",
                           [(at("2026-03-04"), 10.0, 10.0, 10.0, 10.0, 1.0)], {})
    assert rows[0]["funding_periods"] == 0
    assert rows[0]["funding_bps"] != rows[0]["funding_bps"]


# ---------------------------------------------------------------------------
# The universe
# ---------------------------------------------------------------------------


def ticker(inst_id, last, base_volume, bid, ask):
    return {"instId": inst_id, "last": str(last),
            "volCurrency24h": str(base_volume), "vol24h": "0",
            "bidPrice": str(bid), "askPrice": str(ask)}


def test_dollar_volume_is_derived_from_base_units_not_contracts():
    """`getTickers` reports contracts and base units, never quote.

    Ranking on `vol24h` would compare a 1000x-multiplier meme contract with BTC
    on contract COUNT, which is not a size and would put the thinnest
    instruments at the top of the universe.
    """
    api = FakeApi(tickers=[
        ticker("BIG-USDT", 100.0, 1000.0, 99.9, 100.1),      # $100,000
        ticker("SMALL-USDT", 0.001, 1_000_000.0, 0.00099, 0.00101),  # $1,000
    ], rates=[])
    out = liquid_instruments(api, top=10, min_volume=0.0)
    assert [item.inst_id for item in out] == ["BIG-USDT", "SMALL-USDT"]
    assert out[0].volume_usd == pytest.approx(100_000.0)


def test_non_usdt_and_crossed_quotes_are_excluded():
    api = FakeApi(tickers=[
        ticker("BTC-USDC", 100.0, 1000.0, 99.9, 100.1),
        ticker("BAD-USDT", 100.0, 1000.0, 100.1, 99.9),      # crossed
        ticker("OK-USDT", 100.0, 1000.0, 99.9, 100.1),
    ], rates=[])
    assert [i.inst_id for i in liquid_instruments(api, top=10, min_volume=0.0)] \
        == ["OK-USDT"]


def test_funding_interval_is_read_per_instrument():
    """BloFin runs 1h, 4h and 8h contracts side by side, and the interval is
    needed to know a daily total is three settlements or twenty-four."""
    api = FakeApi(
        tickers=[ticker("A-USDT", 100.0, 1000.0, 99.9, 100.1),
                 ticker("B-USDT", 100.0, 1000.0, 99.9, 100.1)],
        rates=[{"instId": "A-USDT", "fundingInterval": "4",
                "fundingIntervalUnit": "hour"},
               {"instId": "B-USDT", "fundingInterval": "8",
                "fundingIntervalUnit": "hour"}])
    out = {item.inst_id: item.funding_interval_hours
           for item in liquid_instruments(api, top=10, min_volume=0.0)}
    assert out == {"A-USDT": 4.0, "B-USDT": 8.0}


def test_spread_snapshot_is_written_beside_the_panel(tmp_path):
    path = tmp_path / "blofin-spreads.csv"
    write_spreads([Instrument("SUI-USDT", 5.6e6, 4.25, 8.0)], path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("symbol,inst_id")
    assert lines[1].startswith("SUIUSDT,SUI-USDT,5600000,4.250,8")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_funding_history_pages_backwards_without_looping(tmp_path):
    """The cursor must move to the OLDEST stamp seen.

    Asking for the newest page again returns the same rows forever; the loop
    would spend its whole page budget on one page and every instrument's
    history would stop at the same recent date.
    """
    history = [{"fundingTime": str(at("2026-03-01") + 8 * 3_600_000 * i),
                "fundingRate": "0.0001"} for i in range(250)]
    api = FakeApi(funding=list(reversed(history)))
    rates = fetch_funding_history(api, "X-USDT", cache=tmp_path, refresh=True,
                                  pages=5)
    assert len(rates) == 250
    assert len(set(api.funding_cursors)) == len(api.funding_cursors)
