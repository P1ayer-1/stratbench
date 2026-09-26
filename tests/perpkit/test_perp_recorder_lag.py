"""The exchange clock and the recorder-lag report, on hand-built archives.

Reading the archive on `t`, the time the recorder wrote a message, is wrong
in bursts by up to 56 s behind BloFin's own `ts`.
These tests pin what the replacement clock yields, what the lag report
computes, and that replay's wall-clock columns follow the clock asked for.
"""

import gzip
import json

import numpy as np
import pytest

from perpkit.analysis import recorder_lag
from perpkit.analysis.recorder_lag import (
    hourly_stats,
    per_second_max,
    polled_lags,
    scan_day,
    socket_lags,
    together,
)
from perpkit.analysis.replay import (
    channel_of,
    exchange_ms,
    merged_events,
    on_exchange_clock,
    replay,
)
from perpkit.features import FeatureEngine
from perpkit.orderbook import OrderBook
from perpkit.tape import TradeTape

DAY = "2026-09-11"
DAY_MS = 1_789_084_800_000   # 2026-09-11 00:00 UTC
NOON = DAY_MS + 12 * 3_600_000


def book(ts, seq, *, bids=((99.0, 1.0),), asks=((101.0, 1.0),), action="update"):
    return {"arg": {"channel": "books", "instId": "SUI-USDT"}, "action": action,
            "data": {"bids": [list(level) for level in bids],
                     "asks": [list(level) for level in asks],
                     "ts": str(ts), "seqId": str(seq),
                     "prevSeqId": "0" if action == "snapshot" else str(seq - 1)}}


def trade(*stamps, price="100", side="buy"):
    return {"arg": {"channel": "trades", "instId": "SUI-USDT"},
            "data": [{"price": price, "size": "1", "side": side, "ts": str(s)}
                     for s in stamps]}


def funding():
    return {"arg": {"channel": "funding-rate", "instId": "SUI-USDT"},
            "data": [{"fundingRate": "0.0001", "fundingTime": "1789113600000"}]}


def mark(ts):
    return {"arg": {"channel": "mark-price", "instId": "SUI-USDT"},
            "data": [{"markPrice": "100", "indexPrice": "100", "ts": str(ts)}]}


def open_interest(ts):
    return {"arg": {"channel": "open-interest", "instId": "SUI-USDT"},
            "data": [{"openInterest": "5", "ts": str(ts)}]}


def write_raw(raw_dir, channel, records, *, day=DAY, name=None):
    path = raw_dir / day / (name or f"{channel}-12.jsonl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for t, n, message in records:
            handle.write(json.dumps({"t": t, "n": n, "m": message}) + "\n")
    return path


# ---------------------------------------------------------------------------
# The venue's stamp
# ---------------------------------------------------------------------------


def test_exchange_ms_reads_each_channels_own_stamp():
    """A trade push is stamped by its newest fill; funding carries no stamp;
    open interest's `ts` is a minute bucket, not an event time, and joining on
    it would place a reading up to a minute early."""
    assert exchange_ms(book(1_000, 1)) == 1_000
    assert exchange_ms(trade(2_000, 2_500, 2_100)) == 2_500
    assert exchange_ms(mark(3_000)) == 3_000
    assert exchange_ms(funding()) is None
    assert exchange_ms(open_interest(1_789_300_800_000)) is None
    assert exchange_ms({"arg": {"channel": "books"}, "data": {"bids": []}}) is None
    assert exchange_ms("not a message") is None


def test_channel_of_strips_hour_and_restart_suffix():
    """Restart files (`.r001`) must land in the same channel, and hyphenated
    channel names must survive, or the socket/polled split misfiles them."""
    from pathlib import Path
    assert channel_of(Path("books-14.jsonl.gz")) == "books"
    assert channel_of(Path("books-14.r001.jsonl.gz")) == "books"
    assert channel_of(Path("funding-rate-00.jsonl.gz")) == "funding-rate"
    assert channel_of(Path("mark-price-23.r002.jsonl.gz")) == "mark-price"


# ---------------------------------------------------------------------------
# The exchange clock
# ---------------------------------------------------------------------------


def test_exchange_clock_takes_the_newest_stamp_so_far_plus_lag():
    """A book arriving after a newer trade is stamped at the trade's time, not
    its own: the venue sent it no earlier. Leading unstamped events take the
    first stamp. Order is untouched and stamps never decrease."""
    events = [
        (1_000, 1, funding()),          # nothing stamped yet: held
        (1_050, 2, book(900, 1)),       # first stamp: 900
        (1_060, 3, trade(950)),         # 950
        (1_070, 4, book(920, 2)),       # older than the trade before it
        (1_200, 5, book(1_100, 3)),
    ]
    out = list(on_exchange_clock(events, lag_ms=10))
    assert [(stamp, n) for stamp, n, _ in out] == [
        (910, 1), (910, 2), (960, 3), (960, 4), (1_110, 5)]
    assert [m for _, _, m in out] == [m for _, _, m in events]


def test_exchange_clock_without_any_stamp_keeps_receive_time():
    events = [(5_000, 1, funding()), (35_000, 2, funding())]
    assert list(on_exchange_clock(events, lag_ms=15)) == events


def test_merged_events_keeps_polled_channels_out_of_the_socket_clock(tmp_path):
    """The socket is 4 s behind; a REST poll written by another process in the
    meantime is fresh. Restamped together, the poll's stamp would enter the
    socket's running maximum and date the backlogged book and trade 3.85 s
    later than BloFin sent them. Kept apart, the poll keeps its own `t` and
    lands after both."""
    raw = tmp_path / "SUI-USDT" / "raw"
    write_raw(raw, "books", [(5_000, 1, book(1_000, 1, action="snapshot"))])
    write_raw(raw, "trades", [(6_000, 2, trade(1_500))])
    write_raw(raw, "mark-price", [(4_900, 77, mark(4_850))])

    received = [(t, n) for t, n, _ in merged_events(raw, DAY, quiet=True)]
    assert received == [(4_900, 77), (5_000, 1), (6_000, 2)]

    stamped = [(t, n) for t, n, _ in merged_events(raw, DAY, clock="exchange", quiet=True)]
    assert stamped == [(1_000, 1), (1_500, 2), (4_900, 77)]

    with pytest.raises(ValueError):
        list(merged_events(raw, DAY, clock="local", quiet=True))


def test_merged_events_channel_filter(tmp_path):
    raw = tmp_path / "SUI-USDT" / "raw"
    write_raw(raw, "books", [(1_000, 1, book(900, 1, action="snapshot"))])
    write_raw(raw, "mark-price", [(1_500, 9, mark(1_450))])
    only = list(merged_events(raw, DAY, channels={"books"}, quiet=True))
    assert [n for _, n, _ in only] == [1]


# ---------------------------------------------------------------------------
# The lag report
# ---------------------------------------------------------------------------


def test_socket_lags_split_at_the_freshest_stamp():
    """`venue` is how far a message was already behind a newer one on the same
    socket when BloFin sent it; `after` is everything after that. A book 5.2 s
    late behind a trade 0.1 s late is 5.0 s venue and 0.2 s after."""
    lags, written = socket_lags([
        (10_100, 1, trade(10_000)),
        (10_150, 2, funding()),
        (10_200, 3, book(5_000, 1)),
        (10_300, 4, book(10_250, 2)),
    ])
    books = lags["books"]
    assert books.lag.tolist() == [5_200, 50]
    assert books.venue.tolist() == [5_000, 0]
    assert books.after.tolist() == [200, 50]
    assert lags["trades"].lag.tolist() == [100]
    assert lags["trades"].venue.tolist() == [0]
    assert written == [10_100, 10_150, 10_200, 10_300]


def test_polled_lags_have_no_venue_part():
    lags = polled_lags([(3_600, 1, mark(3_550)), (13_700, 2, mark(13_500))])
    assert lags.lag.tolist() == [50, 200]
    assert lags.venue.tolist() == [0, 0]


def test_hourly_stats_by_hand():
    lags, _ = socket_lags([
        (10_100, 1, trade(10_000)),
        (10_200, 2, book(5_000, 1)),
        (10_300, 3, book(10_250, 2)),
    ])
    stats = hourly_stats(lags["books"], day_start_ms=0, late_ms=1_000)
    assert list(stats) == [0]
    hour = stats[0]
    assert hour.n == 2
    assert hour.p50 == pytest.approx(2_625)      # halfway between 50 and 5,200
    assert hour.max == 5_200 and hour.min == 50
    assert hour.late_share == 0.5
    assert hour.venue_p90 == pytest.approx(4_500)
    assert hour.after_p90 == pytest.approx(185)   # 50 + 0.9 * 150


def test_per_second_max_leaves_silent_seconds_nan():
    out = per_second_max(np.array([1_500, 1_700, 2_500]), np.array([10, 30, 20]), 0)
    assert np.isnan(out[0])
    assert out[1] == 30 and out[2] == 20


def test_together_by_hand():
    """Three instruments, four seconds. Late seconds: s0 (A and B of three
    active) and s1 (A of three). P(late) = 3 late / 10 active; of the late
    cells' other active instruments, 2 of 6 were late too."""
    late = 2_000
    matrix = np.array([
        [late, late, 0, np.nan],     # A
        [late, 0, 0, 0],             # B
        [0, 0, np.nan, 0],           # C
    ])
    result = together(matrix, late_ms=1_000)
    assert result.late_seconds == 2
    assert result.median_share == pytest.approx(0.5)
    assert result.p_late == pytest.approx(0.3)
    assert result.p_late_given_other == pytest.approx(1 / 3)
    assert result.lift == pytest.approx((1 / 3) / 0.3)


def test_together_when_fully_shared_lift_is_one_over_p():
    """A frozen event loop delays every socket at once; the statistic must say
    so, or a shared backlog could not be told from independent ones."""
    result = together(np.array([[2_000, 0], [2_000, 0]]), late_ms=1_000)
    assert result.median_share == 1.0
    assert result.p_late == 0.5
    assert result.p_late_given_other == 1.0
    assert result.lift == pytest.approx(2.0)


def test_scan_day_end_to_end(tmp_path):
    raw = tmp_path / "SUI-USDT" / "raw"
    write_raw(raw, "books", [
        (NOON + 1_000, 1, book(NOON + 900, 1, action="snapshot")),
        (NOON + 9_000, 3, book(NOON + 1_000, 2)),       # 8 s late
    ])
    write_raw(raw, "trades", [(NOON + 2_000, 2, trade(NOON + 1_950))])
    write_raw(raw, "funding-rate", [(NOON + 9_500, 4, funding())])
    write_raw(raw, "mark-price", [(NOON + 3_000, 50, mark(NOON + 2_880))])

    scan = scan_day(raw, DAY, late_ms=1_000)
    books = scan.hours["books"][12]
    assert books.n == 2 and books.max == 8_000 and books.min == 100
    assert books.late_share == 0.5
    assert books.venue_p90 == pytest.approx(855)    # 0 and 950, p90
    assert scan.hours["trades"][12].max == 50
    assert scan.hours["mark-price"][12].max == 120
    second = 12 * 3600
    assert scan.written[second + 1] == 1 and scan.written[second + 9] == 2
    assert scan.lag_max[second + 9] == 8_000
    assert scan.after_max[second + 9] == 7_050      # 9,000 - freshest 1,950


# ---------------------------------------------------------------------------
# Consumers of the clock
# ---------------------------------------------------------------------------


def test_compute_takes_its_wall_clock_from_now_ms():
    """Replay used to fill these three columns from the replaying machine's
    clock, which describes nothing about the archived moment."""
    bk, tape, engine = OrderBook(), TradeTape(), FeatureEngine()
    bk.apply(book(1_000_000, 1, action="snapshot"))
    tape.add_message(trade(999_000)["data"])
    engine.on_book_event(bk)
    snapshot = engine.compute(bk, tape, now_ms=1_000_250)
    assert snapshot.received_ts == 1_000_250
    assert snapshot.book_age_ms == 250
    assert snapshot.tape_staleness_s == pytest.approx(1.25)
    assert snapshot.ts == 1_000_000


@pytest.mark.parametrize("clock, expected", [
    ("receive", [10_000, 10_050, 10_100]),
    ("exchange", [9_015, 9_915, 9_915]),
])
def test_replay_passes_the_chosen_clock_to_compute(tmp_path, monkeypatch, clock, expected):
    raw = tmp_path / "SUI-USDT" / "raw"
    write_raw(raw, "books", [(10_000, 1, book(9_000, 1, action="snapshot")),
                             (10_100, 3, book(9_500, 2, bids=((99.5, 1.0),)))])
    write_raw(raw, "trades", [(10_050, 2, trade(9_900))])

    seen = []
    real = FeatureEngine.compute

    def spy(self, bk, tape, now_ms=None):
        seen.append(now_ms)
        return real(self, bk, tape, now_ms=now_ms)

    monkeypatch.setattr(FeatureEngine, "compute", spy)
    replay(raw, tmp_path / "out", date=DAY, sample_ms=1_000, horizons=(300.0,),
           threshold_bps=10.0, clock=clock, lag_ms=15)
    assert seen == expected


def test_recorder_lag_refuses_unknown_instruments(tmp_path):
    (tmp_path / "SUI-USDT" / "raw").mkdir(parents=True)
    with pytest.raises(SystemExit) as refused:
        recorder_lag.main(["--data-dir", str(tmp_path), "--instruments", "XRP-USDT"])
    assert "XRP-USDT" in str(refused.value) and "SUI-USDT" in str(refused.value)
