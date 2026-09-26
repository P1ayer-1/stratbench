"""Feature math, including the OFI recursion and the recorder's label timing.

The OFI tests encode the Cont/Kukanov/Stoikov definition case by case, because
that recursion is the single easiest thing in this codebase to get subtly
backwards — and a sign error there would flip the direction of every trade.
"""

import math

import pytest

from perpkit.features import FeatureEngine, FeatureSnapshot
from perpkit.orderbook import OrderBook
from perpkit.recorder import FeatureRecorder, LabelConfig
from perpkit.tape import TradeTape


def book_at(bid, bid_size, ask, ask_size, ts=1_000_000, seq=1):
    book = OrderBook()
    book.apply(
        {
            "action": "snapshot",
            "data": {
                "bids": [[bid, bid_size], [bid - 1, 100]],
                "asks": [[ask, ask_size], [ask + 1, 100]],
                "ts": str(ts),
                "seqId": str(seq),
                "prevSeqId": "0",
            },
        }
    )
    return book


def book_with_mid(mid, ts, half_spread=0.01):
    """A valid two-sided book centred on `mid`.

    Note the half-spread: a book with bid == ask is *locked*, which
    `OrderBook.is_crossed()` treats as untradeable and the feature engine
    refuses to process. Tests that want a working book must build a real one.
    """
    return book_at(mid - half_spread, 10, mid + half_spread, 10, ts=ts)


# ---------------------------------------------------------------------------
# OFI
# ---------------------------------------------------------------------------


def test_first_event_has_zero_ofi():
    """With no previous book there is no flow to measure."""
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 101, 10))
    assert engine.ofi_history[-1][1] == 0.0


def test_bid_price_improves_contributes_full_new_size():
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 101, 10, ts=1000))
    engine.on_book_event(book_at(100.5, 7, 101, 10, ts=1100))
    # Bid ticked up: +7 (all new demand). Ask unchanged price and size: -0.
    assert engine.ofi_history[-1][1] == pytest.approx(7.0)


def test_bid_price_falls_removes_old_size():
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 101, 10, ts=1000))
    engine.on_book_event(book_at(99.5, 4, 101, 10, ts=1100))
    # Bid ticked down: -10 (the old demand vanished).
    assert engine.ofi_history[-1][1] == pytest.approx(-10.0)


def test_bid_same_price_contributes_only_the_delta():
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 101, 10, ts=1000))
    engine.on_book_event(book_at(100, 15, 101, 10, ts=1100))
    assert engine.ofi_history[-1][1] == pytest.approx(5.0)


def test_ask_price_falls_is_negative_pressure():
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 101, 10, ts=1000))
    engine.on_book_event(book_at(100, 10, 100.5, 6, ts=1100))
    # Ask ticked down (sellers pressing): -6.
    assert engine.ofi_history[-1][1] == pytest.approx(-6.0)


def test_ask_price_rises_is_positive_pressure():
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 101, 10, ts=1000))
    engine.on_book_event(book_at(100, 10, 101.5, 3, ts=1100))
    # Ask ticked up (supply pulled): +10, the old ask size.
    assert engine.ofi_history[-1][1] == pytest.approx(10.0)


def test_ofi_is_antisymmetric_for_mirrored_pressure():
    buy_side = FeatureEngine()
    buy_side.on_book_event(book_at(100, 10, 101, 10, ts=1000))
    buy_side.on_book_event(book_at(100, 20, 101, 10, ts=1100))

    sell_side = FeatureEngine()
    sell_side.on_book_event(book_at(100, 10, 101, 10, ts=1000))
    sell_side.on_book_event(book_at(100, 10, 101, 20, ts=1100))

    assert buy_side.ofi_history[-1][1] == pytest.approx(
        -sell_side.ofi_history[-1][1]
    )


def test_ofi_window_sums_recent_increments_only():
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 101, 10, ts=1_000_000))
    engine.on_book_event(book_at(100, 20, 101, 10, ts=1_000_100))  # +10
    engine.on_book_event(book_at(100, 25, 101, 10, ts=1_003_000))  # +5, 3s later
    assert engine._ofi_sum(1_003_000, 1.0) == pytest.approx(5.0)
    assert engine._ofi_sum(1_003_000, 5.0) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Microprice, OBI, spread
# ---------------------------------------------------------------------------


def test_microprice_leans_toward_the_thin_side():
    """Heavy bid, thin ask -> fair value sits above the mid, near the ask."""
    engine = FeatureEngine()
    book = book_at(100, 90, 101, 10)
    engine.on_book_event(book)
    snapshot = engine.compute(book, TradeTape())
    assert snapshot.mid == 100.5
    # (100*10 + 101*90) / 100 = 100.9
    assert snapshot.microprice == pytest.approx(100.9)
    assert snapshot.microprice_delta_bps > 0


def test_microprice_equals_mid_when_balanced():
    engine = FeatureEngine()
    book = book_at(100, 50, 101, 50)
    engine.on_book_event(book)
    snapshot = engine.compute(book, TradeTape())
    assert snapshot.microprice == pytest.approx(snapshot.mid)
    assert snapshot.microprice_delta_bps == pytest.approx(0.0)


def test_obi_bounds_and_sign():
    engine = FeatureEngine()
    book = book_at(100, 100, 101, 0.0001)
    engine.on_book_event(book)
    snapshot = engine.compute(book, TradeTape())
    assert 0 < snapshot.obi_1 <= 1
    assert snapshot.obi_1 > 0.9


def test_spread_bps_calculation():
    engine = FeatureEngine()
    book = book_at(100, 10, 101, 10)
    engine.on_book_event(book)
    snapshot = engine.compute(book, TradeTape())
    # spread 1 on mid 100.5 -> 99.5bps
    assert snapshot.spread_bps == pytest.approx(1 / 100.5 * 10_000, rel=1e-6)


def test_invalid_book_yields_invalid_snapshot():
    engine = FeatureEngine()
    empty = OrderBook()
    snapshot = engine.compute(empty, TradeTape())
    assert not snapshot.is_valid
    assert snapshot.mid is None
    # Every numeric field must still be a real number, never NaN.
    for key, value in snapshot.to_dict().items():
        if isinstance(value, float):
            assert not math.isnan(value), key


def test_crossed_book_is_never_valid():
    engine = FeatureEngine()
    crossed = OrderBook()
    crossed.apply(
        {
            "action": "snapshot",
            "data": {"bids": [[101, 5]], "asks": [[100, 5]], "ts": "1", "seqId": "1"},
        }
    )
    assert not engine.compute(crossed, TradeTape()).is_valid


# ---------------------------------------------------------------------------
# Returns and volatility
# ---------------------------------------------------------------------------


def test_returns_measured_against_the_right_anchor():
    engine = FeatureEngine()
    engine.on_book_event(book_with_mid(100, ts=1_000_000))
    engine.on_book_event(book_with_mid(101, ts=1_002_000))
    # 2 seconds later, mid moved 100 -> 101
    expected = math.log(101 / 100) * 10_000
    assert engine._return_bps(1_002_000, 101.0, 2.0) == pytest.approx(expected)
    # Asking for a 5s return with only 2s of history must NOT silently reuse
    # the 2s anchor and pass a short move off as a long one.
    assert engine._return_bps(1_002_000, 101.0, 5.0) == 0.0


def test_return_is_zero_when_history_is_too_short():
    """No fabricated returns against an anchor we never observed."""
    engine = FeatureEngine()
    engine.on_book_event(book_with_mid(100, ts=1_000_000))
    assert engine._return_bps(1_000_000, 100.0, 30.0) == 0.0


def test_flat_prices_give_zero_volatility():
    engine = FeatureEngine()
    for index in range(20):
        engine.on_book_event(book_with_mid(100, ts=1_000_000 + index * 100))
    assert len(engine.mid_history) == 20  # guard: the books really were valid
    assert engine._realized_vol(1_000_000 + 1900, 10.0) == pytest.approx(0.0)


def test_moving_prices_give_positive_volatility():
    engine = FeatureEngine()
    for index in range(40):
        engine.on_book_event(
            book_with_mid(100 + (index % 2), ts=1_000_000 + index * 100)
        )
    assert engine._realized_vol(1_000_000 + 3900, 10.0) > 0


def test_locked_book_is_rejected_like_a_crossed_one():
    """bid == ask carries no directional information and must not be traded."""
    engine = FeatureEngine()
    engine.on_book_event(book_at(100, 10, 100, 10, ts=1_000_000))
    assert len(engine.mid_history) == 0


# ---------------------------------------------------------------------------
# Trade tape
# ---------------------------------------------------------------------------


def test_tape_flow_imbalance_uses_aggressor_side():
    tape = TradeTape()
    tape.add_message(
        [
            {"price": "100", "size": "3", "side": "buy", "ts": "1000000"},
            {"price": "100", "size": "1", "side": "sell", "ts": "1000100"},
        ]
    )
    # (3 - 1) / 4
    assert tape.flow_imbalance(10.0) == pytest.approx(0.5)


def test_tape_with_no_trades_is_neutral_not_undefined():
    assert TradeTape().flow_imbalance(1.0) == 0.0


def test_tape_trims_by_exchange_time():
    tape = TradeTape(window_seconds=1.0)
    tape.add_message([{"price": "100", "size": "1", "side": "buy", "ts": "1000000"}])
    tape.add_message([{"price": "100", "size": "1", "side": "sell", "ts": "1005000"}])
    assert len(tape.trades) == 1
    assert not tape.trades[0].is_buy


def test_tape_ignores_malformed_trades():
    tape = TradeTape()
    added = tape.add_message(
        [
            {"price": "0", "size": "1", "side": "buy", "ts": "1"},
            {"price": "100", "size": "0", "side": "buy", "ts": "1"},
            {"price": "abc", "size": "1", "side": "buy", "ts": "1"},
            {"price": "100", "size": "2", "side": "buy", "ts": "1"},
        ]
    )
    assert added == 1


def test_vwap():
    tape = TradeTape()
    tape.add_message(
        [
            {"price": "100", "size": "1", "side": "buy", "ts": "1000"},
            {"price": "102", "size": "3", "side": "buy", "ts": "1001"},
        ]
    )
    # (100*1 + 102*3) / 4 = 101.5
    assert tape.vwap(10.0) == pytest.approx(101.5)


# ---------------------------------------------------------------------------
# Recorder — the lookahead-bias guard
# ---------------------------------------------------------------------------


def _valid_snapshot(ts, mid):
    from perpkit.features import FeatureSnapshot

    return FeatureSnapshot(ts=ts, mid=mid, is_valid=True)


def test_recorder_writes_nothing_until_the_horizon_has_passed(tmp_path):
    recorder = FeatureRecorder(
        tmp_path,
        label_config=LabelConfig(horizons_seconds=(5.0,), threshold_bps=1.0),
        sample_interval_ms=0,
    )
    recorder.observe(_valid_snapshot(1_000_000, 100.0))
    recorder.observe(_valid_snapshot(1_002_000, 101.0))
    assert recorder.rows_written == 0  # only 2s of the 5s horizon elapsed
    recorder.close()


def test_recorder_labels_from_strictly_future_prices(tmp_path):
    recorder = FeatureRecorder(
        tmp_path,
        label_config=LabelConfig(horizons_seconds=(5.0,), threshold_bps=1.0),
        sample_interval_ms=0,
    )
    recorder.observe(_valid_snapshot(1_000_000, 100.0))
    recorder.observe(_valid_snapshot(1_005_000, 101.0))
    recorder.observe(_valid_snapshot(1_006_000, 101.0))
    recorder.close()

    import csv

    files = list(tmp_path.glob("features-*.csv"))
    assert len(files) == 1
    rows = list(csv.DictReader(files[0].open()))
    assert len(rows) == 1
    row = rows[0]
    # Forward return from mid 100 to the mid observed 5s later (101).
    expected_bps = (101 / 100 - 1) * 10_000
    assert float(row["fwd_ret_bps_5s"]) == pytest.approx(expected_bps, abs=0.01)
    assert int(row["label_5s"]) == 1


def test_recorder_labels_a_drop_as_negative(tmp_path):
    recorder = FeatureRecorder(
        tmp_path,
        label_config=LabelConfig(horizons_seconds=(1.0,), threshold_bps=1.0),
        sample_interval_ms=0,
    )
    recorder.observe(_valid_snapshot(1_000_000, 100.0))
    recorder.observe(_valid_snapshot(1_001_000, 99.0))
    recorder.observe(_valid_snapshot(1_002_000, 99.0))
    recorder.close()

    import csv

    rows = list(csv.DictReader(next(tmp_path.glob("features-*.csv")).open()))
    assert int(rows[0]["label_1s"]) == -1


def test_recorder_labels_small_move_as_flat(tmp_path):
    recorder = FeatureRecorder(
        tmp_path,
        label_config=LabelConfig(horizons_seconds=(1.0,), threshold_bps=50.0),
        sample_interval_ms=0,
    )
    recorder.observe(_valid_snapshot(1_000_000, 100.0))
    recorder.observe(_valid_snapshot(1_001_000, 100.05))
    recorder.observe(_valid_snapshot(1_002_000, 100.05))
    recorder.close()

    import csv

    rows = list(csv.DictReader(next(tmp_path.glob("features-*.csv")).open()))
    assert int(rows[0]["label_1s"]) == 0


def test_recorder_skips_invalid_snapshots(tmp_path):
    from perpkit.features import FeatureSnapshot

    recorder = FeatureRecorder(
        tmp_path,
        label_config=LabelConfig(horizons_seconds=(1.0,)),
        sample_interval_ms=0,
    )
    recorder.observe(FeatureSnapshot(ts=1_000_000, mid=None, is_valid=False))
    recorder.observe(_valid_snapshot(1_002_000, 100.0))
    recorder.close()
    assert recorder.rows_written == 0


def test_recorder_downsamples_to_the_configured_interval(tmp_path):
    recorder = FeatureRecorder(
        tmp_path,
        label_config=LabelConfig(horizons_seconds=(1.0,)),
        sample_interval_ms=500,
    )
    # 20 snapshots 100ms apart spanning 2s -> ~4 sampled rows, not 20.
    for index in range(30):
        recorder.observe(_valid_snapshot(1_000_000 + index * 100, 100.0))
    recorder.close()
    assert 0 < recorder.rows_written <= 6


def test_history_seconds_distinguishes_warmup_from_a_real_zero():
    """A 0.0 return during warmup must be identifiable as padding."""
    engine = FeatureEngine()
    for index in range(5):
        engine.on_book_event(book_with_mid(100, ts=1_000_000 + index * 100))
    book = book_with_mid(100, ts=1_000_400)
    engine.on_book_event(book)
    snapshot = engine.compute(book, TradeTape())
    assert snapshot.is_valid
    # Only ~0.4s of history, so ret_30s is padding, not a measured zero.
    assert snapshot.ret_30s == 0.0
    assert snapshot.history_seconds < 30


def test_snapshot_ts_is_exchange_time_not_wall_clock():
    """Labels are measured in exchange time so replayed history labels the
    same way live data does. Regression test: `ts` was briefly wall-clock,
    which silently stopped the recorder from ever labelling replayed data.
    """
    engine = FeatureEngine()
    book = book_with_mid(100, ts=1_700_000_000_000)
    engine.on_book_event(book)
    snapshot = engine.compute(book, TradeTape())
    assert snapshot.ts == 1_700_000_000_000
    assert snapshot.received_ts > snapshot.ts  # wall clock is "now", not 2023
    assert snapshot.book_age_ms == snapshot.received_ts - snapshot.ts


# ---------------------------------------------------------------------------
# _Series — the O(log n) windowed-query structure
# ---------------------------------------------------------------------------


def test_series_sum_since_matches_a_naive_filter():
    from perpkit.features import _Series

    series = _Series()
    for index in range(500):
        series.append(1000 + index * 10, float(index))
    for cutoff in (1000, 2500, 4000, 5990, 99999):
        naive = sum(v for t, v in zip(series.ts, series.values) if t >= cutoff)
        assert series.sum_since(cutoff) == pytest.approx(naive)


def test_series_value_at_or_before():
    from perpkit.features import _Series

    series = _Series()
    for index in range(10):
        series.append(100 + index * 10, float(index))
    assert series.value_at_or_before(145) == 4.0   # 140 is the latest <= 145
    assert series.value_at_or_before(140) == 4.0   # inclusive
    assert series.value_at_or_before(99) is None   # before all history


def test_series_compaction_preserves_sums():
    """Compaction rebases the cumulative sums; if it were wrong, every
    windowed value would silently drift after the first compaction."""
    from perpkit.features import _Series

    series = _Series()
    for index in range(20_000):
        series.append(1000 + index, 1.0)

    before = series.sum_since(19_000)
    series.compact(15_000)
    assert len(series.ts) < 20_000, "expected compaction to drop old samples"
    assert series.sum_since(19_000) == pytest.approx(before)


def test_series_does_not_compact_below_the_threshold():
    from perpkit.features import _Series

    series = _Series()
    for index in range(100):
        series.append(1000 + index, 1.0)
    series.compact(1099)
    assert len(series.ts) == 100  # too few to be worth an O(n) rebuild


# ---------------------------------------------------------------------------
# The fast path must equal the naive path
# ---------------------------------------------------------------------------


def _naive_realized_vol(pairs, ts, seconds):
    cutoff = ts - int(seconds * 1000)
    samples = [(t, m) for t, m in pairs if t >= cutoff and m > 0]
    if len(samples) < 3:
        return 0.0
    total = sum(
        math.log(samples[i][1] / samples[i - 1][1]) ** 2
        for i in range(1, len(samples))
        if samples[i - 1][1] > 0 and samples[i][1] > 0
    )
    elapsed = (samples[-1][0] - samples[0][0]) / 1000.0
    return math.sqrt(total / elapsed) if elapsed > 0 else 0.0


def _naive_ofi_sum(pairs, ts, seconds):
    cutoff = ts - int(seconds * 1000)
    return sum(v for t, v in pairs if t >= cutoff)


def test_incremental_volatility_equals_a_full_rescan():
    """Regression guard for the O(log n) rewrite.

    The engine accumulates squared returns incrementally instead of rescanning
    history. This pins that the optimisation did not change the numbers — a
    silent drift here would alter every recorded feature.
    """
    import random

    random.seed(3)
    engine = FeatureEngine()
    mid = 60000.0
    base = 1_757_000_000_000
    for index in range(1200):
        mid += random.gauss(0, 0.7)
        engine.on_book_event(book_with_mid(round(mid, 4), ts=base + index * 20))

    pairs = engine.mid_history
    ts = base + 1199 * 20
    for seconds in (10.0, 60.0):
        assert engine._realized_vol(ts, seconds) == pytest.approx(
            _naive_realized_vol(pairs, ts, seconds), abs=1e-12
        )


def test_incremental_ofi_equals_a_full_rescan():
    import random

    random.seed(4)
    engine = FeatureEngine()
    mid = 60000.0
    base = 1_757_000_000_000
    for index in range(1200):
        mid += random.gauss(0, 0.7)
        engine.on_book_event(
            book_at(round(mid - 0.05, 4), random.uniform(1, 20),
                    round(mid + 0.05, 4), random.uniform(1, 20),
                    ts=base + index * 20)
        )

    pairs = engine.ofi_history
    ts = base + 1199 * 20
    for seconds in (1.0, 5.0):
        assert engine._ofi_sum(ts, seconds) == pytest.approx(
            _naive_ofi_sum(pairs, ts, seconds), abs=1e-9
        )


def _read_header(path):
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as handle:
        return next(_csv.reader(handle))


def _record_one_row(data_dir, horizons):
    """Drive a recorder through one complete labelled row."""
    recorder = FeatureRecorder(
        data_dir,
        label_config=LabelConfig(horizons_seconds=horizons, threshold_bps=1.0),
        sample_interval_ms=0,
    )
    span_ms = int(max(horizons) * 1000)
    base = 1_700_000_000_000
    for offset in (0, span_ms, span_ms * 2):
        recorder.observe(
            FeatureSnapshot(ts=base + offset, mid=100.0, is_valid=True)
        )
    recorder.close()
    return recorder


def test_changing_horizons_rolls_to_a_new_file_instead_of_corrupting_the_old(tmp_path):
    # The recorder appends and only writes a header for a new file. Relabelling
    # at different horizons changes the columns, so appending would file every
    # value under the wrong name.
    first = _record_one_row(tmp_path, (1.0,))
    assert first.rows_written >= 1

    files = sorted(path.name for path in tmp_path.glob("features-*.csv"))
    assert len(files) == 1
    original = tmp_path / files[0]
    original_header = _read_header(original)
    assert "fwd_ret_bps_1s" in original_header
    original_bytes = original.read_bytes()

    second = _record_one_row(tmp_path, (2.0,))
    assert second.rows_written >= 1

    # The first file is untouched...
    assert original.read_bytes() == original_bytes
    # ...and the new schema went somewhere else, with its own header.
    paths = list(tmp_path.glob("features-*.csv"))
    assert len(paths) == 2
    rolled = next(path for path in paths if path != original)
    rolled_header = _read_header(rolled)
    assert "fwd_ret_bps_2s" in rolled_header
    assert "fwd_ret_bps_1s" not in rolled_header


def test_same_horizons_keep_appending_to_one_file(tmp_path):
    _record_one_row(tmp_path, (1.0,))
    _record_one_row(tmp_path, (1.0,))
    files = list(tmp_path.glob("features-*.csv"))
    assert len(files) == 1
    with open(files[0], newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    # One header, and more than one data row.
    assert len(lines) >= 3


def test_out_of_order_snapshots_are_dropped_not_mislabelled(tmp_path):
    # Label lookup is a bisect over an assumed-ordered timestamp list; an
    # out-of-order arrival would silently corrupt every later lookup.
    recorder = FeatureRecorder(
        tmp_path,
        label_config=LabelConfig(horizons_seconds=(1.0,), threshold_bps=1.0),
        sample_interval_ms=0,
    )
    base = 1_700_000_000_000
    recorder.observe(FeatureSnapshot(ts=base + 5000, mid=100.0, is_valid=True))
    recorder.observe(FeatureSnapshot(ts=base, mid=999.0, is_valid=True))
    assert recorder.rows_dropped >= 1
    assert recorder._mid_ts == sorted(recorder._mid_ts)
    assert 999.0 not in recorder._mid_values
    recorder.close()
