"""Order book correctness — especially the sequence-gap handling.

The gap tests matter more than they look. A book that silently drifts produces
plausible-looking features that are simply wrong, and nothing downstream can
detect it. These tests pin the behaviour that the book must go *stale* rather
than carry on.
"""

from perpkit.orderbook import OrderBook


def snapshot(bids, asks, seq=100, ts=1_000_000):
    return {
        "arg": {"channel": "books", "instId": "BTC-USDT"},
        "action": "snapshot",
        "data": {
            "bids": bids,
            "asks": asks,
            "ts": str(ts),
            "prevSeqId": "0",
            "seqId": str(seq),
        },
    }


def update(bids, asks, prev_seq, seq, ts=1_000_100):
    return {
        "arg": {"channel": "books", "instId": "BTC-USDT"},
        "action": "update",
        "data": {
            "bids": bids,
            "asks": asks,
            "ts": str(ts),
            "prevSeqId": str(prev_seq),
            "seqId": str(seq),
        },
    }


def basic_book():
    book = OrderBook()
    book.apply(
        snapshot(
            bids=[["100.0", "5"], ["99.5", "10"], ["99.0", "20"]],
            asks=[["100.5", "3"], ["101.0", "8"], ["101.5", "15"]],
        )
    )
    return book


def test_snapshot_populates_and_sorts():
    book = basic_book()
    assert book.is_ready
    bid, ask = book.best_bid_ask()
    assert bid == 100.0
    assert ask == 100.5
    bids, asks = book.top(3)
    assert [level.price for level in bids] == [100.0, 99.5, 99.0]
    assert [level.price for level in asks] == [100.5, 101.0, 101.5]


def test_string_and_numeric_prices_both_work():
    book = OrderBook()
    book.apply(snapshot(bids=[[100.0, 5]], asks=[["100.5", "3"]]))
    assert book.best_bid_ask() == (100.0, 100.5)


def test_update_modifies_and_inserts_levels():
    book = basic_book()
    book.apply(update(bids=[["100.0", "7"], ["99.75", "2"]], asks=[], prev_seq=100, seq=101))
    assert book.bids[100.0] == 7
    assert book.bids[99.75] == 2
    assert book.seq_id == 101


def test_zero_size_removes_a_level():
    book = basic_book()
    book.apply(update(bids=[["100.0", "0"]], asks=[], prev_seq=100, seq=101))
    assert 100.0 not in book.bids
    assert book.best_bid_ask()[0] == 99.5


def test_sequence_gap_marks_book_stale():
    book = basic_book()
    # Expected prevSeqId 100, got 105 -> we missed messages 101..105.
    changed = book.apply(update(bids=[["100.0", "99"]], asks=[], prev_seq=105, seq=106))
    assert changed is False
    assert not book.is_ready
    assert book.resync_count == 1
    assert "sequence gap" in (book.last_gap_reason or "")
    # And crucially, the bad update was NOT applied.
    assert book.bids[100.0] == 5


def test_stale_book_ignores_further_updates_until_snapshot():
    book = basic_book()
    book.apply(update(bids=[], asks=[], prev_seq=105, seq=106))
    assert not book.is_ready
    book.apply(update(bids=[["100.0", "42"]], asks=[], prev_seq=106, seq=107))
    assert not book.is_ready
    assert book.bids[100.0] == 5
    # A fresh snapshot recovers it.
    book.apply(snapshot(bids=[["100.0", "1"]], asks=[["100.5", "1"]], seq=200))
    assert book.is_ready
    assert book.bids[100.0] == 1


def test_duplicate_message_is_skipped_not_treated_as_a_gap():
    book = basic_book()
    book.apply(update(bids=[["100.0", "7"]], asks=[], prev_seq=100, seq=101))
    # Same message arriving twice: prevSeqId 100 != seq_id 101, but seq <= current.
    book.apply(update(bids=[["100.0", "7"]], asks=[], prev_seq=100, seq=101))
    assert book.is_ready
    assert book.resync_count == 0


def test_update_before_snapshot_is_ignored():
    book = OrderBook()
    book.apply(update(bids=[["100.0", "5"]], asks=[["100.5", "5"]], prev_seq=0, seq=1))
    assert not book.is_ready
    assert not book.bids


def test_depth_volume_and_imbalance_inputs():
    book = basic_book()
    bid_volume, ask_volume = book.depth_volume(2)
    assert bid_volume == 15  # 5 + 10
    assert ask_volume == 11  # 3 + 8
    assert book.depth_volume(100) == (35, 26)


def test_mid_and_spread():
    book = basic_book()
    assert book.mid() == 100.25
    assert abs(book.spread() - 0.5) < 1e-9


def test_crossed_book_is_detected():
    book = OrderBook()
    book.apply(snapshot(bids=[["101.0", "5"]], asks=[["100.0", "5"]]))
    assert book.is_crossed()


def test_reset_clears_everything():
    book = basic_book()
    book.reset()
    assert not book.is_ready
    assert not book.bids and not book.asks
    assert book.seq_id == 0
