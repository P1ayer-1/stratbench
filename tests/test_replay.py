"""Books rebuilt from venue messages, and maker fills bracketed honestly."""

import json
from decimal import Decimal

from predkit.rawlog import RawEventLog
from predkit.replay import BookState, MakerFillModel, MakerOrder, iter_market_events, rebuild
from predkit.schema import Book, Contract, Level, Side, Trade, utc

D = Decimal
CLOCK = utc(2026, 9, 12, 14, 0).timestamp()


def kalshi_contract():
    return Contract("kalshi", "KX-1", "q", utc(2026, 9, 12, 14, 15), "kalshi:KX", "default")


def poly_contract():
    return Contract("polymarket", "0xc", "q", utc(2026, 9, 12, 14, 5), "polymarket:uma:0xc", "crypto_fees_v2",
                    min_size=D("5"), yes_token="YES", no_token="NO")


def test_kalshi_no_bids_become_yes_asks():
    state = BookState("KX-1")
    from predkit.replay import kalshi_parser
    parse = kalshi_parser(kalshi_contract())
    (kind, book), = parse(state, 1, "orderbook_snapshot", {"msg": {"yes": [[48, 100], [47, 50]], "no": [[51, 30]]}})
    assert kind == "book"
    assert book.best_bid == D("0.48") and book.best_ask == D("0.49")       # 1 - 0.51
    parse(state, 2, "orderbook_delta", {"msg": {"price": 51, "delta": -30, "side": "no"}})
    parse(state, 3, "orderbook_delta", {"msg": {"price": 50, "delta": 10, "side": "no"}})
    book = state.to_book()
    assert book.best_ask == D("0.50")
    (kind, trade), = parse(state, 4, "trade", {"msg": {"yes_price": 49, "count": 7, "taker_side": "no", "ts": 1}})
    assert (trade.price, trade.size, trade.aggressor) == (D("0.49"), D("7"), Side.SELL)


def test_polymarket_no_token_changes_land_on_the_yes_book():
    from predkit.replay import polymarket_parser
    state = BookState("0xc")
    parse = polymarket_parser(poly_contract())
    parse(state, 1, "book", {"asset_id": "YES", "bids": [{"price": "0.48", "size": "100"}],
                             "asks": [{"price": "0.52", "size": "80"}], "timestamp": "1000"})
    parse(state, 2, "price_change", {"asset_id": "NO", "changes": [{"price": "0.49", "side": "BUY", "size": "20"}]})
    book = state.to_book()
    assert book.best_ask == D("0.51")                    # a NO bid at 0.49
    assert book.size_at(Side.SELL, "0.51") == D("20")


def test_polymarket_wire_shape_price_changes_carry_both_tokens_in_one_frame():
    """The live frame (read 2026-09-13): `price_changes` with an asset id per
    entry, both tokens in one frame, mirrored levels. Before this test the
    parser looked for `changes` and dropped every delta."""
    from predkit.replay import polymarket_parser
    state = BookState("0xc")
    parse = polymarket_parser(poly_contract())
    parse(state, 1, "book", {"asset_id": "YES", "bids": [{"price": "0.48", "size": "100"}],
                             "asks": [{"price": "0.52", "size": "80"}], "timestamp": "1000"})
    (kind, book), = parse(state, 2, "price_change", {
        "market": "0xc", "timestamp": "1789307642631",
        "price_changes": [{"asset_id": "YES", "price": "0.49", "size": "30", "side": "BUY", "hash": "h1"},
                          {"asset_id": "NO", "price": "0.51", "size": "30", "side": "SELL", "hash": "h2"}]})
    assert kind == "book" and book.best_bid == D("0.49") and book.size_at(Side.BUY, "0.49") == D("30")
    assert book.best_ask == D("0.52")
    parse(state, 3, "price_change", {"market": "0xc", "timestamp": "1789307642700",
                                     "price_changes": [{"asset_id": "YES", "price": "0.49", "size": "0", "side": "BUY"}]})
    assert state.best_bid == D("0.48")                    # size 0 removes the level


def test_events_merge_across_channels_on_receive_time_then_sequence(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("trade", {"i": 1})
    log.write("book", {"i": 2})
    log.write("trade", {"i": 3})
    log.close()
    assert [m["i"] for _, _, _, m in iter_market_events(tmp_path)] == [1, 2, 3]


def test_rebuild_walks_the_archive_through_the_parser(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("book_poll", {"orderbook": {"yes": [[48, 10]], "no": [[51, 10]]}})
    log.write("trade", {"msg": {"yes_price": 48, "count": 1, "taker_side": "no", "ts": 1}})
    log.close()
    events = list(rebuild(tmp_path, kalshi_contract()))
    assert [kind for _, (kind, _) in events] == ["book", "trade"]


def wire(message):
    """A frame as the venue's socket sends it (a space after each comma),
    which the recorder archives verbatim."""
    return json.dumps(message, separators=(", ", ":"))


def poly_book(ts, bid="0.48"):
    return wire({"market": "0xc", "asset_id": "YES", "timestamp": str(ts), "hash": f"b{ts}",
                 "bids": [{"price": bid, "size": "100"}], "asks": [{"price": "0.52", "size": "80"}],
                 "event_type": "book"})


def poly_change(ts, price, size, best_bid, side="BUY", hash_="h"):
    return wire({"market": "0xc", "price_changes": [
        {"asset_id": "YES", "price": price, "size": size, "side": side, "hash": hash_,
         "best_bid": best_bid, "best_ask": "0.52"}], "timestamp": str(ts), "event_type": "price_change"})


def poly_print(ts, tx="0xaa", price="0.49", size="7", **extra):
    return wire({"market": "0xc", "asset_id": "YES", "price": price, "size": size, "fee_rate_bps": "0",
                 "side": "SELL", "timestamp": str(ts), "event_type": "last_trade_price",
                 "transaction_hash": tx, **extra})


def archive(tmp_path, frames):
    """(seconds after CLOCK, channel, frame) written through the recorder's own log."""
    now = [CLOCK]
    log = RawEventLog(tmp_path, clock=lambda: now[0])
    for offset, channel, frame in frames:
        now[0] = CLOCK + offset
        log.write(channel, frame)
    log.close()


def test_a_late_copy_from_the_second_socket_is_replayed_once(tmp_path):
    """The defect found in live recordings: the recorder's lagging socket delivers
    exact copies 5-13 s after the first, past its 4,096-frame de-dup window.
    A price_change carries the level's absolute size, so re-applying the
    copy of `add` after `remove` resurrects a bid the venue had pulled (the
    venue's own best_bid says 0.48); a copied book rolls the book back; a
    copied print is a second fill for the fill model."""
    book, add, remove = poly_book(1000), poly_change(1100, "0.49", "30", "0.49"), poly_change(1200, "0.49", "0", "0.48")
    trade = poly_print(1150)
    archive(tmp_path, [(0.0, "book", book), (0.1, "price_change", add), (0.15, "last_trade_price", trade),
                       (0.2, "price_change", remove),
                       (8.0, "book", book), (8.1, "price_change", add), (8.15, "last_trade_price", trade)])

    applied_every_frame = len(list(iter_market_events(tmp_path)))
    dropped = {}
    touches, trades = [], []
    for t, (kind, event) in rebuild(tmp_path, poly_contract(), dropped=dropped):
        if kind == "book":
            touches.append(event.best_bid)
        else:
            trades.append(event)
    assert applied_every_frame == 7
    assert touches == [D("0.48"), D("0.49"), D("0.48")]           # book, add, remove; no copy applied
    assert [(tr.price, tr.size, tr.aggressor) for tr in trades] == [(D("0.49"), D("7"), Side.SELL)]
    assert dropped == {"book": 1, "price_change": 1, "last_trade_price": 1}


def test_distinct_frames_sharing_a_timestamp_and_hash_are_both_applied(tmp_path):
    """The key is the whole frame. (timestamp, hash) is not unique on the
    wire, and keying on it drops real level changes."""
    archive(tmp_path, [(0.0, "book", poly_book(1000)),
                       (0.1, "price_change", poly_change(1100, "0.49", "30", "0.49", hash_="same")),
                       (0.2, "price_change", poly_change(1100, "0.47", "25", "0.49", hash_="same"))])
    dropped = {}
    books = [event for _, (kind, event) in rebuild(tmp_path, poly_contract(), dropped=dropped) if kind == "book"]
    state = books[-1]                                  # the live state, final once the replay is done
    assert state.size_at(Side.BUY, "0.49") == D("30") and state.size_at(Side.BUY, "0.47") == D("25")
    assert dropped == {}


def test_a_print_is_one_print_however_its_frame_was_spelled(tmp_path):
    """A repeat of (transaction, asset, price, size, side, timestamp) is the
    same print even when its bytes differ; a different transaction at the
    same price, size and millisecond is a different fill."""
    respelled = wire(dict(reversed(list(json.loads(poly_print(1000)).items()))))
    archive(tmp_path, [(0.0, "last_trade_price", poly_print(1000)),
                       (6.0, "last_trade_price", respelled),
                       (6.1, "last_trade_price", poly_print(1000, tx="0xbb"))])
    dropped = {}
    trades = [event for _, (kind, event) in rebuild(tmp_path, poly_contract(), dropped=dropped)]
    assert len(trades) == 2
    assert dropped == {"last_trade_price": 1}


def test_a_kalshi_poll_that_repeats_an_earlier_book_is_applied(tmp_path):
    """A polled REST book is our observation, not a venue event: the same
    book after a different one means the book went back, and dropping it as
    a copy would leave the replay on the stale one."""
    first = {"orderbook": {"yes": [[48, 10]], "no": [[51, 10]]}}
    moved = {"orderbook": {"yes": [[49, 10]], "no": [[51, 10]]}}
    archive(tmp_path, [(0.0, "book_poll", first), (1.0, "book_poll", moved), (2.0, "book_poll", first)])
    touches = [event.best_bid for _, (kind, event) in rebuild(tmp_path, kalshi_contract()) if kind == "book"]
    assert touches == [D("0.48"), D("0.49"), D("0.48")]


def bid_order(price="0.48", size="10", queue="0"):
    return MakerOrder("o1", Side.BUY, D(price), D(size), placed_ms=1000, ttl_ms=10_000, queue_ahead=D(queue))


def test_a_print_at_the_bid_fills_the_optimistic_bound_only_when_queued_behind():
    model = MakerFillModel()
    model.place(bid_order(queue="30"))
    model.on_trade(Trade("M", "0.48", "25", Side.SELL, 2000))
    assert [f.size for f in model.fills["optimistic"]] == [D("10")]
    assert model.fills["pessimistic"] == []                # 25 of the 30 ahead traded
    model.on_trade(Trade("M", "0.48", "8", Side.SELL, 3000))
    assert [f.size for f in model.fills["pessimistic"]] == [D("3")]     # 5 clears the queue, 3 for us


def test_a_print_below_the_bid_clears_the_level_for_both_bounds():
    model = MakerFillModel()
    model.place(bid_order(queue="500"))
    model.on_trade(Trade("M", "0.47", "1", Side.SELL, 2000))
    assert model.fills["pessimistic"][0].cause == "cleared"
    assert model.fills["pessimistic"][0].size == D("10")


def test_a_buy_aggressor_never_fills_a_bid():
    model = MakerFillModel()
    model.place(bid_order())
    model.on_trade(Trade("M", "0.48", "50", Side.BUY, 2000))
    assert model.fills["optimistic"] == [] and model.fills["pessimistic"] == []


def test_the_ask_coming_through_the_bid_fills_it():
    model = MakerFillModel()
    model.place(bid_order())
    model.on_book(Book("M", bids=[Level("0.46", 1)], asks=[Level("0.48", 5)], ts_ms=2000))
    assert model.fills["optimistic"][0].cause == "crossed"


def test_an_order_expires_after_its_ttl():
    model = MakerFillModel()
    model.place(bid_order())
    model.on_trade(Trade("M", "0.48", "5", Side.SELL, 1000 + 10_000))
    assert model.fills["optimistic"] == [] and model.expired == ["o1"]


def test_queue_ahead_defaults_to_the_size_resting_at_the_level():
    model = MakerFillModel()
    book = Book("M", bids=[Level("0.48", 40)], asks=[Level("0.50", 1)])
    model.place(bid_order(), book)
    assert model._queue["o1"] == D("40")
