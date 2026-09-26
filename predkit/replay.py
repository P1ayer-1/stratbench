"""Rebuild books and prints from the archive, and fill maker orders against
them honestly.

    python -m predkit.replay data/<venue>/<market_id> --every 10

Replay is what makes the archive worth more than price history. It reads
every channel of a market's raw directory merged on `(t, n)`, applies each
message to a book through the same parsers the live path uses, and emits
`("book", Book)` and `("trade", Trade)` in arrival order. A fill model then
walks those events with the orders a strategy would have posted.

The fill model is a standard passive-fill rule:

  A resting BID at p is filled by the first sell-aggressor print at or below
  p, or by the best ask coming down to or through p. Symmetrically for an
  ask. An order is cancelled by the caller or expires after its TTL.

with a queue-position bracket rather than a single assumption:

  optimistic    the order is alone at its level: the first qualifying print
                fills it (true when it improved the book, which the maker
                template requires)
  pessimistic   the order sits behind `queue_ahead` contracts already at its
                level, which must print first; a print strictly beyond the
                level (a sell below the bid) means the level was cleared and
                fills it outright

Both bounds are reported. The truth is between them and only paper trading
narrows it. Fee on a maker fill is the venue's maker fee, zero on the tiers
transcribed, plus the builder maker fee if any.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from predkit.rawlog import channels_in, iter_directory
from predkit.schema import ZERO, Book, Contract, Level, Side, Trade, check_price, no_to_yes, to_decimal

Event = Tuple[int, int, str, Any]              # (t, n, channel, message)
Parsed = Tuple[str, Any]                       # ("book", Book) | ("trade", Trade)


# Venues that stamp every frame themselves (Polymarket: the millisecond
# `timestamp` on each book, price_change and last_trade_price, and the book
# `hash`), so two byte-identical frames are one event delivered twice.
# Kalshi's book is a REST poll: an identical poll a second later is a real
# observation that the book went back, and must be applied again.
EXACT_REPEATS_ARE_COPIES = frozenset({"polymarket"})


def iter_market_events(directory: Path, channels: Optional[List[str]] = None, *,
                       warn: bool = False, drop_copies: bool = False,
                       dropped: Optional[Dict[str, int]] = None) -> Iterator[Event]:
    """Every archived message of a market, merged across channels on (t, n).

    Streams: `heapq.merge` over per-channel generators holds one record per
    channel in memory, and the reader underneath holds one decode window per
    open file, so a day replays in tens of MB however large the archive.

    `drop_copies` skips a frame whose archived bytes equal an earlier
    frame's in this directory, counting each skip in `dropped[channel]`
    when a dict is given. Why: the Polymarket recorder merges two sockets
    and drops identical frames over only the last 4,096. In a sample of
    archived 5-minute windows about a quarter held copies the lagging socket
    delivered 5-13 s after the first (about 4% of all frames), on every
    channel. A price_change carries absolute level sizes and a book is a
    whole snapshot, so a late copy re-creates levels already gone: on one
    15-minute window the rebuilt touch disagreed with the venue's own
    best_bid/best_ask on 9.6% of frames, 0.16% with copies dropped. Keyed on
    the bytes, never on (timestamp, hash): distinct frames share those. The
    cost is a 16-byte digest per distinct frame (tens of MB at peak on a
    busy window) and no measurable time.

    What this cannot fix: frames only the lagging socket delivered arrive
    late too, and are applied after newer state, so the rebuilt touch can
    still disagree with the venue's on a window with many copies. Filter
    crossed or locked touches before treating one as a price.
    """
    raw = Path(directory) / "raw"
    names = channels or list(channels_in(raw))

    def tagged(index: int, channel: str):
        # A named function, not a generator expression in the loop: the
        # expression would late-bind `channel` and label every stream with
        # the last one.
        if drop_copies:
            for t, n, m, frame in iter_directory(raw, channel, warn=warn, frames=True):
                yield t, n, index, channel, m, frame
        else:
            for t, n, m in iter_directory(raw, channel, warn=warn):
                yield t, n, index, channel, m, None

    seen = set()
    streams = [tagged(index, channel) for index, channel in enumerate(names)]
    for t, n, _, channel, message, frame in heapq.merge(*streams, key=lambda item: (item[0], item[1], item[2])):
        if frame is not None:
            digest = hashlib.blake2b(frame, digest_size=16).digest()
            if digest in seen:
                if dropped is not None:
                    dropped[channel] = dropped.get(channel, 0) + 1
                continue
            seen.add(digest)
        yield t, n, channel, message


class BookState:
    """A mutable YES-denominated book that snapshots and deltas apply to."""

    def __init__(self, market_id: str):
        self.market_id = market_id
        self.levels: Dict[Side, Dict[Decimal, Decimal]] = {Side.BUY: {}, Side.SELL: {}}
        self.ts_ms = 0
        self.ready = False

    def snapshot(self, book: Book) -> None:
        self.levels = {Side.BUY: {l.price: l.size for l in book.bids if l.size > ZERO},
                       Side.SELL: {l.price: l.size for l in book.asks if l.size > ZERO}}
        self.ts_ms = book.ts_ms
        self.ready = True

    def set_level(self, side: Side, price: Decimal, size: Decimal, ts_ms: int) -> None:
        price = check_price(price)
        if size <= ZERO:
            self.levels[side].pop(price, None)
        else:
            self.levels[side][price] = size
        self.ts_ms = ts_ms

    def add_delta(self, side: Side, price: Decimal, delta: Decimal, ts_ms: int) -> None:
        price = check_price(price)
        self.set_level(side, price, self.levels[side].get(price, ZERO) + delta, ts_ms)

    # The touch, read straight off the level dicts. A replay applies ~100k
    # deltas to a 100-level book per five-minute window; sorting the whole
    # book into a `Book` on each would be the whole cost of a replay, so
    # parsers yield this live state and callers that need an immutable
    # snapshot ask for `to_book()`. Callers must copy what they keep: the
    # same object is mutated by the next event.

    @property
    def best_bid(self) -> Optional[Decimal]:
        bids = self.levels[Side.BUY]
        return max(bids) if bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        asks = self.levels[Side.SELL]
        return min(asks) if asks else None

    @property
    def mid(self) -> Optional[Decimal]:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else (bid + ask) / 2

    def is_crossed(self) -> bool:
        bid, ask = self.best_bid, self.best_ask
        return bid is not None and ask is not None and bid >= ask

    def size_at(self, side: Side, price) -> Decimal:
        return self.levels[side].get(check_price(price), ZERO)

    def to_book(self) -> Book:
        return Book(market_id=self.market_id,
                    bids=[Level(p, s) for p, s in self.levels[Side.BUY].items()],
                    asks=[Level(p, s) for p, s in self.levels[Side.SELL].items()],
                    ts_ms=self.ts_ms)


# ---- venue message -> events -------------------------------------------------

# What makes a `last_trade_price` one print: the fill's transaction and
# everything the fill model reads. Two frames that agree on all of it are one
# print archived twice (about 4% of 5-minute prints in a sample), whatever
# else differs in their bytes. Two fills that agree on
# all of it inside one transaction would be one print; the recorder's own
# de-duplication already merges those (see `Polymarket.stream`).
PRINT_KEY = ("transaction_hash", "asset_id", "price", "size", "side", "timestamp")


def polymarket_parser(contract: Contract, *,
                      dropped: Optional[Dict[str, int]] = None) -> Callable[[BookState, int, str, Any], List[Parsed]]:
    """One window's parser. It remembers the prints it has emitted, so a
    repeat (`PRINT_KEY`) yields nothing and is counted in `dropped`."""
    from predkit.venues.polymarket import parse_book, parse_trade_event

    yes, no = contract.yes_token, contract.no_token
    prints = set()

    def parse(state: BookState, t: int, channel: str, message: Any) -> List[Parsed]:
        if not isinstance(message, dict):
            return []
        asset = message.get("asset_id")
        if channel == "book":
            if asset == yes or asset is None:
                state.snapshot(parse_book(message, contract.market_id, is_yes_token=True))
                return [("book", state)]
            return []            # the NO book is archived; the YES book is the one we quote
        if channel == "price_change":
            # Wire shape (read live 2026-09-13): {"market", "price_changes":
            # [{"asset_id", "price", "size", "side", ...}, ...], "timestamp"}
            # with both tokens' entries in one frame. `size` is the new size
            # at the level. Older frames carried `changes` and a top-level
            # `asset_id`; both are read.
            changes = message.get("price_changes") or message.get("changes") or [message]
            stamp = int(message.get("timestamp", t))
            for change in changes:
                change_asset = change.get("asset_id", asset)
                price = check_price(change["price"])
                side = Side(str(change["side"]).lower())
                size = to_decimal(change.get("size", "0"))
                if change_asset == no:
                    price, side = no_to_yes(price), (Side.SELL if side is Side.BUY else Side.BUY)
                elif change_asset != yes and yes is not None:
                    continue
                state.set_level(side, price, size, stamp)
            return [("book", state)] if state.ready else []
        if channel == "last_trade_price":
            key = tuple(message.get(field) for field in PRINT_KEY)
            if key in prints:
                if dropped is not None:
                    dropped[channel] = dropped.get(channel, 0) + 1
                return []
            trade = parse_trade_event(message, contract.market_id, yes)
            prints.add(key)
            return [("trade", trade)]
        return []

    return parse


def kalshi_parser(contract: Contract) -> Callable[[BookState, int, str, Any], List[Parsed]]:
    from predkit.venues.kalshi import parse_orderbook, parse_trade

    def parse(state: BookState, t: int, channel: str, message: Any) -> List[Parsed]:
        if not isinstance(message, dict):
            return []
        if channel in ("book_poll", "orderbook_snapshot"):
            state.snapshot(parse_orderbook(message, contract.market_id, ts_ms=t))
            return [("book", state)]
        if channel == "orderbook_delta":
            body = message.get("msg", message)
            price = to_decimal(body["price"]) / 100
            delta = to_decimal(body["delta"])
            if str(body.get("side", "yes")).lower() == "yes":
                state.add_delta(Side.BUY, price, delta, t)
            else:
                state.add_delta(Side.SELL, no_to_yes(price), delta, t)
            return [("book", state)] if state.ready else []
        if channel == "trade_poll":
            # A REST tape row, verbatim (dollar strings, created_time).
            return [("trade", parse_trade(message, contract.market_id))]
        if channel == "trade":
            body = message.get("msg", message)
            row = dict(body)
            row.setdefault("taker_side", "yes")
            row.setdefault("ts", t // 1000)
            return [("trade", parse_trade(row, contract.market_id))]
        return []

    return parse


def parser_for(contract: Contract, *, dropped: Optional[Dict[str, int]] = None):
    if contract.venue == "polymarket":
        return polymarket_parser(contract, dropped=dropped)
    if contract.venue == "kalshi":
        return kalshi_parser(contract)
    raise ValueError(f"no replay parser for venue {contract.venue!r}")


def rebuild(directory: Path, contract: Contract, *, warn: bool = False,
            dropped: Optional[Dict[str, int]] = None) -> Iterator[Tuple[int, Parsed]]:
    """(receive_ms, ("book", BookState) | ("trade", Trade)) in arrival order.
    The book event is the live, mutable state (see `BookState`).

    One directory is one window, and each venue event in it is applied
    once: on a venue in `EXACT_REPEATS_ARE_COPIES` a frame archived twice
    is skipped (`iter_market_events`), and so is a Polymarket print already
    emitted (`PRINT_KEY`). `dropped`, when given, counts both per channel."""
    parse = parser_for(contract, dropped=dropped)
    state = BookState(contract.market_id)
    for t, n, channel, message in iter_market_events(
            directory, warn=warn, drop_copies=contract.venue in EXACT_REPEATS_ARE_COPIES, dropped=dropped):
        try:
            events = parse(state, t, channel, message)
        except (KeyError, ValueError, TypeError):
            continue         # a malformed message costs itself, not the replay
        for event in events:
            yield t, event


# ---- maker fill model ---------------------------------------------------------

@dataclass
class MakerOrder:
    order_id: str
    side: Side
    price: Decimal
    size: Decimal
    placed_ms: int
    ttl_ms: int
    queue_ahead: Decimal = ZERO

    @property
    def expires_ms(self) -> int:
        return self.placed_ms + self.ttl_ms

    def improves(self, book: Book) -> bool:
        """Alone at its level: better than the touch and inside the spread."""
        if self.side is Side.BUY:
            return (book.best_bid is None or self.price > book.best_bid) and \
                   (book.best_ask is None or self.price < book.best_ask)
        return (book.best_ask is None or self.price < book.best_ask) and \
               (book.best_bid is None or self.price > book.best_bid)


@dataclass(frozen=True)
class SimFill:
    order_id: str
    side: Side
    price: Decimal
    size: Decimal
    ts_ms: int
    bound: str          # "optimistic" | "pessimistic"
    cause: str          # "print" | "crossed" | "cleared"


@dataclass
class MakerFillModel:
    fills: Dict[str, List[SimFill]] = field(default_factory=lambda: {"optimistic": [], "pessimistic": []})
    _live: Dict[str, Dict[str, MakerOrder]] = field(
        default_factory=lambda: {"optimistic": {}, "pessimistic": {}})
    _queue: Dict[str, Decimal] = field(default_factory=dict)
    expired: List[str] = field(default_factory=list)

    def place(self, order: MakerOrder, book: Optional[Book] = None) -> None:
        """Post an order. `queue_ahead` defaults to the size already resting
        at that price on `book`, which is zero when the order improves it."""
        queue = order.queue_ahead
        if book is not None and queue == ZERO:
            queue = book.size_at(order.side, order.price)
        self._queue[order.order_id] = queue
        for bound in self._live:
            self._live[bound][order.order_id] = order

    def cancel(self, order_id: str) -> None:
        for bound in self._live:
            self._live[bound].pop(order_id, None)
        self._queue.pop(order_id, None)

    def open(self, bound: str = "pessimistic") -> List[MakerOrder]:
        return list(self._live[bound].values())

    def _expire(self, now_ms: int) -> None:
        for order_id, order in list(self._live["optimistic"].items()):
            if now_ms >= order.expires_ms:
                self.cancel(order_id)
                self.expired.append(order_id)

    def on_book(self, book: Book) -> None:
        self._expire(book.ts_ms)
        for bound, live in self._live.items():
            for order_id, order in list(live.items()):
                crossed = (order.side is Side.BUY and book.best_ask is not None
                           and book.best_ask <= order.price) or \
                          (order.side is Side.SELL and book.best_bid is not None
                           and book.best_bid >= order.price)
                if crossed:
                    self._fill(bound, order, order.price, order.size, book.ts_ms, "crossed")

    def on_trade(self, trade: Trade) -> None:
        self._expire(trade.ts_ms)
        for bound, live in self._live.items():
            for order_id, order in list(live.items()):
                if trade.ts_ms < order.placed_ms:
                    continue
                if order.side is Side.BUY:
                    qualifies = trade.aggressor is Side.SELL and trade.price <= order.price
                    beyond = trade.price < order.price
                else:
                    qualifies = trade.aggressor is Side.BUY and trade.price >= order.price
                    beyond = trade.price > order.price
                if not qualifies:
                    continue
                if bound == "optimistic":
                    self._fill(bound, order, order.price, min(order.size, trade.size),
                               trade.ts_ms, "print")
                    continue
                if beyond:
                    self._fill(bound, order, order.price, order.size, trade.ts_ms, "cleared")
                    continue
                ahead = self._queue.get(order_id, ZERO)
                if trade.size <= ahead:
                    self._queue[order_id] = ahead - trade.size
                    continue
                available = trade.size - ahead
                self._queue[order_id] = ZERO
                self._fill(bound, order, order.price, min(order.size, available), trade.ts_ms, "print")

    def _fill(self, bound: str, order: MakerOrder, price: Decimal, size: Decimal,
              ts_ms: int, cause: str) -> None:
        if size <= ZERO:
            return
        self.fills[bound].append(SimFill(order.order_id, order.side, price, size, ts_ms, bound, cause))
        remaining = order.size - size
        live = self._live[bound]
        if remaining <= ZERO:
            live.pop(order.order_id, None)
        else:
            live[order.order_id] = MakerOrder(order.order_id, order.side, order.price, remaining,
                                              order.placed_ms, order.ttl_ms)


__all__ = ["BookState", "EXACT_REPEATS_ARE_COPIES", "MakerFillModel", "MakerOrder", "PRINT_KEY", "SimFill",
           "iter_market_events", "kalshi_parser", "parser_for", "polymarket_parser", "rebuild"]


def main(argv: Optional[List[str]] = None) -> int:
    """Print the rebuilt touch of one recorded market, sampled every `--every` seconds.

        python -m predkit.replay data/polymarket/<conditionId>
    """
    import argparse
    from datetime import datetime, timezone

    from predkit.record_series import read_contract

    parser = argparse.ArgumentParser(description="Replay one recorded market's archive and print its touch.")
    parser.add_argument("directory", help="a market directory holding raw/ and contract.json")
    parser.add_argument("--every", type=float, default=10.0, help="seconds between printed samples")
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    if not (directory / "contract.json").exists():
        raise SystemExit(f"{directory} has no contract.json: record it with predkit.record or "
                         "predkit.record_series, which write one beside the archive")
    contract = read_contract(directory)
    dropped: Dict[str, int] = {}
    books = trades = 0
    next_print = 0
    for t, (kind, event) in rebuild(directory, contract, dropped=dropped):
        if kind == "trade":
            trades += 1
            continue
        books += 1
        if t >= next_print and event.best_bid is not None and event.best_ask is not None:
            stamp = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp}Z  bid {event.best_bid}  ask {event.best_ask}"
                  f"{'  CROSSED' if event.is_crossed() else ''}")
            next_print = t + int(args.every * 1000)
    print(f"{books} book events, {trades} trades, copies dropped {dict(dropped) or 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
