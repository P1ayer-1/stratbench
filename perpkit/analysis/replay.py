"""Rebuild a feature file from archived raw events.

    python -m perpkit.analysis.replay --date 2026-01-01
    python -m perpkit.analysis.replay --date 2026-01-01 --sample-ms 1000 \
        --horizons 300,900,1800 --out data/replayed

This is what makes the raw archive worth keeping. Change a feature, add a new
one, or relabel at a different horizon, then replay every hour you have ever
recorded and get a fresh dataset — without waiting days to collect it again.

It reuses the exact same `OrderBook`, `TradeTape`, `FeatureEngine` and
`FeatureRecorder` as the live path. That is deliberate: if replay used a
separate implementation, a model trained on replayed data would be trained on
subtly different features than the live bot computes, and the discrepancy
would be nearly impossible to find later.

Because the code is shared, replay also reproduces the desyncs: if the book hit
a sequence gap live, it hits the same gap here and the same rows go missing.

Two clocks
----------
Every market-data feature is already on the exchange's clock: the engine
stamps a snapshot with the book's own `ts`, the tape trims on each fill's
`ts`, and labels are measured from those. What reads a wall clock is
`received_ts`, `book_age_ms` and `tape_staleness_s`. Filling those from the
replaying machine's `time.time()` would describe nothing, so `--clock`
decides what "now" is for them:

  receive   (default) the archived receive time `t`: the columns come out as
            the live recorder computed them, delay included. A live
            recorder's sockets have been measured running up to 56 s behind
            BloFin's stamps in bursts, upstream of the process, so these
            columns describe how stale the recorder's book was, not the
            market.
  exchange  the venue's clock plus `--lag-ms`, restamped as `merged_events`
            describes: what a host with a clean feed would have computed.

Row order and every other column are identical under both, because both keep
the socket's arrival order - which is the venue's send order.
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple


from perpkit.layout import resolve_raw_dir
from perpkit.features import FeatureEngine
from perpkit.orderbook import OrderBook
from perpkit.rawlog import iter_events
from perpkit.recorder import FeatureRecorder, LabelConfig
from perpkit.tape import TradeTape

CHANNELS = ("books", "books5", "trades", "funding-rate")

# `perpkit/ingest.py` subscribes all of these on ONE websocket per instrument,
# so the archive holds them in the order that socket delivered them - TCP and
# the SDK's queue are both first in, first out - and `n` is that order.
# Everything else in a day directory (`open-interest`, `mark-price`) is polled
# over REST by another process or thread, whose `n` is unrelated.
SOCKET_CHANNELS = frozenset(CHANNELS)

CLOCKS = ("receive", "exchange")

Event = Tuple[int, int, dict]


def channel_of(path: Path) -> str:
    """`books-14.jsonl.gz` and `books-14.r001.jsonl.gz` -> `books`."""
    return path.name.split(".", 1)[0].rsplit("-", 1)[0]


def exchange_ms(message: dict) -> Optional[int]:
    """The venue's own stamp on an archived BloFin message, in ms, or None.

    `books`/`books5`: `data.ts`, when the book was generated. `trades` and
    `mark-price`: the newest row's `ts`, since a push cannot leave before the
    newest fill in it. `funding-rate` carries no stamp. `open-interest`'s `ts`
    is the minute bucket the value belongs to (always :00.000, received ~24 s
    later), not when it was read, so it is not an event time and is None here.
    """
    if not isinstance(message, dict):
        return None
    channel = (message.get("arg") or {}).get("channel")
    data = message.get("data")
    try:
        if channel in ("books", "books5"):
            return int(data["ts"])
        if channel in ("trades", "mark-price"):
            return max(int(row["ts"]) for row in data)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def on_exchange_clock(events: Iterable[Event], lag_ms: int = 0) -> Iterator[Event]:
    """Restamp ONE socket's events, given in arrival order, onto the venue's clock.

    Each event gets the newest exchange stamp seen so far on the socket, plus
    `lag_ms` of feed latency. Not its own stamp: arrival order is the venue's
    send order, and BloFin has been seen sending a book stamped up to 15 s
    older than the trade in front of it. No host could have had that book at
    its own `ts`, so the running maximum keeps that part of the delay and
    removes only what came after the venue stamped the newest message - the
    gateway, the link, this process. It also keeps the stream in
    arrival order with non-decreasing stamps, so replay applies events exactly
    as live did.

    Events before the first stamp (a `funding-rate` push opening the day) take
    the first stamp; a stream with no stamp at all keeps its receive times.

    What it cannot see is a delay common to every channel of the socket:
    stamps that are all equally old look the same whether the venue, its
    gateway or the recorder held them. In the lag bursts measured so far that
    common part was almost all of it, and it was attached to the connection
    rather than the instrument, which is why a clean host sees the market at
    about `ts + lag`. Still, this clock is a lower bound on when the data
    could be known, and the receive clock is what the recorder actually knew.
    """
    freshest: Optional[int] = None
    held: List[Event] = []
    for received, sequence, message in events:
        stamp = exchange_ms(message)
        if stamp is not None and (freshest is None or stamp > freshest):
            freshest = stamp
        if freshest is None:
            held.append((received, sequence, message))
            continue
        for _, held_sequence, held_message in held:
            yield freshest + lag_ms, held_sequence, held_message
        held.clear()
        yield freshest + lag_ms, sequence, message
    yield from held


def merged_events(
    raw_dir: Path,
    date: Optional[str],
    *,
    clock: str = "receive",
    lag_ms: int = 0,
    channels: Optional[Iterable[str]] = None,
    quiet: bool = False,
) -> Iterator[Event]:
    """All archived events for a date (or everything), in exact arrival order.

    Files are per-channel, so books and trades must be interleaved back into a
    single stream — otherwise every trade would be applied against a book from
    the wrong moment. `heapq.merge` does this lazily, so memory stays flat
    regardless of archive size.

    The sort key is `(receive_ms, sequence)`, not the timestamp alone. Book
    updates and trades regularly share a millisecond, and ordering them wrongly
    changes the computed features — a trade applied before rather than after a
    book update lands in a different feature snapshot.

    `clock="exchange"` yields `(stamp, sequence, message)` instead, where the
    socket channels are merged exactly as above and then restamped by
    `on_exchange_clock` (+`lag_ms`), and polled channels keep their own `t`:
    their writer is another process or a worker thread, stamped one REST
    round trip after the venue answered, and never waited in the socket's
    queue. The two are merged on the stamp alone. Use this clock to join the
    archive against anything stamped at exchange time (Binance aggTrades,
    another host's log); use `receive` to reproduce what the recorder saw.

    `channels` limits which files are read, by channel name.
    """
    if clock not in CLOCKS:
        raise ValueError(f"clock must be one of {CLOCKS}, not {clock!r}")
    pattern = f"{date}/*.jsonl.gz" if date else "*/*.jsonl.gz"
    files = sorted(raw_dir.glob(pattern))
    if channels is not None:
        wanted = set(channels)
        files = [path for path in files if channel_of(path) in wanted]
    if not files:
        raise SystemExit(f"No raw logs matching {pattern} under {raw_dir}")

    if not quiet:
        print(f"Replaying {len(files)} file(s):")
        total = 0
        for path in files:
            size = path.stat().st_size
            total += size
            print(f"  {path.relative_to(raw_dir)}  ({size / 1e6:.1f} MB compressed)")
        print(f"  total {total / 1e6:.1f} MB\n")

    by_arrival = lambda item: (item[0], item[1])  # noqa: E731
    if clock == "receive":
        return heapq.merge(*(iter_events(path) for path in files), key=by_arrival)

    socket = [path for path in files if channel_of(path) in SOCKET_CHANNELS]
    polled = [path for path in files if channel_of(path) not in SOCKET_CHANNELS]
    stamped = on_exchange_clock(
        heapq.merge(*(iter_events(path) for path in socket), key=by_arrival),
        lag_ms,
    )
    return heapq.merge(stamped, *(iter_events(path) for path in polled),
                       key=lambda item: item[0])


def replay(
    raw_dir: Path,
    out_dir: Path,
    *,
    date: Optional[str],
    sample_ms: int,
    horizons: Tuple[float, ...],
    threshold_bps: float,
    clock: str = "receive",
    lag_ms: int = 0,
) -> dict:
    book = OrderBook()
    tape = TradeTape()
    engine = FeatureEngine()
    recorder = FeatureRecorder(
        out_dir,
        label_config=LabelConfig(horizons_seconds=horizons, threshold_bps=threshold_bps),
        sample_interval_ms=sample_ms,
    )

    counts = {"books": 0, "trades": 0, "funding": 0, "other": 0, "desyncs": 0}

    try:
        for now_ms, _, message in merged_events(raw_dir, date, clock=clock,
                                                lag_ms=lag_ms):
            if not isinstance(message, dict):
                continue
            channel = message.get("arg", {}).get("channel", "")

            if channel in ("books", "books5"):
                book.apply(message)
                counts["books"] += 1
                if not book.ready:
                    # Same desync behaviour as live. Live would reconnect and
                    # get a snapshot; here we simply wait for the next
                    # archived snapshot to arrive.
                    counts["desyncs"] += 1
                    continue
                if book.is_crossed():
                    continue
                engine.on_book_event(book)
                recorder.observe(engine.compute(book, tape, now_ms=now_ms))
            elif channel == "trades":
                counts["trades"] += 1
                if tape.add_message(message.get("data")):
                    recorder.observe(engine.compute(book, tape, now_ms=now_ms))
            elif channel == "funding-rate":
                counts["funding"] += 1
                engine.on_funding(message.get("data"))
            else:
                counts["other"] += 1
    finally:
        recorder.close()

    counts.update(recorder.stats())
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instrument", default=None,
                        help="Instrument id, e.g. ADA-USDT. Omit when only "
                             "one instrument has been recorded.")
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data",
                        help="Recording root; the archive is resolved beneath "
                             "it as <data-dir>/<INST-ID>/raw.")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Archive directory outright, bypassing "
                             "--instrument resolution.")
    parser.add_argument("--out", type=Path, default=repo_root / "data" / "replayed")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; omit for all.")
    parser.add_argument("--sample-ms", type=int, default=1000)
    parser.add_argument("--horizons", default="300,900,1800",
                        help="Comma-separated forward horizons in seconds.")
    parser.add_argument("--threshold-bps", type=float, default=10.0)
    parser.add_argument("--clock", choices=CLOCKS, default="receive",
                        help="what 'now' is for received_ts, book_age_ms and "
                             "tape_staleness_s: the archived receive time (as "
                             "live computed them) or the venue's clock + --lag-ms")
    parser.add_argument("--lag-ms", type=int, default=15,
                        help="feed latency added with --clock exchange "
                             "(a host near the venue measured BloFin p50 11 ms)")
    args = parser.parse_args(argv)

    horizons = tuple(float(part) for part in args.horizons.split(",") if part.strip())

    raw_dir = args.raw_dir or resolve_raw_dir(args.data_dir, args.instrument)
    # Replayed features inherit the instrument of the archive they came from,
    # and nothing in the CSV records which that was -- so keep them apart the
    # same way the recorder does, rather than overwriting one instrument's
    # replay with the next.
    out = args.out
    if args.raw_dir is None and raw_dir.parent != args.data_dir:
        out = out / raw_dir.parent.name
    print(f"Replaying {raw_dir} -> {out}")

    result = replay(
        raw_dir, out,
        date=args.date, sample_ms=args.sample_ms,
        horizons=horizons, threshold_bps=args.threshold_bps,
        clock=args.clock, lag_ms=args.lag_ms,
    )

    print("Replay complete:")
    print(f"  book messages   {result['books']:,}")
    print(f"  trade messages  {result['trades']:,}")
    print(f"  funding         {result['funding']:,}")
    print(f"  desync events   {result['desyncs']:,}")
    print(f"  rows written    {result['rowsWritten']:,}")
    print(f"  rows dropped    {result['rowsDropped']:,} (no observable future)")
    print(f"\nOutput: {out}")
    print("Run the predictiveness check against it with --data-dir", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
