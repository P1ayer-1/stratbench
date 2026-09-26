"""Record forced-liquidation orders: Binance USDT-M `!forceOrder@arr`, or
Bybit v5 linear `allLiquidation.<symbol>`.

    python -m perpkit.record_liquidations --check                  # connect, print 3 messages, record nothing
    python -m perpkit.record_liquidations --list-cost              # 60 s sample -> bytes/day, records nothing
    python -m perpkit.record_liquidations                          # Binance, venue-wide
    python -m perpkit.record_liquidations --venue bybit --symbols BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,SUIUSDT

A separate venue is a separate process with its own lock, directory and
logs; starting or stopping this touches no other recorder. No keys.

What lands where
----------------
    data/binance/forceOrder/raw/<day>/forceOrder-<HH>.jsonl.gz        irreplaceable
    data/bybit/allLiquidation/raw/<day>/allLiquidation-<HH>.jsonl.gz  irreplaceable
    data/<venue>/<channel>/.recorder.lock                              this pid
    data/<venue>/<channel>/record.{out,err}.log                        the launcher's redirect

Venue-wide on purpose - see `perpkit/layout.venue_channel_dir` - and never
under `data/<INST-ID>/` or `data/hyperliquid/`.

Read `perpkit/liquidation_stream.py` before trusting the numbers: Binance
documents its stream as at most one liquidation order per symbol per 1000 ms,
so it undercounts in cascades; Bybit documents every liquidation in 500 ms
batches. Neither claim has been measured here yet.

Reconnects are routine (Binance cuts every connection at 24 h). Each one is
logged and the next hour-file gets a `.rNNN` suffix; nothing is ever appended.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional


from perpkit.layout import venue_channel_dir
from perpkit.config import DATA_DIR, RECORD_RAW
from perpkit.supervise import log, supervise
from perpkit.hyperliquid import ExclusiveLock
from perpkit.liquidation_stream import (
    DEFAULT_BYBIT_SYMBOLS,
    DEFAULT_PING_SECONDS,
    DEFAULT_STALL_TIMEOUT_S,
    CostSample,
    LiquidationFeed,
    format_cost,
    venue_spec,
)

# The token a live recorder's command line must contain for its lock to count
# as held. Matches `python -m perpkit.record_liquidations` and a script path
# alike. The lock file itself is per venue directory, so the venue need not be
# part of the token (and a default --venue is not on the command line at all).
LOCK_HOLDER = "record_liquidations"

CHECK_MESSAGES = 3
CHECK_MAX_SECONDS = 180.0     # a calm market can go minutes without a liquidation
SAMPLE_SECONDS = 60.0
STATUS_EVERY_S = 600.0


def parse_symbols(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


async def status_loop(feed: LiquidationFeed, every: float) -> None:
    while True:
        await asyncio.sleep(every)
        log(feed.summary_line() + f"; {feed.log.disk_bytes():,} bytes on disk")


async def record(feed: LiquidationFeed) -> None:
    tasks = [supervise(f"{feed.spec.venue}:{feed.spec.channel}", feed.run),
             supervise(f"{feed.spec.venue}:status",
                       lambda: status_loop(feed, STATUS_EVERY_S))]
    try:
        await asyncio.gather(*tasks)
    finally:
        feed.close()
        log(f"[{feed.spec.venue}] writer flushed and closed.")


def channel_root(spec, data_dir: Path) -> Path:
    return venue_channel_dir(data_dir, spec.venue, spec.channel)


def check(spec, *, data_dir: Path, ping_seconds: float, stall_seconds: float) -> int:
    """Connect, show the acks and the first messages, record nothing."""
    seen: List[dict] = []

    def show(message: dict) -> None:
        seen.append(message)
        print(f"  message {len(seen)}: {json.dumps(message, separators=(',', ':'))}")

    def show_control(message: dict) -> None:
        print(f"  control: {json.dumps(message, separators=(',', ':'))}")

    feed = LiquidationFeed(spec, root=channel_root(spec, data_dir), record_raw=False,
                           ping_seconds=ping_seconds, stall_timeout_s=stall_seconds,
                           stop_after_messages=CHECK_MESSAGES,
                           stop_after_seconds=CHECK_MAX_SECONDS,
                           on_message=show, on_control=show_control, on_log=log)
    print(f"--check: {spec.describe()}\n  {spec.url}\n  waiting for "
          f"{CHECK_MESSAGES} messages, up to {CHECK_MAX_SECONDS:g}s; nothing "
          f"is written.")
    asyncio.run(feed.run())
    print(f"  {len(seen)} message(s) in {feed.elapsed_s or 0:.1f}s; "
          f"{feed.control_messages} control, {feed.unrouted} unrouted, "
          f"{feed.reconnects} reconnect(s), {feed.errors} error(s)"
          + (f"; last error: {feed.last_error}" if feed.last_error else ""))
    if feed.errors or feed.connections == 0:
        return 1
    if not seen and spec.silence_is_a_fault:
        print("  FAIL: connected and acknowledged, but no order in the window. "
              "On this venue that is\n  a connection that will never deliver "
              "(the unrouted /ws/ path has been seen to answer every\n  "
              "LIST_SUBSCRIPTIONS and send nothing), not a calm market.")
        return 1
    if not seen:
        print("  Connected and acknowledged, but no liquidation printed in the "
              "window. Not a failure\n  on this venue: five symbols are quiet "
              "for minutes at a time.")
    return 0


def list_cost(spec, *, data_dir: Path, ping_seconds: float, stall_seconds: float) -> int:
    sample = CostSample()
    feed = LiquidationFeed(spec, root=channel_root(spec, data_dir), record_raw=False,
                           ping_seconds=ping_seconds, stall_timeout_s=stall_seconds,
                           stop_after_seconds=SAMPLE_SECONDS,
                           on_message=sample.add, on_log=lambda m: None)
    asyncio.run(feed.run())
    if feed.connections == 0 or not feed.elapsed_s:
        raise SystemExit(f"Could not connect to {spec.url}: {feed.last_error}")
    print(format_cost(sample.report(feed.elapsed_s)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--venue", choices=("binance", "bybit"), default="binance")
    parser.add_argument(
        "--symbols", default=None,
        help=f"Bybit only, comma-separated (default: "
             f"{','.join(DEFAULT_BYBIT_SYMBOLS)}). Binance's stream is venue-wide.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--ping-seconds", type=float, default=DEFAULT_PING_SECONDS,
                        help="How often to ask the venue for a reply (Bybit "
                             "documents 20 s).")
    parser.add_argument("--stall-seconds", type=float, default=DEFAULT_STALL_TIMEOUT_S,
                        help="Reconnect after this long without ANY message, "
                             "replies included.")
    parser.add_argument("--check", action="store_true",
                        help=f"Connect, print the first {CHECK_MESSAGES} "
                             f"messages and exit without recording.")
    parser.add_argument("--list-cost", action="store_true",
                        help=f"Sample {SAMPLE_SECONDS:g}s, print the disk "
                             f"arithmetic, record nothing.")
    args = parser.parse_args(argv)

    if args.check and args.list_cost:
        raise SystemExit("--check and --list-cost each connect once and "
                         "exit; run them one at a time.")
    spec = venue_spec(args.venue, parse_symbols(args.symbols))
    try:
        LiquidationFeed(spec, root=channel_root(spec, args.data_dir), record_raw=False,
                        ping_seconds=args.ping_seconds,
                        stall_timeout_s=args.stall_seconds)
    except ValueError as exc:
        raise SystemExit(f"Refusing to start: {exc}")

    if args.check:
        return check(spec, data_dir=args.data_dir, ping_seconds=args.ping_seconds,
                     stall_seconds=args.stall_seconds)
    if args.list_cost:
        return list_cost(spec, data_dir=args.data_dir, ping_seconds=args.ping_seconds,
                         stall_seconds=args.stall_seconds)
    if not RECORD_RAW:
        raise SystemExit("PERPKIT_RECORD_RAW is false and this recorder "
                         "produces nothing but the raw archive.\nNothing to do.")

    feed = LiquidationFeed(spec, root=channel_root(spec, args.data_dir), record_raw=True,
                           ping_seconds=args.ping_seconds,
                           stall_timeout_s=args.stall_seconds, on_log=log)
    lock = ExclusiveLock(feed.root / ".recorder.lock",
                         holder=LOCK_HOLDER).acquire()
    log(f"Recording {spec.describe()}")
    log(f"  {spec.url}")
    log(f"  -> {feed.log.root}  [{spec.channel}]  lock {lock.path}")
    log(f"  ping every {args.ping_seconds:g}s, stall after "
        f"{args.stall_seconds:g}s, status every {STATUS_EVERY_S:g}s")
    try:
        asyncio.run(record(feed))
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
