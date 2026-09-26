"""Record a market: archive every message first, parse second, never go quiet.

    python -m predkit.record --venue kalshi --markets KXBTC15M-26SEP1218
    python -m predkit.record --venue polymarket --markets <condition_id> --reference binance:BTCUSDT
    python -m predkit.record --reference binance:BTCUSDT            # a reference feed, alone

Layout, and why it is enforced here
-----------------------------------
One directory per venue per market:

    data/<venue>/<market_id>/raw/<day>/<channel>-<HH>.jsonl.gz
    data/_reference/<name>/raw/...          a Binance book ticker, say

Rows for two markets are structurally identical, so directory layout is the
only thing keeping them apart; `market_dir` is the one place that knows the
path, and nothing else builds one. Reference feeds live under `_reference`
so no market tool can mistake one for a venue book.

What a recorder does, per message
---------------------------------
1. `raw_log.write(channel, message)`. Before anything else.
2. Call `on_message`, if any, and COUNT its failures. A parse error is a bug
   to fix later, over the archive; it must not stop the archive.

What it does when nothing arrives
---------------------------------
Every message is awaited with a deadline (`supervise.each_with_deadline`).
A silent socket is treated exactly like a dropped one: close, back off,
reconnect, and count the stall where `stats()` shows it. The whole loop runs
under `supervise` so no exception ends the process.

Recording needs no keys on either venue. Kalshi's websocket does need a
signed connection even for public channels (their API requires the access
key headers on the handshake as of 2026-09-12); the Kalshi adapter takes a
signer for that and falls back to REST polling of the public book when it
has none.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

from predkit.rawlog import RawEventLog
from predkit.supervise import FeedStalled, each_with_deadline, log, supervise

Message = Tuple[str, Any]           # (channel, verbatim message)
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class Streamer(Protocol):
    """What a venue adapter (or reference feed) provides to be recorded."""

    name: str

    def stream(self, market_id: str) -> AsyncIterator[Message]: ...


def market_dir(data_dir: Path, venue: str, market_id: str) -> Path:
    """The one place that knows where a market's data lives."""
    safe_venue = _SAFE.sub("_", venue)
    safe_market = _SAFE.sub("_", market_id)
    if not safe_venue or not safe_market:
        raise ValueError(f"cannot build a directory for venue={venue!r} market={market_id!r}")
    return Path(data_dir) / safe_venue / safe_market


def reference_dir(data_dir: Path, name: str) -> Path:
    return market_dir(data_dir, "_reference", name)


class MarketRecorder:
    def __init__(self, source: Streamer, market_id: str, directory: Path, *,
                 stall_timeout_s: float = 30.0,
                 on_message: Optional[Callable[[str, Any], None]] = None,
                 raw_log: Optional[RawEventLog] = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 max_backoff_s: float = 30.0):
        self.source = source
        self.market_id = market_id
        self.directory = Path(directory)
        self.stall_timeout_s = stall_timeout_s
        self.on_message = on_message
        self.raw_log = raw_log or RawEventLog(self.directory)
        self._sleep = sleep
        self.max_backoff_s = max_backoff_s

        self.messages = 0
        self.parse_failures = 0
        self.stalls = 0
        self.reconnects = 0
        self.last_message_at: Optional[float] = None
        self.stop = asyncio.Event()

    async def run(self) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            began = time.monotonic()
            try:
                await self._session()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except FeedStalled as exc:
                self.stalls += 1
                log(f"[{self.source.name}:{self.market_id}] stalled: {exc}; reconnecting")
            except Exception as exc:
                log(f"[{self.source.name}:{self.market_id}] feed error: "
                    f"{type(exc).__name__}: {exc}; reconnecting")
            finally:
                self.raw_log.flush()
            if self.stop.is_set():
                break
            self.reconnects += 1
            if time.monotonic() - began > 60:
                backoff = 1.0
            await self._sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)

    async def _session(self) -> None:
        async for channel, message in each_with_deadline(
                self.source.stream(self.market_id), self.stall_timeout_s):
            self.raw_log.write(channel, message)        # archive first
            self.messages += 1
            self.last_message_at = time.time()
            if self.on_message is not None:             # parse second
                try:
                    self.on_message(channel, message)
                except Exception as exc:
                    self.parse_failures += 1
                    if self.parse_failures <= 5:
                        log(f"[{self.source.name}:{self.market_id}] parse failure "
                            f"#{self.parse_failures}: {type(exc).__name__}: {exc}")
            if self.stop.is_set():
                return

    def close(self) -> None:
        self.stop.set()
        self.raw_log.close()

    def stats(self) -> Dict[str, Any]:
        return {
            "venue": self.source.name, "market": self.market_id,
            "messages": self.messages, "parseFailures": self.parse_failures,
            "stalls": self.stalls, "reconnects": self.reconnects,
            "silenceSeconds": (None if self.last_message_at is None
                               else round(time.time() - self.last_message_at, 1)),
            "raw": self.raw_log.stats(),
        }


async def record_all(recorders: List[MarketRecorder]) -> None:
    """Run every recorder under supervision. None may end the process."""
    tasks = [asyncio.create_task(supervise(f"{r.source.name}:{r.market_id}", r.run))
             for r in recorders]
    try:
        await asyncio.gather(*tasks)
    finally:
        for recorder in recorders:
            recorder.close()


def _build_source(venue: str):
    if venue == "polymarket":
        from predkit.venues.polymarket import Polymarket
        return Polymarket()
    if venue == "kalshi":
        from predkit.venues.kalshi import Kalshi
        return Kalshi()
    raise SystemExit(f"unknown venue {venue!r}; known: polymarket, kalshi")


def _build_reference(spec: str):
    kind, _, symbol = spec.partition(":")
    if kind == "binance" and symbol:
        from predkit.venues.binance_reference import BinanceBookTicker
        return BinanceBookTicker(), symbol
    raise SystemExit(f"unknown reference {spec!r}; known: binance:<SYMBOL>")


def _write_contracts(source: Any, recorders: List[MarketRecorder]) -> None:
    """Save each market's `contract.json` beside its archive: replay needs
    the token ids and the resolution time. A failed lookup is logged, not
    fatal; the archive matters more than the metadata."""
    from predkit.record_series import write_contract

    for recorder in recorders:
        try:
            write_contract(recorder.directory, source.market(recorder.market_id))
        except Exception as exc:
            log(f"[{source.name}:{recorder.market_id}] contract lookup failed: "
                f"{type(exc).__name__}: {exc}; replay will need a contract.json")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--venue", choices=("polymarket", "kalshi"),
                        help="omit to record only --reference feeds")
    parser.add_argument("--markets", default="", help="comma-separated market ids (needs --venue)")
    parser.add_argument("--reference", action="append", default=[],
                        help="binance:BTCUSDT; may repeat")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--stall-timeout", type=float, default=None,
                        help="message deadline in seconds; default the source's own, else 30 s")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    recorders: List[MarketRecorder] = []
    if args.venue:
        source = _build_source(args.venue)
        stall = args.stall_timeout or getattr(source, "stall_timeout_s", 30.0)
        recorders = [MarketRecorder(source, m.strip(), market_dir(data_dir, args.venue, m.strip()),
                                    stall_timeout_s=stall)
                     for m in args.markets.split(",") if m.strip()]
        _write_contracts(source, recorders)
    elif args.markets.strip():
        raise SystemExit("--markets needs --venue")
    for spec in args.reference:
        feed, symbol = _build_reference(spec)
        recorders.append(MarketRecorder(feed, symbol, reference_dir(data_dir, f"{feed.name}-{symbol}"),
                                        stall_timeout_s=args.stall_timeout
                                        or getattr(feed, "stall_timeout_s", 30.0)))
    if not recorders:
        raise SystemExit("nothing to record: give --venue with --markets, or --reference")
    for recorder in recorders:
        log(f"recording {recorder.source.name}:{recorder.market_id} -> {recorder.directory}")
    try:
        asyncio.run(record_all(recorders))
    except KeyboardInterrupt:
        pass
    for recorder in recorders:
        log(str(recorder.stats()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
