"""Record a rolling series: every window of a 5- or 15-minute market as it
opens, one directory per window, until the process is stopped.

    python -m predkit.record_series --series polymarket:btc-5m,polymarket:btc-15m,kalshi:KXBTC15M --reference binance:BTCUSDT

A 5-minute market is a new market id every five minutes, so a recorder
pointed at one id is idle within minutes. This one discovers windows:

  polymarket:btc-5m    Gamma slug `btc-updown-5m-<unix start aligned to 300>`
  polymarket:btc-15m   Gamma slug `btc-updown-15m-<aligned to 900>`
  kalshi:KXBTC15M      the series' open market, plus the next ticker computed
                       from the current close time in America/New_York
                       (`KXBTC15M-26SEP122215-15` closes 02:15Z, read live
                       2026-09-13); an unopened next window polls an empty
                       book until it opens, which costs one request a second

References: `binance:<SYMBOL>` (the exchange book ticker, a leader feed; it
is NOT what either venue settles on, see `backtest.label_row`).

Every `poll_s` the discoverer is asked for the current and the next
`ahead` windows. A window not yet recording gets its `contract.json`
written into its directory (replay needs the token ids and the resolution
time) and a `MarketRecorder` under `supervise`. A window past its
resolution by `grace_s` is stopped and closed. The reference feed records
for the whole run under `data/_reference/`.

Archive first, parse second, deadline on every message, one directory per
market: all inherited from `record.py`. Nothing here needs a key; Kalshi
without one polls the public book (see `venues/kalshi.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple
from zoneinfo import ZoneInfo

from predkit.record import MarketRecorder, market_dir, reference_dir
from predkit.schema import Contract
from predkit.supervise import log, supervise

ET = ZoneInfo("America/New_York")


class Discoverer(Protocol):
    venue: Any

    def discover(self, now_s: float, ahead: int) -> List[Contract]: ...


class PolymarketWindows:
    # A next-window subscription is silent for minutes before it opens, and
    # the library's ping keepalive already closes a dead socket within 20 s,
    # so the message deadline is a backstop, not the detector. At 30 s it
    # fired hundreds of times in a day of recording, every one on an idle window.
    stall_timeout_s = 300.0

    def __init__(self, venue: Any, prefix: str, window_s: int):
        self.venue = venue
        self.prefix = prefix
        self.window_s = window_s
        self._by_slug: Dict[str, Optional[Contract]] = {}

    def slug(self, start_s: int) -> str:
        return f"{self.prefix}-{start_s}"

    def discover(self, now_s: float, ahead: int) -> List[Contract]:
        aligned = int(now_s) - int(now_s) % self.window_s
        out = []
        for k in range(0, ahead + 1):
            slug = self.slug(aligned + k * self.window_s)
            if slug not in self._by_slug or self._by_slug[slug] is None:
                try:
                    self._by_slug[slug] = self.venue.market_by_slug(slug)
                except Exception as exc:
                    log(f"[{self.prefix}] lookup {slug} failed: {type(exc).__name__}: {exc}")
                    continue
            if self._by_slug[slug] is not None:
                out.append(self._by_slug[slug])
        # Forget slugs two windows behind so the cache does not grow forever.
        for slug in [s for s in self._by_slug if int(s.rsplit("-", 1)[1]) < aligned - 2 * self.window_s]:
            self._by_slug.pop(slug, None)
        return out


def kalshi_next_ticker(series: str, close_time: datetime, window_s: int) -> str:
    """`KXBTC15M-26SEP122215-15`: the close time of the NEXT window in
    America/New_York as YYMONDDHHMM, then its minute."""
    next_close = (close_time + timedelta(seconds=window_s)).astimezone(ET)
    stamp = next_close.strftime("%y%b%d%H%M").upper()
    return f"{series}-{stamp}-{next_close.strftime('%M')}"


class KalshiWindows:
    stall_timeout_s = 30.0          # the poller yields every second; 30 s of nothing is a hang

    def __init__(self, venue: Any, series: str, window_s: int):
        self.venue = venue
        self.series = series
        self.window_s = window_s

    def discover(self, now_s: float, ahead: int) -> List[Contract]:
        try:
            current = self.venue.list_markets(self.series, limit=5)
        except Exception as exc:
            log(f"[{self.series}] list failed: {type(exc).__name__}: {exc}")
            return []
        out = list(current)
        latest = max((c.resolves_at for c in current), default=None)
        for _ in range(ahead):
            if latest is None:
                break
            ticker = kalshi_next_ticker(self.series, latest, self.window_s)
            try:
                nxt = self.venue.market(ticker)
            except Exception:
                break                      # not listed yet
            out.append(nxt)
            latest = nxt.resolves_at
        return out


def write_contract(directory: Path, contract: Contract) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract.json"
    if path.exists():
        return
    row = asdict(contract)
    row["resolves_at"] = contract.resolves_at.isoformat()
    path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")


def read_contract(directory: Path) -> Contract:
    row = json.loads((Path(directory) / "contract.json").read_text(encoding="utf-8"))
    from decimal import Decimal
    row["resolves_at"] = datetime.fromisoformat(row["resolves_at"])
    for key in ("tick", "min_size"):
        row[key] = Decimal(row[key])
    row["price_ranges"] = tuple(tuple(Decimal(x) for x in r) for r in row.get("price_ranges", ()))
    return Contract(**row)


class SeriesRecorder:
    def __init__(self, name: str, discoverer: Discoverer, data_dir: Path, *, ahead: int = 1,
                 poll_s: float = 20.0, grace_s: float = 120.0, stall_timeout_s: Optional[float] = None,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.name = name
        self.discoverer = discoverer
        self.data_dir = Path(data_dir)
        self.ahead = ahead
        self.poll_s = poll_s
        self.grace_s = grace_s
        # None: the discoverer's own default, which knows how quiet its feed is.
        self.stall_timeout_s = stall_timeout_s if stall_timeout_s is not None \
            else getattr(discoverer, "stall_timeout_s", 30.0)
        self.clock = clock
        self.sleep = sleep
        self.active: Dict[str, Tuple[Contract, MarketRecorder, asyncio.Task]] = {}
        self.started: List[str] = []
        self.finished: List[str] = []
        self.stop = asyncio.Event()

    def _start(self, contract: Contract) -> None:
        directory = market_dir(self.data_dir, contract.venue, contract.market_id)
        write_contract(directory, contract)
        recorder = MarketRecorder(self.discoverer.venue, contract.market_id, directory,
                                  stall_timeout_s=self.stall_timeout_s)
        task = asyncio.create_task(supervise(f"{contract.venue}:{contract.market_id}", recorder.run))
        self.active[contract.market_id] = (contract, recorder, task)
        self.started.append(contract.market_id)
        log(f"[{self.name}] recording {contract.market_id} ({contract.question[:50]}) "
            f"until {contract.resolves_at.isoformat()} -> {directory}")

    async def _finish(self, market_id: str) -> None:
        contract, recorder, task = self.active.pop(market_id)
        recorder.stop.set()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        recorder.close()
        self.finished.append(market_id)
        log(f"[{self.name}] finished {market_id}: {recorder.stats()}")

    async def tick(self) -> None:
        now_s = self.clock()
        # Discovery is synchronous HTTP (Gamma slug lookups, Kalshi market
        # reads). On the loop it blocked ~90 to 120 ms at every window open
        # (measured with the loop-lag gauge below), which is exactly when the new
        # window's socket is busiest; a thread keeps the loop draining sockets.
        contracts = await asyncio.to_thread(self.discoverer.discover, now_s, self.ahead)
        for contract in contracts:
            if contract.market_id not in self.active and \
                    contract.resolves_at_ms + self.grace_s * 1000 > now_s * 1000:
                self._start(contract)
        for market_id, (contract, _, _) in list(self.active.items()):
            if contract.resolves_at_ms + self.grace_s * 1000 <= now_s * 1000:
                await self._finish(market_id)

    async def run(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    await self.tick()
                except Exception as exc:
                    log(f"[{self.name}] tick failed: {type(exc).__name__}: {exc}")
                await self.sleep(self.poll_s)
        finally:
            for market_id in list(self.active):
                await self._finish(market_id)

    def stats(self) -> Dict[str, Any]:
        return {"series": self.name, "active": sorted(self.active), "started": len(self.started),
                "finished": len(self.finished)}


def build_series(spec: str):
    if spec == "polymarket:btc-5m":
        from predkit.venues.polymarket import Polymarket
        return PolymarketWindows(Polymarket(), "btc-updown-5m", 300)
    if spec == "polymarket:btc-15m":
        from predkit.venues.polymarket import Polymarket
        return PolymarketWindows(Polymarket(), "btc-updown-15m", 900)
    if spec.startswith("kalshi:"):
        from predkit.venues.kalshi import Kalshi
        series = spec.split(":", 1)[1].upper()
        window = 900 if series.endswith("15M") else 300
        return KalshiWindows(Kalshi(), series, window)
    raise SystemExit(f"unknown series {spec!r}; known: polymarket:btc-5m, polymarket:btc-15m, kalshi:<SERIES>")


def build_reference(spec: str, data_dir: Path, stall_timeout_s: Optional[float]) -> MarketRecorder:
    """`binance:<SYMBOL>` (book ticker)."""
    kind, _, symbol = spec.partition(":")
    if kind == "binance" and symbol:
        from predkit.venues.binance_reference import BinanceBookTicker
        return MarketRecorder(BinanceBookTicker(), symbol, reference_dir(data_dir, f"binance-{symbol}"),
                              stall_timeout_s=stall_timeout_s or 30.0)
    raise SystemExit(f"unknown reference {spec!r}; known: binance:<SYMBOL>")


class LoopLag:
    """How late the event loop wakes a 100 ms sleep: the number that decides
    whether a venue drops the connection as a slow consumer. Logged once a minute with
    the worst reading, so a stall shows up in the log beside the disconnect
    it caused. `gc.freeze()` at start moves everything allocated during
    setup out of the collector's reach; the hot path allocates strings,
    which the collector does not track, so full collections stay small."""

    def __init__(self, period_s: float = 0.1, report_s: float = 60.0):
        self.period_s = period_s
        self.report_s = report_s
        self.worst_ms = 0.0
        self.last_minute_worst_ms = 0.0

    async def run(self) -> None:
        import gc
        gc.freeze()
        gc.set_threshold(50_000, 20, 50)
        reported = time.monotonic()
        while True:
            before = time.monotonic()
            await asyncio.sleep(self.period_s)
            lag_ms = (time.monotonic() - before - self.period_s) * 1000
            self.worst_ms = max(self.worst_ms, lag_ms)
            if time.monotonic() - reported >= self.report_s:
                self.last_minute_worst_ms = self.worst_ms
                if self.worst_ms >= 25:
                    log(f"[loop] worst wake-up lag in the last minute: {self.worst_ms:.0f} ms")
                self.worst_ms = 0.0
                reported = time.monotonic()


async def run_all(recorders: List[SeriesRecorder], references: List[MarketRecorder]) -> None:
    lag = LoopLag()
    tasks = [asyncio.create_task(supervise(r.name, r.run)) for r in recorders]
    tasks += [asyncio.create_task(supervise(f"reference:{r.market_id}", r.run)) for r in references]
    tasks.append(asyncio.create_task(supervise("loop-lag", lag.run)))
    try:
        await asyncio.gather(*tasks)
    finally:
        for r in references:
            r.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--series", required=True, help="comma-separated, e.g. polymarket:btc-5m,kalshi:KXBTC15M")
    parser.add_argument("--reference", action="append", default=[],
                        help="binance:BTCUSDT; may repeat")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ahead", type=int, default=1, help="windows to open early")
    parser.add_argument("--poll", type=float, default=20.0)
    parser.add_argument("--stall-timeout", type=float, default=None,
                        help="override every series' message deadline; default is per series "
                             "(300 s Polymarket websocket, 30 s Kalshi poll, 30 s reference)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    recorders = [SeriesRecorder(spec, build_series(spec), data_dir, ahead=args.ahead, poll_s=args.poll,
                                stall_timeout_s=args.stall_timeout)
                 for spec in args.series.split(",") if spec.strip()]
    references = [build_reference(spec, data_dir, args.stall_timeout) for spec in args.reference]
    log(f"recording series {[r.name for r in recorders]} references {[r.market_id for r in references]} -> {data_dir}")
    try:
        asyncio.run(run_all(recorders, references))
    except KeyboardInterrupt:
        pass
    for r in recorders:
        log(str(r.stats()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
