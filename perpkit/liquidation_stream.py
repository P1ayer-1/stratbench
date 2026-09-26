"""Forced-liquidation order streams, archived verbatim.

Two venues, one recorder: Binance USDT-M futures' all-market liquidation
stream (`!forceOrder@arr`) and Bybit v5 public linear `allLiquidation.<symbol>`
for a list of symbols. Both are public, unsigned and free. The entrypoint is
`record_liquidations.py`; this module is the part the tests can import with no
`websockets` on the path and no network, the same split as
`record_hyperliquid.py` / `hyperliquid.py`.

Why this exists
---------------
Without a forced-flow series, a liquidation-cascade label has to be
reconstructed from bars and open-interest snapshots. This is that series,
recorded directly: which symbols were liquidated, which side, at what price,
in the same second.

What the venues document (documented, NOT measured - read 2026-09-23)
---------------------------------------------------------------------
Binance `!forceOrder@arr`: "only the latest one liquidation order within
1000ms will be pushed as the snapshot", per symbol. So in a cascade - the one
moment this feed is for - a symbol liquidating ten accounts in a second shows
ONE order. The stream UNDERCOUNTS exactly when it matters, and its rate is a
floor on the true liquidation rate, never an estimate of it. Nothing here has
yet measured how far below; that needs Bybit or Hyperliquid beside it.

Bybit `allLiquidation.<symbol>`: "push all liquidations that occur on Bybit",
"Push frequency: 500ms". Read as a 500 ms batch, not a sample, so it should
count every event - but whether a batch can hold more than one row per symbol
is not stated and has not been measured. That is why Bybit is recorded too:
the two together are the only way to put a number on Binance's undercount.

Connections, per each venue's docs (2026-09-23):
  Binance  `wss://fstream.binance.com/market/ws/!forceOrder@arr` - the
           ROUTED `/market` path, not the `/ws/!forceOrder@arr` the stream's
           own doc page prints. Measured 2026-09-23: the unrouted URL
           connects, answers `LIST_SUBSCRIPTIONS` with `["!forceOrder@arr"]`
           and delivers NOTHING (0 orders in 7 min, 0 `btcusdt@aggTrade` in
           5 s either), while `/market/ws/!forceOrder@arr` delivered 62 in
           90 s. The Connect page says it: an unrouted connection "will only
           receive data from the Public endpoint", and `forceOrder`, like
           `markPrice`, is Market. "A single connection is only valid for 24
           hours; expect to be disconnected at the 24 hour mark." The server
           sends a ping FRAME every 3 min and drops a connection that has not
           ponged within 10 min; `websockets` answers ping frames itself. 10
           incoming messages/s at most.
  Bybit    `wss://stream.bybit.com/v5/public/linear`, subscribe
           `{"op":"subscribe","args":["allLiquidation.BTCUSDT", ...]}`. The
           client must send `{"op":"ping"}` every 20 s or be dropped after
           10 min; the reply is a message, `{"op":"ping","ret_msg":"pong",...}`.

So a reconnect is ROUTINE here, not a fault: Binance guarantees one a day.
Every reconnect is logged with its reason, and the open hour's file is closed
first, so the next write creates `<channel>-<HH>.rNNN.jsonl.gz` beside it
rather than carrying on in a file whose tail may be torn (the rule in
`rawlog.py`).

Liveness in a sparse stream
---------------------------
`ingest._stream` and `HyperliquidMarketFeed._stream` await each message with a
deadline, because an open socket that delivers nothing raises nothing and
`supervise` would call it healthy forever. Book feeds can afford that: a book
is never silent for 30 s. A liquidation stream is - a quiet market prints no
forced orders for minutes - so a bare per-message deadline would mistake a
calm hour for a dead socket and reconnect through it, once per deadline.

Both venues have a request that always gets a reply: Bybit's ping gets a
pong, and Binance's `{"method":"LIST_SUBSCRIPTIONS","id":n}` gets the
connection's subscription list. One goes out every `ping_seconds`, the reply
is a message like any other, and the deadline measures the socket, not the
market. Binance's reply is also checked: a list without `!forceOrder@arr` is
a connection that will never deliver, and is dropped for a new one.

Control messages (acks, pongs, subscription lists) are counted and not
archived, as `HyperliquidMarketFeed` does with `pong` and
`subscriptionResponse`: a channel file holds that channel's events. Everything
with a payload is archived before it is looked at.

Layout (`perpkit/layout.venue_channel_dir`, resolved by the entrypoint)
-----------------------------------------------------------------------
    data/binance/forceOrder/raw/<day>/forceOrder-<HH>.jsonl.gz
    data/bybit/allLiquidation/raw/<day>/allLiquidation-<HH>.jsonl.gz
    data/<venue>/<channel>/.recorder.lock        one writer per channel (pid)

Venue-wide, deliberately: the cross-section is the point, and the symbol is
in every message (`o.s` on Binance, `data[].s` on Bybit).

Seen live 2026-09-23, not in Binance's payload doc: every `o` carries two
extra fields, `"ps"` (the pair, same as `s` here) and `"st": 1`, and `E`
lands about 1,010 ms after `o.T` - the documented 1000 ms snapshot window,
presumably. Archived verbatim either way. Measured rate on the routed path:
25 orders/60 s (0.42/s, 1.5 MB/day gzipped) and 62/90 s in another window.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .rawlog import RawEventLog

BINANCE_WS_BASE = "wss://fstream.binance.com/market/ws"   # routed: see docstring
BINANCE_STREAM = "!forceOrder@arr"
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

# Bybit documents 20 s; Binance has no documented client-ping interval, and
# one LIST_SUBSCRIPTIONS every 20 s is 1/200th of its 10 messages/s cap.
DEFAULT_PING_SECONDS = 20.0
# Three missed replies. Not the 30 s the book feeds use: a reply is only due
# every `ping_seconds`, and the deadline must outlive at least one of them.
DEFAULT_STALL_TIMEOUT_S = 60.0
DEFAULT_BYBIT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "SUIUSDT")

# What `classify` returns.
DATA = "data"          # a payload: archive it
CONTROL = "control"    # an ack, a pong, a subscription list: count it
LOST = "lost"          # the connection no longer carries the stream: reconnect
FATAL = "fatal"        # the venue refused the subscription: stop, say why
UNROUTED = "unrouted"  # nothing this feed knows


class BinanceForceOrders:
    """`!forceOrder@arr` in raw-stream mode: the stream is in the URL."""

    venue = "binance"
    channel = "forceOrder"
    stream = BINANCE_STREAM
    # Venue-wide across ~500 symbols, measured at 0.69/s: a
    # `--check` window with no order at all is the silent-shell failure
    # above, not a calm market.
    silence_is_a_fault = True

    def __init__(self) -> None:
        self.url = f"{BINANCE_WS_BASE}/{self.stream}"

    def describe(self) -> str:
        return (f"Binance USDT-M {self.stream} (venue-wide; documented as at "
                f"most one order per symbol per 1000 ms, so it undercounts "
                f"in cascades)")

    def subscribe_messages(self) -> List[Dict[str, Any]]:
        return []

    def ping_message(self, sequence: int) -> Dict[str, Any]:
        return {"method": "LIST_SUBSCRIPTIONS", "id": sequence}

    def classify(self, message: Dict[str, Any]) -> Tuple[str, str]:
        if message.get("e") == "forceOrder":
            return DATA, ""
        if "id" in message:
            if "error" in message:
                return FATAL, f"Binance error reply: {message['error']}"
            result = message.get("result")
            if isinstance(result, list) and self.stream not in result:
                return LOST, (f"subscription list {result} no longer holds "
                              f"{self.stream}")
            return CONTROL, ""
        return UNROUTED, ""


class BybitAllLiquidation:
    """`allLiquidation.<symbol>` for each symbol, on one public linear socket."""

    venue = "bybit"
    channel = "allLiquidation"
    url = BYBIT_WS_URL
    # Five symbols printed 3 in one minute and 0 in the next three in a
    # live sample: an empty window here is a calm market.
    silence_is_a_fault = False

    def __init__(self, symbols: Sequence[str]) -> None:
        self.symbols = list(dict.fromkeys(symbols))
        if not self.symbols:
            raise ValueError("Bybit needs at least one symbol")
        self.topics = [f"{self.channel}.{symbol}" for symbol in self.symbols]

    def describe(self) -> str:
        return (f"Bybit v5 linear {self.channel} x {len(self.symbols)} "
                f"symbol(s): {', '.join(self.symbols)} (documented as every "
                f"liquidation, pushed in 500 ms batches)")

    def subscribe_messages(self) -> List[Dict[str, Any]]:
        return [{"op": "subscribe", "args": list(self.topics)}]

    def ping_message(self, sequence: int) -> Dict[str, Any]:
        return {"op": "ping"}

    def classify(self, message: Dict[str, Any]) -> Tuple[str, str]:
        if "topic" in message:
            return DATA, ""
        op = message.get("op")
        if op in ("ping", "pong"):
            return CONTROL, ""
        if op == "subscribe":
            if message.get("success") is False:
                return FATAL, (f"Bybit refused the subscription: "
                               f"{message.get('ret_msg') or message}")
            return CONTROL, ""
        return UNROUTED, ""


def venue_spec(venue: str, symbols: Optional[Sequence[str]] = None):
    """The venue object for a `--venue`, with every refusal listed.

    `symbols` is Bybit's; Binance's stream is venue-wide by construction, so
    naming symbols for it would record something other than what was asked.
    """
    problems: List[str] = []
    if venue == "binance":
        if symbols:
            problems.append(
                f"--symbols is for --venue bybit: Binance's {BINANCE_STREAM} "
                f"covers every USDT-M symbol and cannot be narrowed.")
    elif venue == "bybit":
        chosen = list(symbols) if symbols else list(DEFAULT_BYBIT_SYMBOLS)
        duplicates = sorted({s for s in chosen if chosen.count(s) > 1})
        if duplicates:
            problems.append(f"symbol(s) listed more than once: "
                            f"{', '.join(duplicates)}")
        bad = [s for s in chosen if not s.isalnum() or s != s.upper()]
        if bad:
            problems.append(f"Bybit symbols are uppercase like BTCUSDT, not "
                            f"{', '.join(bad)}")
    else:
        problems.append(f"unknown venue {venue!r}; use binance or bybit.")
    if problems:
        raise SystemExit("Refusing to start:\n"
                         + "\n".join(f"  - {p}" for p in problems))
    if venue == "binance":
        return BinanceForceOrders()
    return BybitAllLiquidation(chosen)


class LiquidationFeed:
    """One venue's liquidation stream: connect, subscribe, archive, reconnect.

    Run `run()` under `perpkit.supervise.supervise`. Archive first, count second: the
    `on_message` consumer (the `--list-cost` sampler, the `--check` printer)
    can fail without costing the event.
    """

    def __init__(
        self,
        spec: Any,
        *,
        root: Path,
        record_raw: bool = True,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        ping_seconds: float = DEFAULT_PING_SECONDS,
        reconnect_backoff_s: float = 1.0,
        stop_after_messages: Optional[int] = None,
        stop_after_seconds: Optional[float] = None,
        on_message: Optional[Callable[[Dict[str, Any]], Any]] = None,
        on_control: Optional[Callable[[Dict[str, Any]], Any]] = None,
        on_log: Callable[[str], Any] = print,
        connect: Optional[Callable[..., Any]] = None,
    ):
        if ping_seconds <= 0 or stall_timeout_s <= 0:
            raise ValueError("ping_seconds and stall_timeout_s must be positive")
        if ping_seconds >= stall_timeout_s:
            raise ValueError(
                f"ping_seconds ({ping_seconds:g}) must be shorter than "
                f"stall_timeout_s ({stall_timeout_s:g}): a quiet market would "
                f"otherwise be declared a stall before a reply was even due")
        self.spec = spec
        # `root` is `perpkit.layout.venue_channel_dir(data, venue, channel)`,
        # resolved by the caller so this class holds no path knowledge.
        self.root = Path(root)
        self.log = RawEventLog(self.root, enabled=record_raw,
                               channels={spec.channel},
                               # A sparse stream: 200 lines could be an hour.
                               flush_lines=50)
        self.stall_timeout_s = stall_timeout_s
        self.ping_seconds = ping_seconds
        self.reconnect_backoff_s = reconnect_backoff_s
        self.stop_after_messages = stop_after_messages
        self.stop_after_seconds = stop_after_seconds
        self.on_message = on_message
        self.on_control = on_control
        self._log = on_log
        self._connect = connect

        self.connected = False
        self.connections = 0
        self.messages = 0
        self.control_messages = 0
        self.unrouted = 0
        self.errors = 0
        self.reconnects = 0
        self.stalls = 0
        self.pings = 0
        self.lost_subscriptions = 0
        self.last_error: Optional[str] = None
        self.stopped = False
        # Seconds from the first connection to the stop, for `--list-cost`:
        # Binance's close handshake alone was measured at ~10 s,
        # which counted against a 60 s sample when the clock ran to exit.
        self.elapsed_s: Optional[float] = None
        self._last_message = time.monotonic()
        self._started: Optional[float] = None

    # ---- messages --------------------------------------------------------

    def handle_message(self, message: Any) -> str:
        """Archive or count one message; returns its kind."""
        if not isinstance(message, dict):
            self.unrouted += 1
            return UNROUTED
        kind, detail = self.spec.classify(message)
        if kind == DATA:
            self.messages += 1
            self.log.write(self.spec.channel, message)   # archive first
            if self.on_message is not None:
                try:
                    self.on_message(message)
                except Exception as exc:  # noqa: BLE001 - the archive has it
                    self.errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._log(f"[{self.spec.venue}] consumer error: "
                              f"{self.last_error}")
        elif kind == CONTROL:
            self.control_messages += 1
            if self.on_control is not None:
                try:
                    self.on_control(message)
                except Exception:  # noqa: BLE001
                    pass
        elif kind == LOST:
            self.lost_subscriptions += 1
            self.last_error = detail
            self._log(f"[{self.spec.venue}] {detail} - reconnecting.")
        elif kind == FATAL:
            self.errors += 1
            self.last_error = detail
            self._log(f"[{self.spec.venue}] {detail}")
        else:
            self.unrouted += 1
        return kind

    # ---- the loop --------------------------------------------------------

    async def run(self) -> None:
        """Connect, subscribe, stream; reconnect until told to stop."""
        connect = self._connect
        if connect is None:
            import websockets  # here, so the module imports without it

            connect = websockets.connect
        backoff = self.reconnect_backoff_s
        while not self.stopped:
            reason = "closed by the venue"
            try:
                async with connect(self.spec.url, max_size=8 * 2 ** 20) as ws:
                    for request in self.spec.subscribe_messages():
                        await ws.send(json.dumps(request))
                    self.connected = True
                    self.connections += 1
                    if self._started is None:
                        self._started = time.monotonic()
                    backoff = self.reconnect_backoff_s
                    self._log(f"[{self.spec.venue}] connected "
                              f"(#{self.connections}): {self.spec.describe()}")
                    reason = await self._stream(ws)
            except asyncio.CancelledError:
                raise
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect is the answer
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                reason = self.last_error
            finally:
                self.connected = False
                # Close the open hour: the next write creates `.rNNN` beside
                # it instead of continuing a file this connection may have
                # left torn. See `rawlog.py`.
                self.log.close()
            if self.stopped:
                break
            self.reconnects += 1
            self._log(f"[{self.spec.venue}] connection ended: {reason} - "
                      f"reconnect #{self.reconnects} in {backoff:g}s "
                      f"(next file gets a .rNNN suffix)")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _stream(self, ws: Any) -> str:
        """Pump messages until the socket closes, stalls, or is told to stop.

        Returns why it stopped, for the reconnect log line. The deadline is
        per message; a timeout may only mean a ping is due, and `websockets`
        documents a cancelled `recv()` as losing no message.
        """
        self._last_message = time.monotonic()
        if self._started is None:      # `_stream` driven directly by a test
            self._started = self._last_message
        next_ping = self._last_message + self.ping_seconds
        stop_at = (None if self.stop_after_seconds is None
                   else self._started + self.stop_after_seconds)
        while True:
            now = time.monotonic()
            if stop_at is not None and now >= stop_at:
                self._stop("sample complete")
                return "sample complete"
            silent_until = self._last_message + self.stall_timeout_s
            if now >= silent_until:
                self.stalls += 1
                self._log(f"[{self.spec.venue}] feed stalled: nothing in "
                          f"{self.stall_timeout_s:g}s, not even a reply to "
                          f"our ping (socket still open)")
                return f"stalled {self.stall_timeout_s:g}s"
            if now >= next_ping:
                self.pings += 1
                await ws.send(json.dumps(self.spec.ping_message(self.pings)))
                next_ping = now + self.ping_seconds
                self.log.flush()   # a quiet hour must not sit in the buffer
            deadline = min(silent_until, next_ping)
            if stop_at is not None:
                deadline = min(deadline, stop_at)
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.0, deadline - now))
            except asyncio.TimeoutError:
                continue
            self._last_message = time.monotonic()
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                self.unrouted += 1
                continue
            kind = self.handle_message(message)
            if kind == FATAL:
                raise SystemExit(
                    f"{self.last_error}\nNothing would be recorded on this "
                    f"connection. Check the symbols with --check.")
            if kind == LOST:
                return "subscription lost"
            if (self.stop_after_messages is not None
                    and self.messages >= self.stop_after_messages):
                self._stop(f"{self.messages} message(s) seen")
                return f"{self.messages} message(s) seen"

    def _stop(self, why: str) -> None:
        self.stopped = True
        self.elapsed_s = time.monotonic() - (self._started or time.monotonic())

    # ---- lifecycle -------------------------------------------------------

    def flush(self) -> None:
        self.log.flush()

    def close(self) -> None:
        self.log.close()

    def summary_line(self) -> str:
        return (f"[{self.spec.venue}] {'connected' if self.connected else 'DISCONNECTED'}"
                f"; {self.messages:,} messages, {self.control_messages:,} control,"
                f" {self.reconnects} reconnect(s), {self.stalls} stall(s),"
                f" {self.lost_subscriptions} lost subscription(s),"
                f" {self.errors} error(s)")


# ---------------------------------------------------------------------------
# Disk arithmetic
# ---------------------------------------------------------------------------


class CostSample:
    """What a sample of messages would cost on disk, scaled to a day.

    Each message is measured as the line `RawEventLog` would write - the
    `{"t","n","m"}` wrapper included, with a fixed 13-digit `t` - and the
    whole sample is gzipped once at the writer's level 6. One gzip over the
    sample is a touch better than the hourly file gets (the writer flushes
    every 5 s, and each flush costs a few bytes), so the compressed figure is
    a floor, and a short sample of a calm market is a floor on the rate too.
    """

    T_MS = 1_700_000_000_000

    def __init__(self, compress_level: int = 6):
        self.lines = 0
        self.raw_bytes = 0
        self._buffer = io.BytesIO()
        self._gzip = gzip.GzipFile(fileobj=self._buffer, mode="wb",
                                   compresslevel=compress_level)

    def add(self, message: Dict[str, Any]) -> None:
        self.lines += 1
        line = json.dumps({"t": self.T_MS, "n": self.lines, "m": message},
                          separators=(",", ":"), default=str) + "\n"
        self.raw_bytes += len(line)
        self._gzip.write(line.encode("utf-8"))

    def report(self, seconds: float) -> Dict[str, Any]:
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self._gzip.close()
        gzip_bytes = len(self._buffer.getvalue())
        per_day = 86_400.0 / seconds
        return {
            "seconds": seconds,
            "messages": self.lines,
            "per_second": self.lines / seconds,
            "messages_per_day": round(self.lines * per_day),
            "raw_bytes": self.raw_bytes,
            "gzip_bytes": gzip_bytes,
            "raw_bytes_per_day": round(self.raw_bytes * per_day),
            "gzip_bytes_per_day": round(gzip_bytes * per_day),
        }


def format_cost(report: Dict[str, Any]) -> str:
    mb = 1024 * 1024
    return (
        f"{report['messages']} message(s) in {report['seconds']:g}s = "
        f"{report['per_second']:.3f}/s\n"
        f"per day: {report['messages_per_day']:,} messages, "
        f"{report['raw_bytes_per_day'] / mb:.2f} MB uncompressed, "
        f"{report['gzip_bytes_per_day'] / mb:.2f} MB gzipped\n"
        f"per 30 days gzipped: {30 * report['gzip_bytes_per_day'] / mb:.1f} MB"
    )
