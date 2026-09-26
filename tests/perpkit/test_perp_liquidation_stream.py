"""The liquidation recorders: routing, the archive, liveness, reconnects.

No network: the stream takes a fake socket and `run()` a fake `connect`.
The tests that earn their place are the ones about silence and reconnects -
a liquidation feed is quiet for minutes in a calm market and is cut by
Binance once a day by design, so a recorder that mistakes either for health,
or for a fault, loses the cascade it exists to catch.
"""

import asyncio
import gzip
import json
import os

import pytest

from perpkit.layout import instrument_dirs, venue_channel_dir
from perpkit.liquidation_stream import (
    BINANCE_STREAM,
    CONTROL,
    DATA,
    BinanceForceOrders,
    BybitAllLiquidation,
    CostSample,
    LiquidationFeed,
    venue_spec,
)
from perpkit.rawlog import iter_directory

FORCE = {"e": "forceOrder", "E": 1568014460893, "o": {
    "s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "0.014",
    "p": "9910", "ap": "9910", "X": "FILLED", "l": "0.014", "z": "0.014",
    "T": 1568014460893}}
LISTED = {"result": ["!forceOrder@arr"], "id": 1}
UNLISTED = {"result": [], "id": 2}
BYBIT_LIQ = {"topic": "allLiquidation.ROSEUSDT", "type": "snapshot",
             "ts": 1739502303204, "data": [{
                 "T": 1739502302929, "s": "ROSEUSDT", "S": "Sell",
                 "v": "20000", "p": "0.04499"}]}
BYBIT_ACK = {"success": True, "ret_msg": "", "conn_id": "c1", "op": "subscribe"}
BYBIT_NACK = {"success": False, "ret_msg": "Invalid symbol :[allLiquidation.NOPE]",
              "conn_id": "c1", "op": "subscribe"}
BYBIT_PONG = {"success": True, "ret_msg": "pong", "conn_id": "c1", "op": "ping"}


def binance_feed(tmp_path, **overrides):
    spec = BinanceForceOrders()
    options = dict(root=venue_channel_dir(tmp_path, spec.venue, spec.channel),
                   on_log=lambda message: None)
    options.update(overrides)
    return LiquidationFeed(spec, **options)


def bybit_feed(tmp_path, **overrides):
    spec = BybitAllLiquidation(["BTCUSDT", "ETHUSDT"])
    options = dict(root=venue_channel_dir(tmp_path, spec.venue, spec.channel),
                   on_log=lambda message: None)
    options.update(overrides)
    return LiquidationFeed(spec, **options)


def archived(tmp_path, venue, channel):
    return [m for _, _, m in iter_directory(
        tmp_path / venue / channel / "raw", channel)]


class FakeSocket:
    def __init__(self, messages=(), gap=0.0, then=None):
        self.queue = [json.dumps(message) for message in messages]
        self.gap = gap
        self.then = then          # raised once the queue is empty
        self.sent = []

    async def recv(self):
        if self.queue:
            if self.gap:
                await asyncio.sleep(self.gap)
            return self.queue.pop(0)
        if self.then is not None:
            raise self.then
        await asyncio.Event().wait()   # an open socket delivering nothing

    async def send(self, text):
        self.sent.append(json.loads(text))


class FakeConnect:
    """`websockets.connect` stand-in: each call yields the next socket."""

    def __init__(self, sockets):
        self.sockets = list(sockets)
        self.urls = []

    def __call__(self, url, **kwargs):
        self.urls.append(url)
        return self

    async def __aenter__(self):
        return self.sockets.pop(0)

    async def __aexit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Layout: venue-wide, invisible to the per-instrument tools
# ---------------------------------------------------------------------------


def test_the_channel_lives_beside_the_instruments_not_under_one(tmp_path):
    """Guards the feed being filed under `data/BTC-USDT/` or split per symbol:
    the path is `data/binance/forceOrder/raw/<day>/forceOrder-<HH>.jsonl.gz`."""
    feed = binance_feed(tmp_path)
    feed.handle_message(FORCE)
    feed.close()
    (path,) = list((tmp_path / "binance" / "forceOrder" / "raw").rglob("*.jsonl.gz"))
    assert path.name.startswith("forceOrder-") and path.name.endswith(".jsonl.gz")
    assert path.parent.parent == tmp_path / "binance" / "forceOrder" / "raw"
    assert feed.root == venue_channel_dir(tmp_path, "binance", "forceOrder")


def test_a_venue_channel_is_not_an_instrument_directory(tmp_path):
    """Guards a BloFin tool reading the liquidation archive as an instrument."""
    (tmp_path / "BTC-USDT" / "raw").mkdir(parents=True)
    binance_feed(tmp_path).handle_message(FORCE)
    bybit_feed(tmp_path).handle_message(BYBIT_LIQ)
    assert list(instrument_dirs(tmp_path)) == ["BTC-USDT"]


def test_a_venue_named_like_an_instrument_is_refused():
    """Guards `venue_channel_dir(data, "BTC-USDT", ...)` quietly creating a
    directory `instrument_dirs` would then claim."""
    with pytest.raises(ValueError, match="instrument"):
        venue_channel_dir("data", "BTC-USDT", "forceOrder")
    with pytest.raises(ValueError):
        venue_channel_dir("data", "Binance", "forceOrder")


# ---------------------------------------------------------------------------
# Routing: what is archived and what is only counted
# ---------------------------------------------------------------------------


def test_binance_orders_are_archived_verbatim_and_replies_are_not(tmp_path):
    """Guards two failures at once: a payload rewritten on the way to disk,
    and LIST_SUBSCRIPTIONS replies (one per 20 s, ~4,300 a day) padding the
    channel file with rows that are not liquidations."""
    feed = binance_feed(tmp_path)
    assert feed.handle_message(LISTED) == CONTROL
    assert feed.handle_message(FORCE) == DATA
    assert feed.handle_message(FORCE) == DATA
    feed.close()
    assert archived(tmp_path, "binance", "forceOrder") == [FORCE, FORCE]
    assert (feed.messages, feed.control_messages) == (2, 1)


def test_bybit_liquidations_are_archived_and_acks_and_pongs_are_not(tmp_path):
    feed = bybit_feed(tmp_path)
    feed.handle_message(BYBIT_ACK)
    feed.handle_message(BYBIT_PONG)
    feed.handle_message(BYBIT_LIQ)
    feed.close()
    assert archived(tmp_path, "bybit", "allLiquidation") == [BYBIT_LIQ]
    assert (feed.messages, feed.control_messages) == (1, 2)


def test_the_archived_line_is_the_rawlog_wrapper(tmp_path):
    """Guards a writer of its own: the line on disk must be `{"t","n","m"}`
    with `n` starting at 1, exactly as `RawEventLog` writes every other
    channel, or `iter_events` and `replay.py` cannot read it."""
    feed = binance_feed(tmp_path)
    feed.handle_message(FORCE)
    feed.close()
    (path,) = list((tmp_path / "binance").rglob("*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert list(record) == ["t", "n", "m"]
    assert record["n"] == 1
    assert record["m"] == FORCE
    assert record["t"] > 1_600_000_000_000


def test_a_failing_consumer_does_not_cost_the_archive(tmp_path):
    def explode(message):
        raise RuntimeError("sampler bug")

    feed = binance_feed(tmp_path, on_message=explode)
    feed.handle_message(FORCE)
    feed.close()
    assert archived(tmp_path, "binance", "forceOrder") == [FORCE]
    assert feed.errors == 1


def test_a_stream_missing_from_the_subscription_list_ends_the_connection(tmp_path):
    """Guards the silent failure: a connection Binance still answers but no
    longer feeds would pass the per-message deadline forever."""
    feed = binance_feed(tmp_path, stall_timeout_s=1.0, ping_seconds=0.5)
    asyncio.run(feed._stream(FakeSocket([LISTED, UNLISTED])))
    feed.close()
    assert feed.lost_subscriptions == 1
    assert feed.control_messages == 1
    assert "!forceOrder@arr" in feed.last_error


def test_a_refused_bybit_subscription_stops_with_the_venues_reason(tmp_path):
    """Guards a recorder that runs for days archiving nothing because one
    symbol was misspelt: the refusal carries Bybit's own message."""
    feed = bybit_feed(tmp_path, stall_timeout_s=1.0, ping_seconds=0.5)
    with pytest.raises(SystemExit) as caught:
        asyncio.run(feed._stream(FakeSocket([BYBIT_NACK])))
    feed.close()
    assert "Invalid symbol" in str(caught.value)
    assert feed.errors == 1
    assert not (tmp_path / "bybit").exists()


# ---------------------------------------------------------------------------
# Liveness: silence, pings, and the sparse-stream deadline
# ---------------------------------------------------------------------------


def test_silence_ends_the_stream(tmp_path):
    feed = binance_feed(tmp_path, stall_timeout_s=0.05, ping_seconds=0.02)
    asyncio.run(feed._stream(FakeSocket()))
    feed.close()
    assert feed.stalls == 1


def test_a_trickle_is_not_a_stall(tmp_path):
    """Five messages 20 ms apart under a 100 ms deadline are all delivered;
    the stall only comes once they stop."""
    feed = binance_feed(tmp_path, stall_timeout_s=0.1, ping_seconds=0.05)
    asyncio.run(feed._stream(FakeSocket([FORCE] * 5, gap=0.02)))
    feed.close()
    assert feed.messages == 5
    assert feed.stalls == 1


def test_binance_is_asked_for_its_subscriptions_and_bybit_is_pinged(tmp_path):
    """Guards a quiet market being read as a dead socket: the request that
    gets a reply must go out before the deadline, with the venue's own
    wording - Binance ids are unsigned ints, Bybit's ping is `{"op":"ping"}`."""
    feed = binance_feed(tmp_path, stall_timeout_s=0.3, ping_seconds=0.05)
    socket = FakeSocket()
    asyncio.run(feed._stream(socket))
    feed.close()
    assert socket.sent[0] == {"method": "LIST_SUBSCRIPTIONS", "id": 1}
    assert socket.sent[1] == {"method": "LIST_SUBSCRIPTIONS", "id": 2}
    assert feed.pings >= 2

    feed = bybit_feed(tmp_path, stall_timeout_s=0.3, ping_seconds=0.05)
    socket = FakeSocket()
    asyncio.run(feed._stream(socket))
    feed.close()
    assert socket.sent[0] == {"op": "ping"}
    assert feed.pings >= 2


def test_a_reply_to_the_ping_keeps_the_socket_alive(tmp_path):
    """Guards the deadline counting only liquidations: pongs arriving 50 ms
    apart under a 250 ms deadline are messages, so no stall across ten of
    them (500 ms, twice the deadline).

    The margins are wide on purpose. On Windows the event loop's timer ticks
    about every 15.6 ms, and from Python 3.12 `wait_for` cancels an inner
    task whose result lands in the same tick as the timeout; a fake socket
    30 ms apart under a 50 ms ping deadline loses that race every time."""
    feed = bybit_feed(tmp_path, stall_timeout_s=0.25, ping_seconds=0.12)
    asyncio.run(feed._stream(FakeSocket([BYBIT_PONG] * 10, gap=0.05)))
    feed.close()
    assert feed.control_messages == 10
    assert feed.messages == 0
    assert feed.stalls == 1     # only after the pongs stop


def test_a_ping_interval_at_or_past_the_deadline_is_refused(tmp_path):
    """Guards a configuration that stalls every quiet minute: the reply must
    be due before the deadline it is meant to feed."""
    with pytest.raises(ValueError, match="ping_seconds"):
        binance_feed(tmp_path, stall_timeout_s=20.0, ping_seconds=20.0)


# ---------------------------------------------------------------------------
# Reconnects: routine, logged, and a new file rather than an append
# ---------------------------------------------------------------------------


def test_a_dropped_connection_is_reconnected_into_a_new_file(tmp_path):
    """Guards the 24-hour cut being written into a file whose tail the dead
    connection may have torn: two orders on the first socket, one on the
    second, and the day holds `forceOrder-HH.jsonl.gz` with 2 records plus
    `forceOrder-HH.r001.jsonl.gz` with 1, the reconnect logged once."""
    lines = []
    first = FakeSocket([FORCE, FORCE], then=ConnectionError("24h mark"))
    second = FakeSocket([FORCE])
    connect = FakeConnect([first, second])
    feed = binance_feed(tmp_path, on_log=lines.append, connect=connect,
                        reconnect_backoff_s=0.01, stop_after_messages=3,
                        stall_timeout_s=1.0, ping_seconds=0.5)
    asyncio.run(feed.run())

    assert connect.urls == ["wss://fstream.binance.com/market/ws/!forceOrder@arr"] * 2
    assert (feed.connections, feed.reconnects, feed.messages) == (2, 1, 3)
    day_dir, = [p for p in (tmp_path / "binance" / "forceOrder" / "raw").iterdir()]
    names = sorted(p.name for p in day_dir.iterdir())
    assert len(names) == 2
    assert names[1] == names[0].replace(".jsonl.gz", ".r001.jsonl.gz")
    counts = [sum(1 for _ in gzip.open(day_dir / name, "rt")) for name in names]
    assert counts == [2, 1]
    assert [line for line in lines if "reconnect #1" in line and "24h mark" in line]


def test_bybit_subscribes_every_symbol_in_one_request(tmp_path):
    socket = FakeSocket([BYBIT_ACK, BYBIT_LIQ])
    feed = bybit_feed(tmp_path, connect=FakeConnect([socket]),
                      stop_after_messages=1, stall_timeout_s=1.0, ping_seconds=0.5)
    asyncio.run(feed.run())
    assert socket.sent[0] == {"op": "subscribe",
                              "args": ["allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT"]}
    assert archived(tmp_path, "bybit", "allLiquidation") == [BYBIT_LIQ]


def test_a_time_boxed_sample_stops_without_recording(tmp_path):
    """`--list-cost` and `--check` must leave no file behind."""
    seen = []
    feed = binance_feed(tmp_path, record_raw=False, connect=FakeConnect([FakeSocket([FORCE])]),
                        stop_after_seconds=0.05, on_message=seen.append,
                        stall_timeout_s=1.0, ping_seconds=0.5)
    asyncio.run(feed.run())
    assert seen == [FORCE]
    assert feed.stopped
    assert not (tmp_path / "binance").exists()


# ---------------------------------------------------------------------------
# The command line's refusals
# ---------------------------------------------------------------------------


def test_binance_uses_the_routed_market_path():
    """Guards the URL the stream's own doc page prints: on 2026-09-23
    `wss://fstream.binance.com/ws/!forceOrder@arr` connected, acknowledged the
    subscription and delivered nothing for 7 minutes; only the `/market`
    routed path delivers `forceOrder`."""
    assert BinanceForceOrders().url == "wss://fstream.binance.com/market/ws/!forceOrder@arr"
    assert BinanceForceOrders().silence_is_a_fault is True
    assert BybitAllLiquidation(["BTCUSDT"]).silence_is_a_fault is False


def test_symbols_on_binance_are_refused_with_the_reason():
    with pytest.raises(SystemExit) as caught:
        venue_spec("binance", ["BTCUSDT"])
    assert BINANCE_STREAM in str(caught.value)
    assert "--venue bybit" in str(caught.value)


def test_bybit_defaults_and_refusals():
    """Guards a lowercase or repeated symbol reaching Bybit, whose refusal
    would then arrive only after the process was already running."""
    spec = venue_spec("bybit", None)
    assert spec.topics == ["allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT",
                           "allLiquidation.SOLUSDT", "allLiquidation.DOGEUSDT",
                           "allLiquidation.SUIUSDT"]
    with pytest.raises(SystemExit) as caught:
        venue_spec("bybit", ["btcusdt", "ETHUSDT", "ETHUSDT"])
    message = str(caught.value)
    assert "btcusdt" in message and "more than once: ETHUSDT" in message


# ---------------------------------------------------------------------------
# Disk arithmetic
# ---------------------------------------------------------------------------


def test_cost_sample_scales_the_rawlog_line_to_a_day():
    """Two messages `{"a":1}` in 60 s. Each line on disk is
    `{"t":1700000000000,"n":1,"m":{"a":1}}` + newline = 38 bytes, so 76 bytes
    per minute = 109,440 per day and 2,880 messages per day."""
    sample = CostSample()
    sample.add({"a": 1})
    sample.add({"a": 1})
    report = sample.report(60.0)
    lines = (b'{"t":1700000000000,"n":1,"m":{"a":1}}\n'
             b'{"t":1700000000000,"n":2,"m":{"a":1}}\n')
    assert report["raw_bytes"] == 76 == len(lines)
    assert report["raw_bytes_per_day"] == 109_440
    assert report["messages_per_day"] == 2_880
    assert report["per_second"] == pytest.approx(2 / 60)
    assert report["gzip_bytes"] == len(gzip.compress(lines, compresslevel=6))
    assert report["gzip_bytes_per_day"] == report["gzip_bytes"] * 1440


def test_the_lock_is_a_pid_file_in_the_channel_directory(tmp_path):
    """One writer per channel: the lock names this process, beside `raw/`."""
    from perpkit.hyperliquid import ExclusiveLock

    feed = binance_feed(tmp_path)
    lock = ExclusiveLock(feed.root / ".recorder.lock", holder="x").acquire()
    try:
        assert (tmp_path / "binance" / "forceOrder" / ".recorder.lock"
                ).read_text(encoding="utf-8") == str(os.getpid())
    finally:
        lock.release()
