"""Feed-level routing and desync recovery.

`MicrostructureFeed.handle_message` is pure — it takes a parsed message and
mutates state — so it is testable without a websocket, a network, or the SDK.
The class itself imports the SDK at module load, so these tests are skipped
when it isn't installed rather than failing.
"""

import asyncio

import pytest

pytest.importorskip("blofin", reason="BloFin SDK not installed")

from perpkit.ingest import MicrostructureFeed


def make_feed(tmp_path):
    return MicrostructureFeed("BTC-USDT", data_dir=tmp_path, record=False)


def books(bids, asks, seq, prev=None, ts=1_700_000_000_000, action="update"):
    data = {"bids": bids, "asks": asks, "ts": str(ts), "seqId": str(seq)}
    data["prevSeqId"] = "0" if prev is None else str(prev)
    return {"arg": {"channel": "books"}, "action": action, "data": data}


def seed(feed):
    feed.handle_message(
        books([[100.0, 5], [99.5, 5]], [[100.5, 5], [101.0, 5]], seq=1, action="snapshot")
    )


def test_snapshot_then_update_produces_features(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    feed.handle_message(books([[100.0, 9]], [], seq=2, prev=1, ts=1_700_000_000_100))
    assert feed.latest.is_valid
    assert feed.latest.mid == pytest.approx(100.25)


def test_sequence_gap_requests_resync(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    assert feed.handle_message(books([[100.0, 9]], [], seq=9, prev=8)) is True


def test_persistently_crossed_book_requests_resync(tmp_path):
    """Regression: a crossed book used to invalidate features forever with no
    recovery path, because nothing removes the stale level on its own."""
    feed = make_feed(tmp_path)
    seed(feed)
    # Put a bid above the best ask and never clear it.
    results = []
    for index in range(4):
        results.append(
            feed.handle_message(
                books([[102.0, 5]], [], seq=2 + index, prev=1 + index)
            )
        )
    assert feed.book.is_crossed() or not feed.book.ready
    assert results[-1] is True, "feed must eventually force a resync"
    assert results[0] is False, "a single crossed update should be tolerated"


def test_trades_route_to_the_tape(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    feed.handle_message(
        {
            "arg": {"channel": "trades"},
            "data": [{"price": "100.2", "size": "3", "side": "buy", "ts": "1700000000050"}],
        }
    )
    assert len(feed.tape.trades) == 1
    assert feed.latest.tfi_5s == pytest.approx(1.0)


def test_funding_rate_is_captured(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    feed.handle_message(
        {
            "arg": {"channel": "funding-rate"},
            "data": [{"fundingRate": "0.0002", "fundingTime": "1700000600000"}],
        }
    )
    assert feed.engine.funding_rate == pytest.approx(0.0002)


def test_unknown_channel_is_ignored(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    assert feed.handle_message({"arg": {"channel": "nonsense"}, "data": [{}]}) is False


# ---------------------------------------------------------------------------
# The stall watchdog
# ---------------------------------------------------------------------------
#
# The failure this guards is the one that raises nothing: socket open, pings
# answered, subscription silently delivering no data. `supervise` cannot help
# there - it restarts loops that fail, and this loop does not fail, it waits.


class FakeClient:
    """A `listen()` that yields a scripted prefix and then goes silent."""

    def __init__(self, messages, *, then_silent=True):
        self._messages = list(messages)
        self._then_silent = then_silent
        self.closed = False

    async def listen(self):
        for message in self._messages:
            yield message
        if self._then_silent:
            await asyncio.Event().wait()  # never returns, like a dead feed

    async def close(self):
        self.closed = True


def test_silence_past_the_timeout_ends_the_stream(tmp_path):
    feed = make_feed(tmp_path)
    feed.stall_timeout_s = 0.05
    client = FakeClient([])

    desynced = asyncio.run(feed._stream(client))

    assert desynced is False, "a stall is not a desync; the reason differs"
    assert feed.stalls == 1
    # Returning is the whole point: the caller reconnects. Blocking here is
    # what silently killed the overnight data.


def test_messages_keep_the_stream_alive(tmp_path):
    feed = make_feed(tmp_path)
    feed.stall_timeout_s = 5.0
    snapshot = books([[100.0, 5]], [[100.5, 5]], seq=1, action="snapshot")
    client = FakeClient([snapshot, books([[100.0, 9]], [], seq=2, prev=1)],
                        then_silent=False)

    desynced = asyncio.run(feed._stream(client))

    assert desynced is False
    assert feed.stalls == 0, "a feed that is delivering must never be stalled"
    assert feed.messages == 2


def test_the_timeout_measures_silence_not_total_time(tmp_path):
    """A slow feed is not a stalled one.

    Getting this wrong in the obvious way — a deadline on the whole stream
    rather than on each message — would reconnect a perfectly healthy feed
    every `stall_timeout_s`, throwing away the book snapshot each time.
    """
    feed = make_feed(tmp_path)
    feed.stall_timeout_s = 0.1

    class Trickle:
        async def listen(self):
            for index in range(6):
                await asyncio.sleep(0.03)   # under the timeout, repeatedly
                yield books([[100.0, 5 + index]], [[100.5, 5]],
                            seq=index + 1,
                            action="snapshot" if index == 0 else "update",
                            prev=None if index == 0 else index)

        async def close(self):
            pass

    asyncio.run(feed._stream(Trickle()))

    assert feed.stalls == 0
    assert feed.messages == 6, "0.18s of trickle beats a 0.1s per-message gap"


def test_a_desync_is_reported_separately_from_a_stall(tmp_path):
    feed = make_feed(tmp_path)
    feed.stall_timeout_s = 5.0
    # A gap: seq jumps without a matching prevSeqId.
    client = FakeClient(
        [books([[100.0, 5]], [[100.5, 5]], seq=1, action="snapshot"),
         books([[100.0, 9]], [], seq=9, prev=8)],
        then_silent=False,
    )

    assert asyncio.run(feed._stream(client)) is True
    assert feed.stalls == 0


def test_status_exposes_silence_before_the_watchdog_acts(tmp_path):
    """A human watching a status panel should see a stall coming."""
    feed = make_feed(tmp_path)
    assert feed.status()["silenceSeconds"] is None, "nothing received yet"

    seed(feed)
    status = feed.status()
    assert status["silenceSeconds"] is not None
    assert status["silenceSeconds"] < 1.0
    assert status["stalls"] == 0


# ---------------------------------------------------------------------------
# Behind the venue
# ---------------------------------------------------------------------------
#
# The other failure that raises nothing: every message arrives, in sequence,
# 50 s after BloFin stamped it (seen live on two instruments at once). The stall
# watchdog cannot see it, because the socket is never silent.

NOW_MS = 1_789_129_800_000


def at(monkeypatch, ms):
    monkeypatch.setattr("perpkit.ingest.time.time", lambda: ms / 1000.0)


def test_behind_is_receipt_minus_the_newest_stamp_on_the_socket(tmp_path, monkeypatch):
    """A book older than a trade already delivered is measured against the
    trade: the socket is only as far behind as its freshest message."""
    feed = make_feed(tmp_path)
    at(monkeypatch, NOW_MS)
    feed.handle_message(books([[100.0, 5]], [[100.5, 5]], seq=1, action="snapshot",
                              ts=NOW_MS - 300))
    assert feed.behind_ms == 300
    feed.handle_message({"arg": {"channel": "trades"},
                         "data": [{"price": "100.2", "size": "1", "side": "buy",
                                   "ts": str(NOW_MS - 100)}]})
    assert feed.behind_ms == 100
    feed.handle_message(books([[100.0, 6]], [], seq=2, prev=1, ts=NOW_MS - 250))
    assert feed.behind_ms == 100
    feed.handle_message({"arg": {"channel": "funding-rate"},
                         "data": [{"fundingRate": "0.0002", "fundingTime": "1"}]})
    assert feed.behind_ms == 100, "an unstamped push measures nothing"
    assert feed.status()["maxBehindVenueMs"] == 300


def test_an_episode_prints_once_at_start_and_once_when_caught_up(tmp_path, monkeypatch, capsys):
    feed = make_feed(tmp_path)
    at(monkeypatch, NOW_MS)
    feed.handle_message(books([[100.0, 5]], [[100.5, 5]], seq=1, action="snapshot",
                              ts=NOW_MS - 6_000))
    # Received 30 s on: the freshest stamp is still the snapshot's (-6 s), so
    # the socket is 36 s behind, not the 50 s this older update alone says.
    at(monkeypatch, NOW_MS + 30_000)
    feed.handle_message(books([[100.0, 6]], [], seq=2, prev=1, ts=NOW_MS - 20_000))
    # Under the threshold but not under a second: still the same episode.
    at(monkeypatch, NOW_MS + 60_000)
    feed.handle_message(books([[100.0, 7]], [], seq=3, prev=2, ts=NOW_MS + 57_000))
    at(monkeypatch, NOW_MS + 90_000)
    feed.handle_message(books([[100.0, 8]], [], seq=4, prev=3, ts=NOW_MS + 89_500))

    lines = [line for line in capsys.readouterr().out.splitlines() if "venue" in line or "caught up" in line]
    assert lines == [
        "Microstructure feed behind the venue: BTC-USDT is receiving data BloFin "
        "stamped 6.0s ago (socket open, in sequence).",
        "Microstructure feed caught up: BTC-USDT after 90s, peak 36.0s behind.",
    ]
    assert feed.behind_episodes == 1
    assert feed.status()["behindVenueMs"] == 500
    assert feed.max_behind_ms == 36_000


# ---- a resync must not leave the old client running ------------------------
#
# A known leak: a long-running record.py grows by gigabytes. On Python 3.11
# `asyncio.wait_for` swallows a cancellation that lands after the awaited
# `recv()` has already completed, so the SDK's `close()` - which only cancels
# its receiver task - sometimes left that task running. The receiver then saw
# its socket closed, ran the SDK's own `_reconnect`, re-subscribed and kept
# putting a whole instrument's feed into an unbounded `_messageQueue` that no
# one would ever read again. Each feed resync (stall, gap, crossed book) is a
# chance to leak one more.


class SwallowingSdkClient:
    """The SDK client's shape, including the Python 3.11 race, made certain.

    The receiver swallows the first cancel as `wait_for` does when `recv()`
    had already completed, and reconnects on a closed socket whenever it still
    holds subscriptions, as `BlofinWsClient._handleDisconnect` does.
    """

    instances = []

    def __init__(self, isDemo=False):
        self._subscriptions = set()
        self._messageQueue = asyncio.Queue()
        self._receiverTask = None
        self._heartbeatTask = None
        self._ws = None
        self.socket_open = False
        self.reconnects = 0
        self.swallow_next_cancel = True
        SwallowingSdkClient.instances.append(self)

    async def connect(self):
        self.socket_open = True
        self._receiverTask = asyncio.create_task(self._receiver())

    async def _subscribe(self, channel, inst_id):
        self._subscriptions.add(f"{channel}:{inst_id}")

    async def subscribeOrderBook(self, inst_id, depth="books"):
        await self._subscribe(depth, inst_id)

    async def subscribeTrades(self, inst_id):
        await self._subscribe("trades", inst_id)

    async def subscribeFundingRate(self, inst_id):
        await self._subscribe("funding-rate", inst_id)

    async def subscribeTickers(self, inst_id):
        await self._subscribe("tickers", inst_id)

    async def subscribeCandles(self, inst_id, bar):
        await self._subscribe(f"candle{bar}", inst_id)

    async def subscribeOrders(self, inst_id):
        await self._subscribe("orders", inst_id)

    async def _receiver(self):
        while True:
            try:
                await asyncio.sleep(0.001)          # recv()
            except asyncio.CancelledError:
                if self.swallow_next_cancel:
                    self.swallow_next_cancel = False
                    continue
                raise
            if not self.socket_open:
                if not self._subscriptions:
                    break                           # _handleDisconnect -> False
                self.socket_open = True             # _reconnect succeeded
                self.reconnects += 1
            # An update with no snapshot: the feed's book is not ready, so the
            # feed asks for a resync on the first message.
            await self._messageQueue.put({
                "arg": {"channel": "books"}, "action": "update",
                "data": {"bids": [], "asks": [], "ts": "1", "seqId": "2",
                         "prevSeqId": "1"},
            })

    async def listen(self):
        while True:
            yield await self._messageQueue.get()

    async def close(self):
        if self._receiverTask:
            self._receiverTask.cancel()
        self.socket_open = False


def test_a_resync_leaves_nothing_of_the_old_client_running(tmp_path, monkeypatch):
    import perpkit.ingest as ingest

    SwallowingSdkClient.instances = []
    monkeypatch.setattr(ingest, "BlofinWsPublicClient", SwallowingSdkClient)
    feed = MicrostructureFeed("BTC-USDT", data_dir=tmp_path, record=False,
                              record_raw=False)

    async def scenario():
        run = asyncio.create_task(feed.run())
        await asyncio.sleep(0.3)   # one resync; the reconnect backoff is 1 s
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        await asyncio.sleep(0.05)
        old = SwallowingSdkClient.instances[0]
        return old._receiverTask.done(), old.reconnects

    receiver_done, reconnects = asyncio.run(scenario())

    assert len(SwallowingSdkClient.instances) == 1
    assert receiver_done, "the old client's receiver outlived the resync"
    assert reconnects == 0, "the old client re-subscribed after being closed"
