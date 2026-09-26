"""The live microstructure feed: book + trades + funding -> features -> disk.

This runs as its own websocket connection, separate from any other stream a
dashboard or trading tool might hold. That separation is deliberate: the
microstructure feed is the one that will be churning, resyncing, and getting
restarted while a strategy is developed, and it should not be able to take
anything else down, nor be taken down by it.

Recovery behaviour: if the order book detects a sequence gap it marks itself
stale, and this loop tears the connection down and reconnects to force a fresh
snapshot. Reconnecting is cheap; trading on a silently-wrong book is not.

Behind the venue, and not by our doing
--------------------------------------
A socket can stay open, deliver every message in sequence, and still deliver
it late. On a live recorder, two instruments' sockets have been seen running
50 s behind BloFin's own stamps for ten minutes while two others, in the same
process and event loop, carried more bytes a second under a second late; when
one of the lagging sockets died 43 s behind, one message was left in this
process and the new socket was 273 ms fresh. The backlog was upstream -
BloFin's gateway or the path to it - so nothing here can drain it faster. What
this process can do is say so: `behind_ms` is receipt minus the newest
exchange stamp the socket has delivered, and an episode past
`behind_log_s` is printed when it starts and when it ends. Reconnecting on it
was rejected: the queued messages are the archive, and only the venue has them.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from blofin.websocket_client import BlofinWsPublicClient

from .features import FeatureEngine, FeatureSnapshot
from .orderbook import OrderBook
from .rawlog import RawEventLog
from .recorder import FeatureRecorder, LabelConfig
from .tape import TradeTape


async def close_ws_client(client: Any, *, rounds: int = 5) -> bool:
    """Close an SDK websocket client so that no part of it keeps running.

    `BlofinWsClient.close()` cancels its receiver task once and trusts the
    cancel. On Python 3.11 that trust is misplaced: `asyncio.wait_for`, which
    the receiver wraps round every `recv()`, swallows a cancellation that
    lands after `recv()` has completed, which on a busy socket is often. The
    surviving receiver then finds its socket closed, runs the SDK's own
    `_reconnect`, re-subscribes, and puts the instrument's whole feed into an
    unbounded `_messageQueue` that nothing reads again, so a long-running
    `record.py` grows by gigabytes. Each resync (stall, sequence gap, crossed
    book) is a chance to orphan one more client. Reproduced with a fake busy
    socket: 27 of 40 closed clients were still receiving and had reconnected.

    Empty the subscriptions first so that no reconnect path has anything to
    restore. A receiver that survives the cancel then exits on its own at the
    closed socket. Then cancel and await the tasks until they are done, close
    whatever socket a reconnect already in flight opened, and drop the queue.
    Returns False if a task is somehow still alive after `rounds` tries.
    """
    subscriptions = getattr(client, "_subscriptions", None)
    if subscriptions is not None:
        subscriptions.clear()
    try:
        await client.close()
    except Exception:
        pass

    def live_tasks():
        tasks = (getattr(client, "_receiverTask", None),
                 getattr(client, "_heartbeatTask", None))
        return [task for task in tasks if task is not None and not task.done()]

    for _ in range(rounds):
        pending = live_tasks()
        if not pending:
            break
        for task in pending:
            task.cancel()
        await asyncio.wait(pending, timeout=1.0)

    ws = getattr(client, "_ws", None)
    if ws is not None:
        try:
            await ws.close()
        except Exception:
            pass
    queue = getattr(client, "_messageQueue", None)
    if queue is not None:
        while not queue.empty():
            queue.get_nowait()
    return not live_tasks()


class MicrostructureFeed:
    """Owns the book, the tape, the feature engine and the recorder.

    `latest` always holds the most recent computed snapshot, so any consumer
    (a dashboard, a signal engine) can read current features without
    subscribing to anything.
    """

    def __init__(
        self,
        inst_id: str,
        *,
        use_demo: bool = False,
        book_depth: str = "books",
        data_dir: Optional[Path] = None,
        record: bool = True,
        sample_interval_ms: int = 250,
        label_config: Optional[LabelConfig] = None,
        tape_window_seconds: float = 60.0,
        record_raw: bool = True,
        stall_timeout_s: float = 30.0,
        behind_log_s: float = 5.0,
        on_snapshot: Optional[Callable[[FeatureSnapshot], Any]] = None,
    ):
        self.inst_id = inst_id
        self.use_demo = use_demo
        self.book_depth = book_depth
        self.stall_timeout_s = stall_timeout_s
        self.behind_log_ms = int(behind_log_s * 1000)
        self.on_snapshot = on_snapshot

        self.book = OrderBook()
        self.tape = TradeTape(window_seconds=tape_window_seconds)
        self.engine = FeatureEngine()

        # BOTH writers are scoped to the instrument, and that is not tidiness.
        # The rows for two symbols are structurally identical - same columns,
        # same order, same dtypes - and differ only in which instrument they
        # describe. Written to a shared path they concatenate into one matrix
        # that no schema check can object to, because nothing about the schema
        # is wrong. The only thing that separates them is where they live.
        self.data_root = Path(data_dir or Path("data"))
        self.instrument_dir = self.data_root / inst_id
        self.recorder = FeatureRecorder(
            self.instrument_dir,
            label_config=label_config,
            sample_interval_ms=sample_interval_ms,
            enabled=record,
        )
        # The raw archive. Written before any parsing, so a bug in the feature
        # code can never corrupt or lose the source data — the archive can
        # always be replayed once the bug is fixed.
        self.raw_log = RawEventLog(self.instrument_dir, enabled=record_raw)

        self.latest: FeatureSnapshot = FeatureSnapshot()
        self.connected = False
        self.reconnects = 0
        self.stalls = 0
        self.messages = 0
        self.started_at = time.time()
        # Wall-clock of the last message off the wire. `_stream()` is what
        # acts on it; this is here so `status()` can show the silence to a
        # human before the watchdog decides.
        self.last_message_at: Optional[float] = None

        # How far behind the venue this socket is running; see the module
        # docstring. `_freshest_ts` is the newest exchange stamp delivered on
        # the current connection.
        self._freshest_ts = 0
        self.behind_ms: Optional[int] = None
        self.max_behind_ms = 0
        self.behind_episodes = 0
        self._episode_started_at: Optional[float] = None
        self._episode_peak_ms = 0

        # A single crossed update can just be a mid-update race, so tolerate a
        # few in a row before forcing an expensive resubscribe.
        self.crossed_events = 0
        self.crossed_tolerance = 3

    # ---- message handling ------------------------------------------------

    def handle_message(self, message: Dict[str, Any]) -> bool:
        """Route one websocket message. Returns True if a resync is needed."""
        channel = message.get("arg", {}).get("channel", "")
        data = message.get("data")
        if data is None:
            return False

        self.messages += 1
        self.last_message_at = time.time()
        # Archive first, parse second. If anything below throws, the event is
        # already safely on disk and can be replayed after the fix.
        self.raw_log.write(channel, message)
        self._track_behind(channel, data, self.last_message_at)

        if channel in ("books", "books5"):
            self.book.apply(message)
            if not self.book.ready:
                return True  # sequence gap — caller must reconnect
            # A crossed book (bid >= ask) means our mirror has drifted from the
            # exchange's: we have kept a level that was actually removed, or
            # missed one. Features computed from it are invalid, and unlike a
            # sequence gap nothing will repair it on its own — the stale level
            # simply sits there. So treat a persistently crossed book as a
            # desync and force a fresh snapshot.
            if self.book.is_crossed():
                self.crossed_events += 1
                if self.crossed_events >= self.crossed_tolerance:
                    self.book.mark_stale(
                        f"book crossed for {self.crossed_events} consecutive updates"
                    )
                    self.crossed_events = 0
                    return True
                return False
            self.crossed_events = 0
            self.engine.on_book_event(self.book)
            self._emit()
        elif channel == "trades":
            if self.tape.add_message(data):
                self._emit()
        elif channel == "funding-rate":
            self.engine.on_funding(data)

        return False

    @staticmethod
    def _venue_ts(channel: str, data: Any) -> int:
        """BloFin's stamp on a push: the book's `ts`, or the newest trade's. 0 if none."""
        try:
            if channel in ("books", "books5"):
                return int(data["ts"])
            if channel == "trades":
                return max(int(row["ts"]) for row in data)
        except (KeyError, TypeError, ValueError):
            pass
        return 0

    def _track_behind(self, channel: str, data: Any, now: float) -> None:
        stamp = self._venue_ts(channel, data)
        if stamp <= 0:
            return
        self._freshest_ts = max(self._freshest_ts, stamp)
        behind = int(now * 1000) - self._freshest_ts
        self.behind_ms = behind
        self.max_behind_ms = max(self.max_behind_ms, behind)

        if self._episode_started_at is None:
            if behind >= self.behind_log_ms:
                self._episode_started_at = now
                self._episode_peak_ms = behind
                self.behind_episodes += 1
                print(f"Microstructure feed behind the venue: {self.inst_id} is "
                      f"receiving data BloFin stamped {behind / 1000:.1f}s ago "
                      f"(socket open, in sequence).")
            return
        self._episode_peak_ms = max(self._episode_peak_ms, behind)
        # Hysteresis: back under a second, not merely under the threshold,
        # so one episode prints two lines rather than a flicker.
        if behind < 1000:
            print(f"Microstructure feed caught up: {self.inst_id} after "
                  f"{now - self._episode_started_at:.0f}s, peak "
                  f"{self._episode_peak_ms / 1000:.1f}s behind.")
            self._episode_started_at = None

    def _emit(self) -> None:
        snapshot = self.engine.compute(self.book, self.tape)
        self.latest = snapshot
        self.recorder.observe(snapshot)
        if self.on_snapshot is not None:
            try:
                self.on_snapshot(snapshot)
            except Exception as exc:
                print(f"Feature consumer error: {exc}")

    # ---- the loop --------------------------------------------------------

    async def run(self) -> None:
        """Connect, subscribe, and stream until cancelled. Reconnects forever."""
        backoff = 1.0
        while True:
            client = BlofinWsPublicClient(isDemo=self.use_demo)
            try:
                await client.connect()
                await client.subscribeOrderBook(self.inst_id, depth=self.book_depth)
                await client.subscribeTrades(self.inst_id)
                await client.subscribeFundingRate(self.inst_id)

                self.connected = True
                backoff = 1.0
                mode = "demo" if self.use_demo else "production"
                print(
                    f"Microstructure feed connected ({mode}): "
                    f"{self.inst_id} [{self.book_depth}, trades, funding-rate]"
                )

                self.last_message_at = time.time()
                if await self._stream(client):
                    print(
                        "Order book desync "
                        f"({self.book.last_gap_reason}) - resyncing."
                    )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Microstructure feed error: {exc}")
            finally:
                self.connected = False
                self.book.reset()
                # A new connection is judged on its own stamps; the episode,
                # if any, stays open until one of them is fresh.
                self._freshest_ts = 0
                self.reconnects += 1
                if not await close_ws_client(client):
                    print(f"Microstructure feed: {self.inst_id}'s old "
                          f"websocket client is still running after close.")

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _stream(self, client: Any) -> bool:
        """Pump messages until the feed desyncs, ends, or goes silent.

        Returns True if the book desynced (the caller reports the reason).

        The silence check is the point of this method existing. `async for
        message in client.listen()` has no deadline, so a subscription that
        stops delivering while the socket stays open blocks here forever: no
        exception, so `supervise` sees a healthy loop, and the recorder writes
        nothing until someone notices by hand. That is strictly worse than a
        crash: a crash at least leaves a mark.

        Cancelling `__anext__` mid-await is safe precisely because we never
        resume the iterator afterwards - the caller closes the client and
        reconnects, which is the only honest response to a feed we can no
        longer account for.
        """
        messages = client.listen().__aiter__()
        while True:
            try:
                message = await asyncio.wait_for(
                    messages.__anext__(), timeout=self.stall_timeout_s
                )
            except asyncio.TimeoutError:
                self.stalls += 1
                print(
                    f"Microstructure feed stalled: no message in "
                    f"{self.stall_timeout_s:g}s (socket still open) - "
                    f"reconnecting."
                )
                return False
            except StopAsyncIteration:
                return False

            if self.handle_message(message):
                return True

    def close(self) -> None:
        self.recorder.close()
        self.raw_log.close()

    # ---- reporting -------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        bid, ask = self.book.best_bid_ask()
        return {
            "connected": self.connected,
            "instId": self.inst_id,
            "bookReady": self.book.is_ready,
            "bookLevels": {"bids": len(self.book.bids), "asks": len(self.book.asks)},
            "bestBid": bid,
            "bestAsk": ask,
            "seqId": self.book.seq_id,
            "resyncs": self.book.resync_count,
            "reconnects": self.reconnects,
            "stalls": self.stalls,
            "stallTimeoutSeconds": self.stall_timeout_s,
            "silenceSeconds": (
                None if self.last_message_at is None
                else round(time.time() - self.last_message_at, 1)
            ),
            "behindVenueMs": self.behind_ms,
            "maxBehindVenueMs": self.max_behind_ms,
            "behindEpisodes": self.behind_episodes,
            "messages": self.messages,
            "uptimeSeconds": round(time.time() - self.started_at, 1),
            "tape": self.tape.snapshot(),
            "recorder": self.recorder.stats(),
            "rawLog": self.raw_log.stats(),
        }
