"""Two sockets, one stream: no duplicates, no gap when one drops."""

import asyncio
import json

from predkit.venues.polymarket import redundant_stream


class FakeSocket:
    """An async context manager yielding frames from a script; a `RuntimeError`
    entry raises mid-stream, like a 1013 close."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.script:
            await asyncio.sleep(10)          # a live socket with nothing to say hangs
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        await asyncio.sleep(0)
        return item


def frame(i, kind="price_change"):
    return json.dumps({"event_type": kind, "market": "0xc", "i": i, "timestamp": str(1000 + i)})


async def collect(stream, n):
    out = []
    async for item in stream:
        out.append(item)
        if len(out) >= n:
            break
    return out


async def test_frames_seen_on_both_sockets_are_delivered_once():
    frames = [frame(i) for i in range(20)]
    sockets = iter([FakeSocket(frames), FakeSocket(frames)])
    got = await collect(redundant_stream(lambda: next(sockets), "sub", connections=2, label="t"), 20)
    assert [json.loads(raw)["i"] for _, raw in got] == list(range(20))
    assert all(kind == "price_change" for kind, _ in got)


async def test_a_dropped_socket_is_reopened_and_the_other_carries_the_gap():
    frames = [frame(i) for i in range(30)]
    first = FakeSocket(frames[:10] + [RuntimeError("received 1013 slow consumer")])
    replacement = FakeSocket(frames[10:])           # a reopened socket delivers the current frames
    second = FakeSocket(frames)
    sockets = iter([first, second, replacement])
    redundant_stream.drops.clear()

    async def no_sleep(_):
        await asyncio.sleep(0)

    got = await collect(redundant_stream(lambda: next(sockets), "sub", connections=2, label="t", sleep=no_sleep), 30)
    assert [json.loads(raw)["i"] for _, raw in got] == list(range(30))
    assert redundant_stream.drops["t"] == 1
    assert replacement.sent == ["sub"]                 # the reopened socket re-subscribed


class Ahead(FakeSocket):
    """Sets `done` once its script is delivered, so a `Behind` socket can
    start only then: the lagging socket, a whole script later."""

    def __init__(self, script, done):
        super().__init__(script)
        self.done = done

    async def __anext__(self):
        if not self.script:
            self.done.set()
        return await super().__anext__()


class Behind(FakeSocket):
    def __init__(self, script, after):
        super().__init__(script)
        self.after = after

    async def __anext__(self):
        await self.after.wait()
        return await super().__anext__()


async def test_a_copy_thousands_of_frames_late_is_still_dropped():
    """Measured on live recordings: the lagging socket's copies landed 5-13 s
    after the first, up to ~18,000 frames at the 1,400/s peak, so a
    4,096-frame window let many of them into the archive. Here the copies
    come 6,000 frames late and only the one new frame gets through."""
    frames = [frame(i) for i in range(6001)]
    done = asyncio.Event()
    sockets = iter([Ahead(frames[:6000], done), Behind(frames, done)])
    got = await collect(redundant_stream(lambda: next(sockets), "sub", connections=2, label="t"), 6001)
    assert [json.loads(raw)["i"] for _, raw in got] == list(range(6001))


async def test_the_window_is_bounded():
    """Memory stays flat on a long window: a copy older than the window is
    delivered again (replay drops it; `replay.iter_market_events`)."""
    frames = [frame(i) for i in range(5)]
    done = asyncio.Event()
    sockets = iter([Ahead(frames[:4], done), Behind(frames[:1] + frames[4:], done)])
    got = await collect(redundant_stream(lambda: next(sockets), "sub", connections=2, label="t",
                                         dedupe_window=3), 6)
    assert [json.loads(raw)["i"] for _, raw in got] == [0, 1, 2, 3, 0, 4]


async def test_connect_batches_are_split_and_never_deduplicated():
    batch = json.dumps([{"event_type": "book", "asset_id": "Y", "bids": []}, {"event_type": "book", "asset_id": "N", "bids": []}])
    sockets = iter([FakeSocket([batch, frame(1)]), FakeSocket([batch, frame(1)])])
    got = await collect(redundant_stream(lambda: next(sockets), "sub", connections=2, label="t"), 5)
    kinds = [kind for kind, _ in got]
    assert kinds.count("book") == 4 and kinds.count("price_change") == 1
