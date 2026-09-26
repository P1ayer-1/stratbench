"""The recorder archives first, parses second, and treats silence as a drop."""

import asyncio

import pytest

from predkit.rawlog import iter_directory
from predkit.record import MarketRecorder, market_dir, reference_dir


class FakeFeed:
    name = "fake"

    def __init__(self, batches):
        self.batches = list(batches)
        self.sessions = 0

    async def stream(self, market_id):
        self.sessions += 1
        # A real socket that has nothing to say hangs; it does not return.
        batch = self.batches.pop(0) if self.batches else ["hang"]
        for item in batch:
            if item == "hang":
                await asyncio.sleep(10)
            else:
                yield item


def test_market_dir_is_one_per_venue_per_market(tmp_path):
    assert market_dir(tmp_path, "kalshi", "KXBTC15M-26SEP1218") == tmp_path / "kalshi" / "KXBTC15M-26SEP1218"
    assert reference_dir(tmp_path, "binance-BTCUSDT") == tmp_path / "_reference" / "binance-BTCUSDT"
    assert market_dir(tmp_path, "polymarket", "0xabc/../x").name == "0xabc_.._x"


async def test_archives_every_message_even_when_parsing_fails(tmp_path):
    feed = FakeFeed([[("book", {"a": 1}), ("trade", {"b": 2})]])
    failures = []

    def parse(channel, message):
        failures.append(channel)
        raise ValueError("bad")

    recorder = MarketRecorder(feed, "M", tmp_path, on_message=parse, stall_timeout_s=1)
    await recorder._session()
    recorder.close()
    assert recorder.parse_failures == 2
    assert [m for _, _, m in iter_directory(tmp_path / "raw", "book", warn=False)] == [{"a": 1}]
    assert [m for _, _, m in iter_directory(tmp_path / "raw", "trade", warn=False)] == [{"b": 2}]


async def test_a_stall_reconnects_and_is_counted(tmp_path):
    feed = FakeFeed([[("book", {"n": 1}), "hang"], [("book", {"n": 2})]])
    recorder = MarketRecorder(feed, "M", tmp_path, stall_timeout_s=0.05, sleep=_no_sleep)

    async def stop_after_two():
        while recorder.messages < 2:
            await asyncio.sleep(0.01)
        recorder.stop.set()

    await asyncio.gather(recorder.run(), stop_after_two())
    recorder.close()
    # One stall on the hang, plus possibly one more on the empty session
    # that ran before `stop` was seen; never zero, and never a hang.
    assert recorder.stalls >= 1
    assert feed.sessions >= 2
    assert [m for _, _, m in iter_directory(tmp_path / "raw", "book", warn=False)] == [{"n": 1}, {"n": 2}]


async def _no_sleep(_):
    pass
