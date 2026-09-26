"""Loops that cannot die or go silent."""

import asyncio

import pytest

from predkit.supervise import FeedStalled, each_with_deadline, supervise


async def test_supervise_restarts_a_loop_that_raises():
    runs = []

    async def start():
        runs.append(1)
        raise RuntimeError("dropped keep-alive")

    async def no_sleep(_):
        pass

    await supervise("t", start, sleep=no_sleep, max_restarts=3)
    assert len(runs) == 4          # the first run plus three restarts


async def test_supervise_reraises_cancellation():
    """Ctrl+C is not a failure; swallowing it makes the process unkillable."""
    async def start():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await supervise("t", start, max_restarts=0)


async def test_a_silent_feed_raises_instead_of_blocking():
    async def silent():
        yield 1
        await asyncio.sleep(10)
        yield 2

    seen = []
    with pytest.raises(FeedStalled):
        async for item in each_with_deadline(silent(), timeout_s=0.05):
            seen.append(item)
    assert seen == [1]


async def test_a_finished_feed_just_ends():
    async def three():
        for i in range(3):
            yield i

    assert [i async for i in each_with_deadline(three(), timeout_s=1)] == [0, 1, 2]
