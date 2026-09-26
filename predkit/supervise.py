"""Loops that cannot die or go silent.

Two failures that cost real recordings, and the two shapes that answer them:

**A loop must not take another loop down.** `asyncio.gather` propagates the
first exception it sees, so one dropped keep-alive on a chart endpoint
ended a seven-hour recording run, archive and all. `supervise`
restarts any loop that falls over, with backoff, and re-raises only
`CancelledError` (Ctrl+C and shutdown are not failures; swallowing them makes
a process unkillable).

**A stalled feed raises nothing.** `async for message in socket` has no
deadline, so a subscription that stops delivering while the socket stays
open blocks forever: no exception, `supervise` sees a healthy loop, and the
recorder writes nothing until a human notices. That is strictly worse than a
crash, which at least leaves a mark. `each_with_deadline` wraps every await
on a feed in a timeout and raises `FeedStalled`, which the caller answers by
reconnecting. Cancelling the iterator mid-await is safe precisely because it
is never resumed afterwards.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
import time
import traceback
from typing import AsyncIterator, Awaitable, Callable, TypeVar

# A supervised loop alive at least this long is healthy, so its backoff
# resets; a loop that runs six hours and then hits one bad response must not
# inherit last night's backoff.
HEALTHY_AFTER_SECONDS = 60.0
# Long enough not to hammer a venue that is down, short enough that an
# unattended run recovers promptly once it is back.
MAX_RESTART_BACKOFF_SECONDS = 60.0

T = TypeVar("T")


def log(message: str) -> None:
    """Print with a UTC timestamp, unbuffered.

    Both halves matter for a process meant to run for days: the timestamp
    lines a failure up against the venue's own logs, and without `flush` a
    redirected stdout is block-buffered and the last few KB, which is the
    part describing the failure, is lost when the process dies.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp}Z  {message}", flush=True)


async def supervise(name: str, start: Callable[[], Awaitable[None]], *,
                    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                    clock: Callable[[], float] = time.monotonic,
                    max_restarts: int = -1) -> None:
    """Run a loop forever, restarting it if it ever falls over.

    `start` is a factory rather than a coroutine because a coroutine can only
    be awaited once, and this may need to build several. `max_restarts` is
    for tests; -1 is forever.
    """
    backoff = 1.0
    restarts = 0
    while True:
        began = clock()
        try:
            await start()
            log(f"[{name}] returned unexpectedly (it should run forever).")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if clock() - began >= HEALTHY_AFTER_SECONDS:
                backoff = 1.0
            restarts += 1
            log(f"[{name}] crashed: {type(exc).__name__}: {exc}")
            log(f"[{name}] restart #{restarts} in {backoff:.0f}s. Traceback follows.")
            traceback.print_exc()
            sys.stdout.flush()
        if 0 <= max_restarts < restarts:
            return
        await sleep(backoff)
        backoff = min(backoff * 2, MAX_RESTART_BACKOFF_SECONDS)


class FeedStalled(Exception):
    """No message inside the deadline while the connection stayed open."""


async def each_with_deadline(source: AsyncIterator[T], timeout_s: float) -> AsyncIterator[T]:
    """Yield from `source`, raising `FeedStalled` if any single message is
    late. The source is never resumed after a stall: the caller closes it
    and reconnects, which is the only honest response to a feed it can no
    longer account for."""
    iterator = source.__aiter__()
    while True:
        try:
            message = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise FeedStalled(f"no message in {timeout_s:g}s (connection still open)") from None
        except StopAsyncIteration:
            return
        yield message


__all__ = ["FeedStalled", "HEALTHY_AFTER_SECONDS", "MAX_RESTART_BACKOFF_SECONDS",
           "each_with_deadline", "log", "supervise"]
