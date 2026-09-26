"""Keep long-running loops alive, and log so a failure can be lined up later.

A recorder that dies at 03:00 costs the hours nobody was awake for, and the
raw archive is the half that cannot be re-collected: that market moment is
gone. So no loop is permitted to take another down with it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
import time
import traceback
from typing import Awaitable, Callable

# A supervised loop that has been alive at least this long is treated as
# healthy, so its restart backoff resets. Without it, a loop that runs fine for
# hours and then hits one bad response would inherit the backoff from an old
# failure and sit out a minute for no reason.
HEALTHY_AFTER_SECONDS = 60.0

# Cap on the restart backoff. Long enough not to hammer a venue that is down,
# short enough that an unattended overnight run recovers promptly once it is
# back.
MAX_RESTART_BACKOFF_SECONDS = 60.0


def log(message: str) -> None:
    """Print with a UTC timestamp, unbuffered.

    Both halves matter for a process meant to run for days. Without the
    timestamp there is no way to line a failure up against the exchange's own
    status page; without `flush`, a redirected stdout is block-buffered and
    the last few KB, which is exactly the part describing the failure, is lost
    when the process dies.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp}Z  {message}", flush=True)


async def supervise(name: str, start: Callable[[], Awaitable[None]]) -> None:
    """Run a background loop forever, restarting it if it ever falls over.

    `asyncio.gather` over raw loops propagates the first exception it sees, so
    an unhandled error in ANY loop (one dropped HTTP connection on a
    convenience endpoint, say) ends the whole process, recorders included.
    Each loop runs under this instead.

    `start` is a factory rather than a coroutine because a coroutine can only
    be awaited once, and this may need to build several.

    `CancelledError` is re-raised untouched: that is Ctrl+C and shutdown, not
    a failure, and swallowing it would make the process unkillable.
    """
    backoff = 1.0
    restarts = 0
    while True:
        began = time.monotonic()
        try:
            await start()
            log(f"[{name}] returned unexpectedly (it should run forever).")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if time.monotonic() - began >= HEALTHY_AFTER_SECONDS:
                backoff = 1.0
            restarts += 1
            log(f"[{name}] crashed: {type(exc).__name__}: {exc}")
            log(f"[{name}] restart #{restarts} in {backoff:.0f}s. "
                f"Traceback follows.")
            traceback.print_exc()
            sys.stdout.flush()
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, MAX_RESTART_BACKOFF_SECONDS)
