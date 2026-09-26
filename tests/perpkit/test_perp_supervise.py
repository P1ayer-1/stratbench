"""No supervised loop may take the recorder down with it.

`asyncio.gather` propagates the first exception it sees, so one unhandled
error in any loop (a dropped keep-alive connection, which every long-lived
connection pool meets eventually) would end the whole process, recorders and
raw archive included. Raw events are irreplaceable, because that market moment
is over. These tests pin `perpkit.supervise`: restart on exception, re-raise
cancellation, reset backoff after a healthy run.

Each test drives its own event loop.
"""

import asyncio

import pytest

from perpkit import supervise as server


def run(coro, timeout=5.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


class Boom(RuntimeError):
    """Stands in for RemoteDisconnected and everything like it."""


# ---------------------------------------------------------------------------
# supervise
# ---------------------------------------------------------------------------


def test_a_crashing_loop_is_restarted_rather_than_propagating(monkeypatch):
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 0.0)
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 4:
            raise Boom("remote end closed connection")
        await asyncio.sleep(10)  # settled; will be cancelled by the timeout

    async def drive():
        task = asyncio.create_task(server.supervise("flaky", flaky))
        while len(attempts) < 4:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(drive())
    assert len(attempts) == 4


def test_one_loop_crashing_does_not_stop_another(monkeypatch):
    """One loop dying must not stop another: the recorder keeps going."""
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 0.0)
    collected = []

    async def side_loop():
        raise Boom("RemoteDisconnected")

    async def collection_loop():
        while True:
            collected.append(1)
            await asyncio.sleep(0.01)

    async def drive():
        # `gather` already returns a future; this mirrors a recorder that
        # gathers several supervised loops.
        gathered = asyncio.gather(
            server.supervise("side", side_loop),
            server.supervise("collection", collection_loop),
        )
        await asyncio.sleep(0.2)
        still_running = not gathered.done()
        gathered.cancel()
        try:
            await gathered
        except asyncio.CancelledError:
            pass
        return still_running

    assert run(drive()) is True
    # The collection loop kept ticking throughout, despite the other one
    # failing on every single restart.
    assert len(collected) > 5


def test_cancellation_is_not_swallowed():
    """Ctrl+C must still stop the process. A supervisor that catches
    CancelledError would make it unkillable."""
    started = asyncio.Event()

    async def forever():
        started.set()
        await asyncio.sleep(30)

    async def drive():
        task = asyncio.create_task(server.supervise("forever", forever))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(drive())


def test_a_loop_that_returns_is_restarted_not_silently_dropped(monkeypatch):
    # These loops are all `while True`. One returning means something is
    # wrong, and quietly losing it is how you find out days later that the
    # recorder stopped.
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 0.0)
    calls = []

    async def returns_immediately():
        calls.append(1)

    async def drive():
        task = asyncio.create_task(
            server.supervise("quitter", returns_immediately))
        while len(calls) < 3:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(drive())
    assert len(calls) >= 3


def test_backoff_resets_after_a_healthy_run(monkeypatch):
    # A loop that ran for hours and then hit one bad response should retry
    # promptly, not inherit the backoff from last night's failure.
    monkeypatch.setattr(server, "HEALTHY_AFTER_SECONDS", 0.0)
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 30.0)
    delays = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        # `server.asyncio` IS the global module, so this patch is global --
        # call the captured original or fake_sleep recurses into itself.
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    async def always_fails():
        raise Boom("nope")

    async def drive():
        task = asyncio.create_task(server.supervise("flaky", always_fails))
        while len(delays) < 4:
            await real_sleep(0)   # not the patched one; it would be recorded
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    # Every attempt counts as "healthy" under this monkeypatch, so the backoff
    # is reset each time instead of doubling away.
    assert delays[:4] == [1.0, 1.0, 1.0, 1.0]


def test_backoff_grows_when_failures_are_immediate(monkeypatch):
    monkeypatch.setattr(server, "HEALTHY_AFTER_SECONDS", 1e9)
    monkeypatch.setattr(server, "MAX_RESTART_BACKOFF_SECONDS", 8.0)
    delays = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        # `server.asyncio` IS the global module, so this patch is global --
        # call the captured original or fake_sleep recurses into itself.
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    async def always_fails():
        raise Boom("nope")

    async def drive():
        task = asyncio.create_task(server.supervise("flaky", always_fails))
        while len(delays) < 6:
            await real_sleep(0)   # not the patched one
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert delays[:5] == [1.0, 2.0, 4.0, 8.0, 8.0]  # doubles, then caps


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def test_log_lines_carry_a_utc_timestamp_and_flush(capsys):
    # An overnight failure is only diagnosable if its line can be lined up
    # against the exchange's logs, and only survives the crash if it is
    # flushed rather than sitting in a block buffer.
    server.log("something broke")
    out = capsys.readouterr().out.strip()
    assert out.endswith("something broke")
    assert out.count(":") >= 2
    assert "Z  " in out
