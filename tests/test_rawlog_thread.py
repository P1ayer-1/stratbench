"""The writer thread: order kept, flush synchronous, failures counted, the
caller never touches a file."""

import threading
from datetime import datetime, timezone

from predkit.rawlog import RawEventLog, iter_directory

CLOCK = datetime(2026, 9, 13, 14, 13, tzinfo=timezone.utc).timestamp()


def test_threaded_writer_keeps_arrival_order_and_counts(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    for i in range(2000):
        log.write("price_change" if i % 3 else "book", {"i": i})
    log.close()
    assert log.lines_written == 2000 and log.lines_on_disk == 2000
    events = sorted(list(iter_directory(tmp_path / "raw", "book", warn=False))
                    + list(iter_directory(tmp_path / "raw", "price_change", warn=False)),
                    key=lambda e: (e[0], e[1]))
    assert [m["i"] for _, _, m in events] == list(range(2000))
    assert [n for _, n, _ in events] == list(range(1, 2001))     # n assigned on the caller's thread


def test_flush_returns_only_once_the_thread_has_written(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK, flush_lines=10 ** 9, flush_seconds=10 ** 9)
    for i in range(500):
        log.write("book", {"i": i})
    log.flush()
    assert log.lines_on_disk == 500
    assert len(list(iter_directory(tmp_path / "raw", "book", warn=False))) == 500
    log.close()


def test_the_caller_thread_never_opens_the_file(tmp_path, monkeypatch):
    writer_threads = set()
    original = RawEventLog._create

    def spy(self, directory, channel, hour):
        writer_threads.add(threading.current_thread().name)
        return original(self, directory, channel, hour)

    monkeypatch.setattr(RawEventLog, "_create", spy)
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("book", {})
    log.close()
    assert writer_threads == {"rawlog-writer"}


def test_a_write_failure_is_counted_and_the_thread_carries_on(tmp_path, monkeypatch, capsys):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    calls = {"n": 0}
    original = RawEventLog._handle_for

    def flaky(self, channel, now):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk hiccup")
        return original(self, channel, now)

    monkeypatch.setattr(RawEventLog, "_handle_for", flaky)
    log.write("book", {"i": 1})
    log.write("book", {"i": 2})
    log.close()
    assert log.write_failures == 1 and log.lines_on_disk == 1
    assert [m for _, _, m in iter_directory(tmp_path / "raw", "book", warn=False)] == [{"i": 2}]
    assert "message lost" in capsys.readouterr().err


def test_unthreaded_mode_is_still_available_for_the_kill_tests(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK, threaded=False)
    log.write("book", {"i": 1})
    assert log.lines_on_disk == 1 and log._thread is None
    log.close()
    assert log.stats()["queued"] == 0
