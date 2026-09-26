"""Raw archive: write, rotate, read back, and survive every kind of tear.

Pinned with the measurements that motivated the design. The hard-kill tests
run a real child process and `os._exit` it, because a closed file and a
killed one look nothing alike on disk.
"""

import gzip
import json
import os
import random
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from predkit.rawlog import RawEventLog, channels_in, iter_directory, iter_events

ROOT = Path(__file__).resolve().parents[1]
# Pinned so every child writes into the same hour.
CLOCK = datetime(2026, 9, 12, 14, 13, tzinfo=timezone.utc).timestamp()


def message(seq):
    return {"event_type": "price_change", "changes": [{"price": "0.48", "side": "BUY", "size": str(seq)}],
            "timestamp": 1_789_000_000_000 + seq}


def test_writes_and_reads_back_identically(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    originals = [message(i) for i in range(1, 21)]
    for item in originals:
        log.write("price_change", item)
    log.close()
    assert [m for _, _, m in iter_directory(tmp_path / "raw", "price_change")] == originals


def test_receive_time_and_sequence_are_stored(tmp_path):
    """Feed lag cannot be reconstructed later, and two messages in one
    millisecond need `n` to recover their order."""
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("book", message(1))
    log.write("trade", message(2))
    log.close()
    (t1, n1, _), = iter_directory(tmp_path / "raw", "book")
    (t2, n2, _), = iter_directory(tmp_path / "raw", "trade")
    assert t1 == t2 == int(CLOCK * 1000)
    assert (n1, n2) == (1, 2)


def test_every_channel_is_archived_by_default(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("book", {})
    log.write("anything_else", {})
    log.close()
    assert channels_in(tmp_path / "raw") == ("anything_else", "book")


def test_a_restart_in_the_same_hour_writes_beside_not_into(tmp_path):
    first = RawEventLog(tmp_path, clock=lambda: CLOCK)
    first.write("book", message(1))
    first.close()
    second = RawEventLog(tmp_path, clock=lambda: CLOCK)
    second.write("book", message(2))
    second.close()
    names = [p.name for p in sorted((tmp_path / "raw").rglob("book-*.jsonl.gz"))]
    assert names == ["book-14.jsonl.gz", "book-14.r001.jsonl.gz"]
    assert len(list(iter_directory(tmp_path / "raw", "book"))) == 2


def test_truncated_final_line_costs_one_message(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    for i in range(1, 6):
        log.write("book", message(i))
    log.close()
    path = next((tmp_path / "raw").rglob("book-*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        content = handle.read()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content + '{"t":123,"m":{"incompl')
    assert len(list(iter_events(path, warn=False))) == 5


def trade(index):
    rng = random.Random(index)
    return {"event_type": "last_trade_price", "price": f"{rng.uniform(0.3, 0.7):.2f}",
            "size": str(rng.randrange(5, 500)), "side": rng.choice(["BUY", "SELL"]),
            "timestamp": 1_789_000_000_000 + index, "asset_id": str(rng.randrange(10 ** 30))}


WRITER = textwrap.dedent("""
    import os, sys
    root, repo, clock, first, count, unflushed, ending = sys.argv[1:8]
    sys.path[:0] = [repo, os.path.join(repo, "tests")]
    from predkit.rawlog import RawEventLog
    from test_rawlog import trade
    first, count, unflushed = int(first), int(count), int(unflushed)
    log = RawEventLog(root, flush_lines=1, clock=lambda: float(clock))
    for index in range(first, first + count):
        log.write("trade", trade(index))
    log.flush()
    log.flush_lines = 10 ** 9
    for index in range(first + count, first + count + unflushed):
        log.write("trade", trade(index))
    if ending == "kill":
        os._exit(0)
    log.close()
""")


def run_writer(root, *, first, count, unflushed=0, ending="close"):
    subprocess.run([sys.executable, "-c", WRITER, str(root), str(ROOT), repr(CLOCK), str(first),
                    str(count), str(unflushed), ending], check=True, timeout=120)


def append_like_an_old_writer(path, messages):
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for n, m in enumerate(messages, start=1):
            handle.write(json.dumps({"t": int(CLOCK * 1000), "n": n, "m": m}, separators=(",", ":")) + "\n")


def test_a_hard_kill_then_a_restart_loses_nothing(tmp_path):
    """Measured with real processes: with an appending writer this
    returned ZERO of ten records."""
    run_writer(tmp_path, first=0, count=5, ending="kill")
    run_writer(tmp_path, first=5, count=5)
    assert [m for _, _, m in iter_directory(tmp_path / "raw", "trade", warn=False)] == \
        [trade(i) for i in range(10)]


def test_the_crashed_file_is_untouched_by_the_restart(tmp_path):
    run_writer(tmp_path, first=0, count=5, ending="kill")
    (torn,) = (tmp_path / "raw").rglob("trade-*.jsonl.gz")
    before = torn.read_bytes()
    run_writer(tmp_path, first=5, count=5)
    assert torn.read_bytes() == before
    names = [p.name for p in sorted((tmp_path / "raw").rglob("trade-*.jsonl.gz"))]
    assert names == ["trade-14.jsonl.gz", "trade-14.r001.jsonl.gz"]


def test_archives_torn_by_an_appending_writer_are_recovered(tmp_path, capsys):
    run_writer(tmp_path, first=0, count=5, ending="kill")
    (path,) = (tmp_path / "raw").rglob("trade-*.jsonl.gz")
    append_like_an_old_writer(path, [trade(i) for i in range(5, 10)])
    assert [m for _, _, m in iter_events(path)] == [trade(i) for i in range(10)]
    assert path.name in capsys.readouterr().err


@pytest.mark.parametrize("unflushed", (3000, 4000, 8000))
def test_a_kill_mid_block_invents_nothing_and_loses_no_later_member(tmp_path, unflushed):
    """A decoder that only resynchronises on a zlib error swallowed the next
    member in 13 of 30 kills and invented a record in 2. A record may only come from bytes before the next header."""
    run_writer(tmp_path, first=0, count=1000, unflushed=unflushed, ending="kill")
    (path,) = (tmp_path / "raw").rglob("trade-*.jsonl.gz")
    later = [{"after": n} for n in range(100)]
    append_like_an_old_writer(path, later)
    recovered = [m for _, _, m in iter_events(path, warn=False)]
    torn = [m for m in recovered if "after" not in m]
    assert recovered == torn + later
    assert len(torn) >= 1000
    assert torn == [trade(i) for i in range(len(torn))]


def test_a_file_still_being_written_yields_what_was_flushed(tmp_path, capsys):
    log = RawEventLog(tmp_path, flush_lines=1, clock=lambda: CLOCK)
    try:
        for i in range(5):
            log.write("trade", trade(i))
        log.flush()                       # synchronous: the writer thread has passed the marker
        path = next((tmp_path / "raw").rglob("trade-*.jsonl.gz"))
        assert [m for _, _, m in iter_events(path)] == [trade(i) for i in range(5)]
        assert path.name in capsys.readouterr().err
    finally:
        log.close()
