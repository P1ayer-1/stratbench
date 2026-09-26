"""Raw archive: write, rotate, read back, and survive truncation.

The round-trip test is the important one. If replayed events don't reproduce
the live features exactly, then a model trained on replayed data is trained on
something the live bot never sees — a silent, very expensive discrepancy.
"""

import gzip
import json
import random
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from perpkit.features import FeatureEngine
from perpkit.orderbook import OrderBook
from perpkit.rawlog import RawEventLog, iter_directory, iter_events
from perpkit.tape import TradeTape

REPO = Path(__file__).resolve().parents[2]


# Pinned so every child process writes into the same hour. Two runs straddling
# an hour boundary would land in separate files and pass for the wrong reason.
CLOCK = datetime(2026, 9, 11, 14, 13, tzinfo=timezone.utc).timestamp()


def message(seq, price=100.0):
    return {
        "arg": {"channel": "books", "instId": "BTC-USDT"},
        "action": "update",
        "data": {
            "bids": [[price - 0.5, 5]], "asks": [[price + 0.5, 5]],
            "ts": str(1_700_000_000_000 + seq * 100),
            "seqId": str(seq), "prevSeqId": str(seq - 1),
        },
    }


def test_writes_and_reads_back_identically(tmp_path):
    log = RawEventLog(tmp_path)
    originals = [message(seq) for seq in range(1, 21)]
    for item in originals:
        log.write("books", item)
    log.close()

    recovered = [m for _, _, m in iter_directory(tmp_path / "raw", "books")]
    assert recovered == originals


def test_receive_timestamp_is_stored(tmp_path):
    """Feed latency cannot be reconstructed later if it isn't captured now."""
    log = RawEventLog(tmp_path)
    log.write("books", message(1))
    log.close()
    (received_ms, _, _), = list(iter_directory(tmp_path / "raw", "books"))
    assert received_ms > 1_700_000_000_000


def test_only_configured_channels_are_archived(tmp_path):
    log = RawEventLog(tmp_path, channels={"books"})
    log.write("books", message(1))
    log.write("tickers", {"arg": {"channel": "tickers"}, "data": [{}]})
    log.close()
    assert log.lines_written == 1
    assert not list((tmp_path / "raw").rglob("tickers-*.jsonl.gz"))


def test_disabled_log_writes_nothing(tmp_path):
    log = RawEventLog(tmp_path, enabled=False)
    for seq in range(10):
        log.write("books", message(seq))
    log.close()
    assert log.lines_written == 0
    assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").rglob("*"))


def test_files_are_split_per_channel(tmp_path):
    log = RawEventLog(tmp_path)
    log.write("books", message(1))
    log.write("trades", {"arg": {"channel": "trades"}, "data": [{"price": "1"}]})
    log.close()
    names = sorted(p.name.split("-")[0] for p in (tmp_path / "raw").rglob("*.jsonl.gz"))
    assert names == ["books", "trades"]


def test_reopening_in_the_same_hour_keeps_the_first_run(tmp_path):
    """A restart within the same hour must not destroy that hour's history."""
    first = RawEventLog(tmp_path)
    first.write("books", message(1))
    first.close()

    second = RawEventLog(tmp_path)
    second.write("books", message(2))
    second.close()

    assert len(list(iter_directory(tmp_path / "raw", "books"))) == 2


def test_truncated_final_line_is_tolerated(tmp_path):
    """Killing the process mid-write should cost one message, not the file."""
    log = RawEventLog(tmp_path)
    for seq in range(1, 6):
        log.write("books", message(seq))
    log.close()

    path = next((tmp_path / "raw").rglob("books-*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        content = handle.read()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content + '{"t":123,"m":{"incompl')  # torn write

    recovered = list(iter_events(path))
    assert len(recovered) == 5  # the five good ones, no exception


# ---------------------------------------------------------------------------
# Hard kills: a real process, killed with no close() and no gzip trailer
# ---------------------------------------------------------------------------


def trade(index):
    """A trade whose fields vary enough that gzip cannot flatten the stream.

    Uniform messages would never fill a deflate block between flushes, and a
    kill in the middle of a block is one of the tears being tested.
    """
    rng = random.Random(index)
    return {
        "arg": {"channel": "trades", "instId": "BTC-USDT"},
        "data": [{"tradeId": str(rng.randrange(10 ** 12)),
                  "price": f"{rng.uniform(100_000, 110_000):.1f}",
                  "size": str(rng.randrange(1, 500)),
                  "side": rng.choice(["buy", "sell"]),
                  "ts": str(1_789_135_980_000 + index)}],
    }


WRITER = textwrap.dedent("""
    import os, sys, time
    repo, root, clock, first, count, unflushed, ending = sys.argv[1:8]
    sys.path[:0] = [repo, os.path.join(repo, "tests", "perpkit")]
    time.time = lambda: float(clock)
    from perpkit.rawlog import RawEventLog
    from test_perp_rawlog import trade

    first, count, unflushed = int(first), int(count), int(unflushed)
    log = RawEventLog(root, flush_lines=1)
    for index in range(first, first + count):
        log.write("trades", trade(index))
    log.flush()
    # Then write without flushing, so what reaches disk ends mid-block.
    log.flush_lines = 10 ** 9
    for index in range(first + count, first + count + unflushed):
        log.write("trades", trade(index))
    if ending == "kill":
        os._exit(0)   # no close(), no atexit, no trailer: a power cut
    log.close()
""")


def run_writer(root, *, first, count, unflushed=0, ending="close"):
    subprocess.run(
        [sys.executable, "-c", WRITER, str(REPO), str(root), repr(CLOCK),
         str(first), str(count), str(unflushed), ending],
        check=True, timeout=120,
    )


def append_like_the_old_writer(path, messages):
    """What an earlier `RawEventLog` did: `gzip.open(path, "at")`, a
    new member on the end of whatever was there. Every archive hour written
    that way that saw a restart after a crash looks like this."""
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for n, message in enumerate(messages, start=1):
            handle.write(json.dumps({"t": int(CLOCK * 1000), "n": n, "m": message},
                                    separators=(",", ":")) + "\n")


def test_a_hard_kill_then_a_restart_in_the_same_hour_loses_nothing(tmp_path):
    """With the old append-mode writer, this returned ZERO of ten records.

    The first recorder flushes five trades and dies without closing, so its
    gzip member has no trailer. The restart appended a second member to the
    same file; reading ran through the torn member into the second's header,
    raised, and gave up before yielding anything - including the five records
    written before the crash. A power cut cost the whole hour, not its last
    few seconds.
    """
    run_writer(tmp_path, first=0, count=5, ending="kill")
    run_writer(tmp_path, first=5, count=5)

    recovered = [m for _, _, m in iter_directory(tmp_path / "raw", "trades")]
    assert recovered == [trade(i) for i in range(10)]


def test_a_restart_never_appends_to_the_file_it_found(tmp_path):
    """The crashed file stays byte-for-byte as the crash left it, and the
    restart writes beside it under a name that sorts after it."""
    run_writer(tmp_path, first=0, count=5, ending="kill")
    (torn,) = (tmp_path / "raw").rglob("trades-*.jsonl.gz")
    before = torn.read_bytes()

    run_writer(tmp_path, first=5, count=5)

    assert torn.read_bytes() == before
    names = [p.name for p in sorted((tmp_path / "raw").rglob("trades-*.jsonl.gz"))]
    assert names == ["trades-14.jsonl.gz", "trades-14.r001.jsonl.gz"]


def test_archives_torn_by_the_old_appending_writer_are_recovered(tmp_path, capsys):
    """History already on disk was written by the appending writer, so the
    reader has to recover it - fixing the writer alone saves nothing past."""
    run_writer(tmp_path, first=0, count=5, ending="kill")
    (path,) = (tmp_path / "raw").rglob("trades-*.jsonl.gz")
    append_like_the_old_writer(path, [trade(i) for i in range(5, 10)])

    assert [m for _, _, m in iter_events(path)] == [trade(i) for i in range(10)]
    assert path.name in capsys.readouterr().err   # said, not papered over


# Unflushed trades before the kill. With this file's `trade()`, a decoder that
# only resynchronises on a zlib error: 3000 swallows the appended member
# whole, 4000 decodes 37 bytes past its header, and 8000 does both and also
# completes the torn last line into a well-formed record with the wrong
# contents. A different zlib may tear elsewhere; the assertions hold anyway.
MID_BLOCK_KILLS = (3000, 4000, 8000)


@pytest.mark.parametrize("unflushed", MID_BLOCK_KILLS)
def test_a_kill_mid_block_invents_nothing_and_loses_no_later_member(tmp_path, unflushed):
    """Killed between flushes, the torn member ends inside a deflate block.

    Measured over 30 such kills (1,000 to 30,000 unflushed trades):
    decoding carried on through the next member's bytes as if they continued
    the block. In 13 it never raised at all, silently swallowing the member a
    restart appended. In 2 it completed the torn last line into a well-formed
    record carrying the next sequence number and the wrong contents, which no
    ordering check could catch. So a record may only come from bytes before
    the next real gzip header.
    """
    run_writer(tmp_path, first=0, count=1000, unflushed=unflushed, ending="kill")
    (path,) = (tmp_path / "raw").rglob("trades-*.jsonl.gz")
    later = [{"after": n} for n in range(100)]
    append_like_the_old_writer(path, later)

    recovered = [m for _, _, m in iter_events(path, warn=False)]
    torn = [m for m in recovered if "after" not in m]
    assert recovered == torn + later          # nothing lost after the tear
    assert len(torn) >= 1000                  # everything flushed survives
    assert torn == [trade(i) for i in range(len(torn))]   # nothing invented


def test_a_file_still_being_written_yields_what_has_been_flushed(tmp_path, capsys):
    """The current hour is an unterminated gzip stream for as long as it is
    recorded, and analysis runs while recording continues."""
    log = RawEventLog(tmp_path, flush_lines=1)
    try:
        for index in range(5):
            log.write("trades", trade(index))
        path = next((tmp_path / "raw").rglob("trades-*.jsonl.gz"))
        assert [m for _, _, m in iter_events(path)] == [trade(i) for i in range(5)]
        assert path.name in capsys.readouterr().err
    finally:
        log.close()


def test_compression_actually_helps(tmp_path):
    """Book messages are highly repetitive; if gzip weren't earning its keep
    the per-day storage estimates would be wrong."""
    log = RawEventLog(tmp_path)
    for seq in range(1, 2001):
        log.write("books", message(seq))
    log.close()

    on_disk = log.disk_bytes()
    assert on_disk < log.bytes_estimate / 4, (
        f"expected >4x compression, got {log.bytes_estimate / on_disk:.1f}x"
    )


# ---------------------------------------------------------------------------
# The round-trip that matters
# ---------------------------------------------------------------------------


def test_replayed_events_reproduce_live_features_exactly(tmp_path):
    """Replay must be bit-identical to the live path, or training data and
    production inputs silently diverge."""
    log = RawEventLog(tmp_path)

    live_book, live_tape, live_engine = OrderBook(), TradeTape(), FeatureEngine()
    live_snapshots = []

    events = [
        {
            "arg": {"channel": "books"}, "action": "snapshot",
            "data": {"bids": [[99.5, 5], [99.0, 8]], "asks": [[100.5, 5], [101.0, 8]],
                     "ts": "1700000000000", "seqId": "1", "prevSeqId": "0"},
        }
    ]
    for seq in range(2, 40):
        events.append({
            "arg": {"channel": "books"}, "action": "update",
            "data": {
                "bids": [[99.5, 5 + seq % 7]], "asks": [[100.5, 5 + seq % 5]],
                "ts": str(1_700_000_000_000 + seq * 100),
                "seqId": str(seq), "prevSeqId": str(seq - 1),
            },
        })
        if seq % 4 == 0:
            events.append({
                "arg": {"channel": "trades"},
                "data": [{"price": "100.0", "size": "1", "side": "buy",
                          "ts": str(1_700_000_000_000 + seq * 100)}],
            })

    for event in events:
        channel = event["arg"]["channel"]
        log.write(channel, event)
        if channel == "books":
            live_book.apply(event)
            if live_book.is_ready and not live_book.is_crossed():
                live_engine.on_book_event(live_book)
                live_snapshots.append(live_engine.compute(live_book, live_tape))
        else:
            live_tape.add_message(event["data"])
    log.close()

    # Now replay from disk with fresh state.
    replay_book, replay_tape, replay_engine = OrderBook(), TradeTape(), FeatureEngine()
    replay_snapshots = []
    # Sort on (receive_ms, sequence) — the timestamp alone is too coarse to
    # recover the true interleaving of books and trades.
    merged = sorted(
        list(iter_directory(tmp_path / "raw", "books"))
        + list(iter_directory(tmp_path / "raw", "trades")),
        key=lambda item: (item[0], item[1]),
    )
    for _, _, event in merged:
        channel = event["arg"]["channel"]
        if channel == "books":
            replay_book.apply(event)
            if replay_book.is_ready and not replay_book.is_crossed():
                replay_engine.on_book_event(replay_book)
                replay_snapshots.append(replay_engine.compute(replay_book, replay_tape))
        else:
            replay_tape.add_message(event["data"])

    assert len(replay_snapshots) == len(live_snapshots)
    for live, replayed in zip(live_snapshots, replay_snapshots):
        live_values = live.to_dict()
        replay_values = replayed.to_dict()
        # `received_ts` and anything derived from wall clock legitimately
        # differ between the two runs; everything computed from market data
        # must be identical.
        for key in live_values:
            if key in ("received_ts", "book_age_ms", "tape_staleness_s"):
                continue
            assert live_values[key] == replay_values[key], f"{key} diverged"
