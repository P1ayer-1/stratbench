"""The open-interest poller: dedupe, isolation, and never dying on a failure.

No network. `fetch_open_interest` takes an `opener`, so every test here runs
against a canned payload - which is also the point of not using the SDK for
this endpoint (see the module docstring).
"""

import gzip
import json
from pathlib import Path

import pytest

from perpkit.openinterest import (
    CHANNEL,
    OpenInterestPoller,
    as_message,
    fetch_open_interest,
)
from perpkit.rawlog import iter_directory


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_for(*payloads):
    """An opener that returns each payload in turn, then repeats the last."""
    calls = {"n": 0}

    def _open(request, timeout=None):
        index = min(calls["n"], len(payloads) - 1)
        calls["n"] += 1
        item = payloads[index]
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    _open.calls = calls
    return _open


def payload(*rows):
    return {"code": "0", "msg": "success", "data": list(rows)}


def row(inst_id, oi, ts):
    return {
        "instId": inst_id,
        "openInterest": str(oi),
        "openInterestCurrency": str(oi),
        "ts": str(ts),
    }


def archived(data_dir: Path, inst_id: str):
    return list(iter_directory(data_dir / inst_id / "raw", CHANNEL))


# ---- the endpoint wrapper -------------------------------------------------


def test_fetch_returns_rows_verbatim():
    rows = fetch_open_interest(opener=opener_for(payload(row("BTC-USDT", 5, 1))))
    assert rows == [row("BTC-USDT", 5, 1)]


def test_fetch_raises_on_api_error_code():
    bad = {"code": "152001", "msg": "instType required", "data": None}
    with pytest.raises(ValueError, match="152001"):
        fetch_open_interest(opener=opener_for(bad))


def test_message_envelope_matches_the_websocket_shape():
    """Readers must not need to know this channel arrived over HTTP."""
    message = as_message(row("BTC-USDT", 5, 1))
    assert message["arg"]["channel"] == CHANNEL
    assert message["arg"]["instId"] == "BTC-USDT"
    assert message["data"] == [row("BTC-USDT", 5, 1)]


# ---- dedupe ---------------------------------------------------------------


def test_repeated_ts_is_written_once(tmp_path):
    """The endpoint updates once a minute; polling faster must not duplicate.

    This is the whole reason the poll interval can be shorter than the publish
    interval - three chances to catch each minute, one row on disk.
    """
    same = payload(row("BTC-USDT", 5416679, 1788978660000))
    poller = OpenInterestPoller(["BTC-USDT"], data_dir=tmp_path,
                                opener=opener_for(same))

    assert poller.poll_once() == 1
    assert poller.poll_once() == 0
    assert poller.poll_once() == 0
    poller.close()

    assert len(archived(tmp_path, "BTC-USDT")) == 1


def test_new_ts_is_a_new_row(tmp_path):
    poller = OpenInterestPoller(
        ["BTC-USDT"], data_dir=tmp_path,
        opener=opener_for(
            payload(row("BTC-USDT", 5416679, 1788978660000)),
            payload(row("BTC-USDT", 5402916, 1788978720000)),
        ),
    )
    poller.poll_once()
    poller.poll_once()
    poller.close()

    events = archived(tmp_path, "BTC-USDT")
    assert [event[2]["data"][0]["ts"] for event in events] == [
        "1788978660000", "1788978720000",
    ]


# ---- isolation ------------------------------------------------------------


def test_each_instrument_writes_to_its_own_directory(tmp_path):
    """Same rule record.py enforces: two symbols, two trees, no shared file."""
    poller = OpenInterestPoller(
        ["BTC-USDT", "ADA-USDT"], data_dir=tmp_path,
        opener=opener_for(payload(
            row("BTC-USDT", 5416679, 1788978660000),
            row("ADA-USDT", 1196810, 1788978660000),
        )),
    )
    assert poller.poll_once() == 2
    poller.close()

    assert len(archived(tmp_path, "BTC-USDT")) == 1
    assert len(archived(tmp_path, "ADA-USDT")) == 1
    assert (tmp_path / "BTC-USDT" / "raw") != (tmp_path / "ADA-USDT" / "raw")


def test_untracked_instruments_in_the_response_are_ignored(tmp_path):
    """One request returns all 473 instruments. Only the recorded ones land."""
    poller = OpenInterestPoller(
        ["BTC-USDT"], data_dir=tmp_path,
        opener=opener_for(payload(
            row("BTC-USDT", 5416679, 1788978660000),
            row("IOTX-USDT", 65086647, 1788978660000),
            row("ESP-USDT", 176721, 1788978660000),
        )),
    )
    assert poller.poll_once() == 1
    poller.close()

    assert not (tmp_path / "IOTX-USDT").exists()


def test_writes_the_open_interest_channel_only(tmp_path):
    """It must be unable to write a file a live recorder holds open.

    `books-14.jsonl.gz` belonging to a running record.py is the file this
    process must never touch; restricting the channel set is what guarantees
    that, and is why record_oi.py needs no restart of anything.
    """
    poller = OpenInterestPoller(["BTC-USDT"], data_dir=tmp_path,
                                opener=opener_for(payload()))
    log = poller.logs["BTC-USDT"]
    assert log.channels == {CHANNEL}

    log.write("books", {"arg": {"channel": "books"}, "data": [{}]})
    log.close()

    written = [path.name for path in (tmp_path / "BTC-USDT" / "raw").rglob("*.jsonl.gz")]
    assert all(name.startswith(CHANNEL) for name in written)


# ---- resilience -----------------------------------------------------------


def test_a_failed_poll_is_counted_not_raised(tmp_path):
    """A dropped connection must not end an overnight collection run."""
    messages = []
    poller = OpenInterestPoller(
        ["BTC-USDT"], data_dir=tmp_path,
        opener=opener_for(OSError("connection reset")),
    )

    assert poller.poll_once(on_log=messages.append) == 0
    assert poller.failures == 1
    assert any("poll failed" in message for message in messages)


def test_recovers_after_a_failure(tmp_path):
    poller = OpenInterestPoller(
        ["BTC-USDT"], data_dir=tmp_path,
        opener=opener_for(
            OSError("connection reset"),
            payload(row("BTC-USDT", 5416679, 1788978660000)),
        ),
    )
    poller.poll_once()
    assert poller.poll_once() == 1
    poller.close()

    assert len(archived(tmp_path, "BTC-USDT")) == 1


def test_missing_instrument_is_warned_once_not_every_poll(tmp_path):
    messages = []
    poller = OpenInterestPoller(
        ["NOSUCH-USDT"], data_dir=tmp_path,
        opener=opener_for(payload(row("BTC-USDT", 5416679, 1788978660000))),
    )
    for _ in range(5):
        poller.poll_once(on_log=messages.append)

    assert sum("NOSUCH-USDT" in message for message in messages) == 1


def test_disabled_writes_nothing(tmp_path):
    poller = OpenInterestPoller(
        ["BTC-USDT"], data_dir=tmp_path, enabled=False,
        opener=opener_for(payload(row("BTC-USDT", 5416679, 1788978660000))),
    )
    poller.poll_once()
    poller.close()

    assert not (tmp_path / "BTC-USDT" / "raw").exists()


def test_rows_are_readable_as_gzipped_jsonl(tmp_path):
    """The archive format is the same one every other channel uses."""
    poller = OpenInterestPoller(
        ["BTC-USDT"], data_dir=tmp_path,
        opener=opener_for(payload(row("BTC-USDT", 5416679, 1788978660000))),
    )
    poller.poll_once()
    poller.close()

    path = next((tmp_path / "BTC-USDT" / "raw").rglob("*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())

    assert set(record) == {"t", "n", "m"}
    assert record["m"]["arg"]["channel"] == CHANNEL
    assert record["m"]["data"][0]["openInterest"] == "5416679"


# ---------------------------------------------------------------------------
# Exclusive ownership
# ---------------------------------------------------------------------------
#
# Two pollers on one instrument used to append interleaved gzip members to the
# same hourly file. Measured: 20 rows each from two writers produced a file
# from which 0 of 40 records could be read back. Now each writer
# creates its file exclusively, so that destruction is gone and what remains
# is duplication: the per-process `ts` dedupe cannot see the other process, so
# the lock is the only thing stopping `record.py`'s poller and `record_oi.py`
# from archiving every row twice.

import os

from perpkit.openinterest import OpenInterestPoller
from perpkit.rawlog import iter_events


def test_two_writers_in_one_hour_no_longer_destroy_it(tmp_path):
    """The measurement the lock was written because of, now a regression test:
    the second writer must never open the first one's file."""
    from perpkit.openinterest import CHANNEL, as_message
    from perpkit.rawlog import RawEventLog

    first = RawEventLog(tmp_path / "X-USDT", channels={CHANNEL}, flush_lines=1)
    second = RawEventLog(tmp_path / "X-USDT", channels={CHANNEL}, flush_lines=1)
    for index in range(20):
        row = {"instId": "X-USDT", "openInterest": str(index), "ts": str(index)}
        first.write(CHANNEL, as_message(row))
        second.write(CHANNEL, as_message(row))
    first.close()
    second.close()

    archives = sorted((tmp_path / "X-USDT" / "raw").rglob("*.jsonl.gz"))
    assert len(archives) == 2, "each writer has a file of its own"
    recovered = [event for path in archives for event in iter_events(path, warn=False)]
    assert len(recovered) == 40, "both survive - as duplicates, hence the lock"


def test_a_second_poller_on_the_same_instrument_is_refused(tmp_path):
    first = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    try:
        with pytest.raises(SystemExit, match="already being recorded"):
            OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    finally:
        first.close()


def test_the_refusal_names_the_owning_process(tmp_path):
    """An error you cannot act on is half an error."""
    first = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    try:
        with pytest.raises(SystemExit) as caught:
            OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
        assert str(os.getpid()) in str(caught.value)
    finally:
        first.close()


def test_disjoint_instruments_do_not_conflict(tmp_path):
    """One poller per instrument, not one per machine."""
    first = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    second = OpenInterestPoller(["Y-USDT"], data_dir=tmp_path)
    first.close()
    second.close()


def test_a_partial_overlap_claims_nothing(tmp_path):
    """A refusal must not leave half its locks behind.

    Otherwise a failed start would block the instruments it managed to claim
    before hitting the conflict, and collection stops on symbols nobody is
    recording.
    """
    first = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    try:
        with pytest.raises(SystemExit):
            OpenInterestPoller(["Y-USDT", "X-USDT"], data_dir=tmp_path)
        # Y was claimed first and must have been given back.
        third = OpenInterestPoller(["Y-USDT"], data_dir=tmp_path)
        third.close()
    finally:
        first.close()


def test_closing_releases_the_claim(tmp_path):
    first = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    first.close()
    second = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    second.close()


def test_a_stale_lock_from_a_dead_process_is_taken_over(tmp_path):
    """A power cut must not stop collection forever.

    Power cuts happen; a lock file surviving
    one would turn a lost evening into permanently lost OI.
    """
    lock = tmp_path / "X-USDT" / "raw" / ".open-interest.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("999999999", encoding="utf-8")   # no such pid

    poller = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    assert int(lock.read_text(encoding="utf-8")) == os.getpid()
    poller.close()


def test_a_disabled_poller_claims_nothing(tmp_path):
    """Nothing is being written, so nothing needs owning."""
    disabled = OpenInterestPoller(["X-USDT"], data_dir=tmp_path, enabled=False)
    live = OpenInterestPoller(["X-USDT"], data_dir=tmp_path)
    live.close()
    disabled.close()
