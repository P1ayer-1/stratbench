"""Recording mark and index price, and sharing an archive with the OI poller.

The series matters because the basis - mark against the traded book - is the
one input to `perpkit/analysis/funding_carry.py`'s convergence cost that nothing here could
observe over time, and the carry's liquidation is struck against the mark
rather than against any price that printed.

The tests that earn their place are the ones about coexistence: this poller
writes into the same per-instrument archive an open-interest poller is already
writing to, and two pollers on one channel have already cost an hour once
(0 of 40 records, before archive files were created exclusively).

No network: the opener is injected.
"""

import json

import pytest

from perpkit.markprice import CHANNEL, MARK_PRICE_PATH, MarkPricePoller
from perpkit.openinterest import OpenInterestPoller
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
    calls = {"n": 0, "urls": []}

    def _open(request, timeout=None):
        index = min(calls["n"], len(payloads) - 1)
        calls["n"] += 1
        calls["urls"].append(getattr(request, "full_url", str(request)))
        item = payloads[index]
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    _open.calls = calls
    return _open


def payload(*rows):
    return {"code": "0", "msg": "success", "data": list(rows)}


def row(inst_id, mark, index, ts):
    return {"instId": inst_id, "markPrice": str(mark),
            "indexPrice": str(index), "ts": str(ts)}


def poller_at(tmp_path, opener, **kwargs):
    return MarkPricePoller(["SUI-USDT"], data_dir=tmp_path, opener=opener,
                           **kwargs)


# ---------------------------------------------------------------------------
# What it records
# ---------------------------------------------------------------------------


def test_both_prices_are_archived_not_just_the_mark():
    """indexPrice is half the point: mark minus index is the funding basis,
    and an archive holding only one of them cannot reconstruct it."""
    assert "markPrice" in row("SUI-USDT", 1, 2, 3)
    assert "indexPrice" in row("SUI-USDT", 1, 2, 3)


def test_a_poll_writes_one_row_per_instrument(tmp_path):
    opener = opener_for(payload(row("SUI-USDT", "0.73", "0.7324", 1000),
                                row("BTC-USDT", "77000", "77010", 1000)))
    poller = poller_at(tmp_path, opener)
    assert poller.poll_once() == 1          # only SUI is subscribed
    poller.close()

    events = list(iter_directory(tmp_path / "SUI-USDT" / "raw", CHANNEL))
    assert len(events) == 1
    data = events[0][2]["data"][0]
    assert data["markPrice"] == "0.73"
    assert data["indexPrice"] == "0.7324"


def test_it_calls_the_mark_price_endpoint(tmp_path):
    opener = opener_for(payload(row("SUI-USDT", "0.73", "0.7324", 1000)))
    poller = poller_at(tmp_path, opener)
    poller.poll_once()
    poller.close()
    assert opener.calls["urls"][0].endswith(MARK_PRICE_PATH)


def test_the_envelope_matches_every_other_channel(tmp_path):
    """A reader must not need to know this arrived over HTTP."""
    opener = opener_for(payload(row("SUI-USDT", "0.73", "0.7324", 1000)))
    poller = poller_at(tmp_path, opener)
    poller.poll_once()
    poller.close()

    events = list(iter_directory(tmp_path / "SUI-USDT" / "raw", CHANNEL))
    message = events[0][2]
    assert message["arg"] == {"channel": "mark-price", "instId": "SUI-USDT"}
    assert isinstance(message["data"], list)


def test_an_unchanged_timestamp_is_not_written_twice(tmp_path):
    """The series publishes continuously, so the poll interval is the sample
    rate - but polling faster than the publisher must still cost nothing."""
    same = payload(row("SUI-USDT", "0.73", "0.7324", 1000))
    poller = poller_at(tmp_path, opener_for(same, same, same))
    assert poller.poll_once() == 1
    assert poller.poll_once() == 0
    assert poller.poll_once() == 0
    poller.close()


def test_a_moved_timestamp_is_written(tmp_path):
    poller = poller_at(tmp_path, opener_for(
        payload(row("SUI-USDT", "0.73", "0.7324", 1000)),
        payload(row("SUI-USDT", "0.7301", "0.7325", 1003))))
    assert poller.poll_once() == 1
    assert poller.poll_once() == 1
    poller.close()


def test_a_failed_request_is_survived_not_raised(tmp_path):
    """A poller that dies on a dropped connection silently stops recording,
    and the hours it misses cannot be fetched later."""
    poller = poller_at(tmp_path, opener_for(
        ConnectionError("gateway"),
        payload(row("SUI-USDT", "0.73", "0.7324", 1000))))
    assert poller.poll_once() == 0
    assert poller.failures == 1
    assert poller.poll_once() == 1
    poller.close()


def test_it_samples_faster_than_open_interest_by_default(tmp_path):
    """OI publishes once a minute; this publishes continuously, so the two
    have no business sharing a cadence."""
    poller = poller_at(tmp_path, opener_for(payload()))
    assert poller.poll_seconds == 10.0
    poller.close()


# ---------------------------------------------------------------------------
# Sharing the archive with the open-interest poller
# ---------------------------------------------------------------------------


def test_it_runs_alongside_an_open_interest_poller_on_the_same_instrument(tmp_path):
    """The lock is named after the CHANNEL, so different channels coexist.

    This is the whole reason mark-price can start without stopping anything.
    """
    oi = OpenInterestPoller(["SUI-USDT"], data_dir=tmp_path,
                            opener=opener_for(payload()))
    marks = poller_at(tmp_path, opener_for(payload()))   # must not raise
    assert (tmp_path / "SUI-USDT" / "raw" / ".open-interest.lock").exists()
    assert (tmp_path / "SUI-USDT" / "raw" / ".mark-price.lock").exists()
    oi.close()
    marks.close()


def test_a_second_mark_price_poller_is_still_refused(tmp_path):
    """Two on ONE channel archive every row twice - and before files were
    created exclusively, destroyed the hour outright (0 of 40, measured)."""
    first = poller_at(tmp_path, opener_for(payload()))
    with pytest.raises(SystemExit) as excinfo:
        poller_at(tmp_path, opener_for(payload()))
    assert "mark-price" in str(excinfo.value)
    first.close()


def test_closing_releases_the_lock_so_a_restart_works(tmp_path):
    first = poller_at(tmp_path, opener_for(payload()))
    first.close()
    assert not (tmp_path / "SUI-USDT" / "raw" / ".mark-price.lock").exists()
    second = poller_at(tmp_path, opener_for(payload()))   # must not raise
    second.close()


def test_the_two_channels_write_separate_files(tmp_path):
    """One file per channel per hour is what keeps them from touching."""
    oi_rows = {"instId": "SUI-USDT", "openInterest": "5", "ts": "1000"}
    oi = OpenInterestPoller(["SUI-USDT"], data_dir=tmp_path,
                            opener=opener_for(payload(oi_rows)))
    marks = poller_at(tmp_path, opener_for(
        payload(row("SUI-USDT", "0.73", "0.7324", 1000))))
    oi.poll_once()
    marks.poll_once()
    oi.close()
    marks.close()

    raw = tmp_path / "SUI-USDT" / "raw"
    written = sorted(p.name.rsplit("-", 1)[0]
                     for p in raw.rglob("*.jsonl.gz"))
    assert written == ["mark-price", "open-interest"]
    # And each decodes independently, end to end.
    for channel in ("mark-price", "open-interest"):
        assert len(list(iter_directory(raw, channel))) == 1
