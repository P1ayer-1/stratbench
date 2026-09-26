"""Verbatim frames: the archive line is byte-identical to a parsed write,
and the tag scan agrees with a real parse."""

import json
from datetime import datetime, timezone

from predkit.rawlog import RawEventLog, iter_directory
from predkit.venues.polymarket import _event_type

CLOCK = datetime(2026, 9, 13, 14, 13, tzinfo=timezone.utc).timestamp()


def test_a_raw_frame_is_embedded_verbatim_and_reads_back_as_the_object(tmp_path):
    frame = '{"event_type":"price_change","market":"0xc","price_changes":[{"asset_id":"1","price":"0.48","size":"10","side":"BUY"}],"timestamp":"1789307642631"}'
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("price_change", frame)
    log.write("price_change", json.loads(frame))
    log.close()
    (t1, n1, m1), (t2, n2, m2) = iter_directory(tmp_path / "raw", "price_change", warn=False)
    assert m1 == m2 == json.loads(frame)
    assert (n1, n2) == (1, 2)


def test_a_reader_gets_the_frame_bytes_as_archived(tmp_path):
    """Replay drops the recorder's late copies by these bytes: a verbatim
    frame must come back exactly as the socket sent it (spacing included),
    and a parsed write as the writer's compact JSON."""
    frame = '{"event_type":"price_change", "market":"0xc", "timestamp":"1789307642631"}'
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("x", frame)
    log.write("x", {"a": [1, "}"]})
    log.close()
    (_, _, m1, b1), (_, _, m2, b2) = iter_directory(tmp_path / "raw", "x", warn=False, frames=True)
    assert (m1, b1) == (json.loads(frame), frame.encode())
    assert (m2, b2) == ({"a": [1, "}"]}, b'{"a":[1,"}"]}')


def test_event_type_scan_matches_a_parse():
    for frame in ('{"event_type":"book","asset_id":"1","bids":[]}',
                  '{"market":"0xc","price_changes":[],"event_type":"price_change","timestamp":"1"}',
                  '{"asset_id":"1","event_type":"last_trade_price","price":"0.5"}'):
        assert _event_type(frame) == json.loads(frame)["event_type"]
    assert _event_type('{"no":"type"}') == "unknown"
    assert _event_type('{"event_type":"') == "unknown"


def test_a_frame_that_is_not_json_costs_one_line_not_the_file(tmp_path):
    log = RawEventLog(tmp_path, clock=lambda: CLOCK)
    log.write("x", "not json at all")
    log.write("x", '{"ok":1}')
    log.close()
    assert [m for _, _, m in iter_directory(tmp_path / "raw", "x", warn=False)] == [{"ok": 1}]
