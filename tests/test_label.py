"""Rows come from the venue's result and the archive's own book, sampled at
fixed offsets before resolution."""

import json
from decimal import Decimal

from predkit.backtest import read_rows
from predkit.label import label_venue, touch_at_offsets
from predkit.rawlog import RawEventLog
from predkit.record_series import write_contract
from predkit.schema import Contract, utc

D = Decimal


def kalshi_window(tmp_path, ticker="KX-1", end=utc(2026, 9, 13, 2, 15)):
    c = Contract("kalshi", ticker, "q", end, "kalshi:KX", "default")
    d = tmp_path / "kalshi" / ticker
    write_contract(d, c)
    base = c.resolves_at.timestamp()
    clock = [base - 300]
    log = RawEventLog(d, clock=lambda: clock[0])
    for step, (yes, no) in enumerate([(48, 50), (49, 50), (50, 49), (51, 48), (52, 47)]):
        clock[0] = base - 300 + step * 60                       # -300, -240, -180, -120, -60
        log.write("book_poll", {"orderbook": {"yes": [[yes, 10]], "no": [[no, 10]]}})
    log.close()
    return c, d


class FakeVenue:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def resolution(self, market_id):
        self.calls += 1
        return self.results.get(market_id)


def test_touch_is_the_last_book_at_or_before_each_offset(tmp_path):
    c, d = kalshi_window(tmp_path)
    touch = touch_at_offsets(d, c, (240, 120, 30))
    assert touch[240][0] == D("0.49") and touch[240][1] == D("0.50")     # book written exactly at -240
    assert touch[120][0] == D("0.51") and touch[120][1] == D("0.52")     # -120 sample: no 48 -> ask 0.52
    assert touch[30][0] == D("0.52") and touch[30][1] == D("0.53")       # last book before -30 is the -60 one


def test_rows_are_labelled_by_the_venue_and_cached(tmp_path):
    c, d = kalshi_window(tmp_path)
    venue = FakeVenue({"KX-1": False})
    now = c.resolves_at.timestamp() + 600
    resolved, written, pending = label_venue(tmp_path, "kalshi", venue, offsets=(240, 60), now_s=now)
    assert (resolved, written, pending) == (1, 2, 0)
    rows = read_rows(tmp_path / "rows" / "kalshi.jsonl")
    assert all(r.resolved_yes is False and r.resolution_source == "kalshi:KX" for r in rows)
    assert sorted(r.extra["offset_s"] for r in rows) == ["240", "60"]
    assert json.loads((d / "result.json").read_text())["resolved_yes"] is False
    # Second pass: nothing re-fetched, nothing re-written.
    assert label_venue(tmp_path, "kalshi", venue, offsets=(240, 60), now_s=now) == (1, 0, 0)
    assert venue.calls == 1


def test_an_unresolved_window_is_left_pending(tmp_path):
    c, d = kalshi_window(tmp_path)
    venue = FakeVenue({})
    now = c.resolves_at.timestamp() + 600
    assert label_venue(tmp_path, "kalshi", venue, offsets=(60,), now_s=now) == (0, 0, 1)
    assert not (d / "result.json").exists()
