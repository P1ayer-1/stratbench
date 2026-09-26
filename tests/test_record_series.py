"""Rolling windows: discovered ahead, started once, stopped after grace."""

import asyncio
from datetime import timedelta
from pathlib import Path

from predkit.record_series import (KalshiWindows, PolymarketWindows, SeriesRecorder, kalshi_next_ticker,
                                   read_contract, write_contract)
from predkit.schema import Contract, utc


def test_kalshi_next_ticker_is_the_next_close_in_new_york_time():
    # KXBTC15M-26SEP122215-15 closes 2026-09-13T02:15Z = 22:15 EDT on Sep 12.
    assert kalshi_next_ticker("KXBTC15M", utc(2026, 9, 13, 2, 15), 900) == "KXBTC15M-26SEP122230-30"
    # Across midnight ET: 03:45Z = 23:45 EDT -> next is 00:00 EDT Sep 13.
    assert kalshi_next_ticker("KXBTC15M", utc(2026, 9, 13, 3, 45), 900) == "KXBTC15M-26SEP130000-00"


class FakePoly:
    name = "polymarket"

    def __init__(self):
        self.lookups = []

    def market_by_slug(self, slug):
        self.lookups.append(slug)
        start = int(slug.rsplit("-", 1)[1])
        if start > 1_789_265_100 + 600:
            return None                                   # not listed yet
        return Contract("polymarket", f"0x{start}", slug, utc(2026, 9, 13, 2, 0) + timedelta(seconds=start - 1_789_264_800 + 300),
                        "chainlink", "crypto_fees_v2", yes_token="Y", no_token="N")

    async def stream(self, market_id):
        while True:
            yield ("book", {"m": market_id})
            await asyncio.sleep(0.01)


def test_polymarket_windows_are_aligned_slugs_current_and_next():
    disc = PolymarketWindows(FakePoly(), "btc-updown-5m", 300)
    found = disc.discover(1_789_265_130, ahead=1)
    assert [c.question for c in found] == ["btc-updown-5m-1789265100", "btc-updown-5m-1789265400"]
    disc.discover(1_789_265_130, ahead=1)
    assert len(disc.venue.lookups) == 2                    # cached, not refetched


async def test_series_recorder_starts_each_window_once_and_stops_after_grace(tmp_path):
    now = [1_789_265_130.0]
    disc = PolymarketWindows(FakePoly(), "btc-updown-5m", 300)

    async def no_sleep(_):
        await asyncio.sleep(0)

    rec = SeriesRecorder("t", disc, tmp_path, ahead=1, poll_s=0, grace_s=10, clock=lambda: now[0], sleep=no_sleep)
    await rec.tick()
    assert sorted(rec.active) == ["0x1789265100", "0x1789265400"]
    assert (tmp_path / "polymarket" / "0x1789265100" / "contract.json").exists()
    await asyncio.sleep(0.05)                                # let the recorders take a few messages
    await rec.tick()
    assert rec.started.count("0x1789265100") == 1            # never started twice
    now[0] = 1_789_265_400 + 10 + 1                          # first window resolved at +300, grace 10
    await rec.tick()
    assert "0x1789265100" in rec.finished and "0x1789265100" not in rec.active
    for market_id in list(rec.active):
        await rec._finish(market_id)
    assert list((tmp_path / "polymarket" / "0x1789265100" / "raw").rglob("book-*.jsonl.gz"))


def test_contract_round_trips_through_contract_json(tmp_path):
    from decimal import Decimal
    c = Contract("kalshi", "KX-1", "q", utc(2026, 9, 13, 2, 15), "kalshi:KX", "default",
                 price_ranges=((Decimal("0"), Decimal("0.1"), Decimal("0.001")),), extra={"rules_primary": "BRTI"})
    write_contract(tmp_path, c)
    assert read_contract(tmp_path) == c
