"""The public trade tape: paged, de-duplicated, oldest first; and both
venues' resolutions read from their own settled rows."""

import json
from collections import OrderedDict

import httpx

from predkit.venues.kalshi import Kalshi
from predkit.venues.polymarket import Polymarket


def kalshi_with_pages(pages):
    calls = []

    def handler(request):
        calls.append(dict(request.url.params))
        cursor = request.url.params.get("cursor")
        index = int(cursor) if cursor else 0
        page = pages[index] if index < len(pages) else []
        nxt = str(index + 1) if index + 1 < len(pages) else ""
        return httpx.Response(200, json={"trades": page, "cursor": nxt})

    return Kalshi(http=httpx.Client(transport=httpx.MockTransport(handler))), calls


def trade(i, stamp="2026-09-13T20:04:33Z"):
    return {"trade_id": f"t{i}", "yes_price_dollars": "0.5000", "count_fp": "1.00", "taker_side": "yes",
            "created_time": stamp}


def test_trades_since_pages_until_a_seen_id_and_returns_oldest_first():
    # newest-first pages: [t9..t7], [t6..t4], [t3..t1]; t4 already seen.
    venue, calls = kalshi_with_pages([[trade(9), trade(8), trade(7)], [trade(6), trade(5), trade(4)], [trade(3)]])
    seen = OrderedDict([("t4", None)])
    fresh = venue._trades_since("KX", 0, seen, page=3)
    assert [r["trade_id"] for r in fresh] == ["t5", "t6", "t7", "t8", "t9"]
    assert len(calls) == 2 and "min_ts" not in calls[0]
    assert set(seen) == {"t4", "t5", "t6", "t7", "t8", "t9"}


def test_trades_since_asks_from_a_second_before_the_newest_seen():
    venue, calls = kalshi_with_pages([[trade(1)]])
    venue._trades_since("KX", 1_789_330_000, OrderedDict(), page=3)
    assert calls[0]["min_ts"] == "1789329999"


def test_kalshi_resolution_only_when_finalized():
    def handler(request):
        ticker = request.url.path.rsplit("/", 1)[1]
        status, result = {"KX-open": ("active", ""), "KX-yes": ("finalized", "yes"), "KX-no": ("finalized", "no")}[ticker]
        return httpx.Response(200, json={"market": {"ticker": ticker, "status": status, "result": result}})

    venue = Kalshi(http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert venue.resolution("KX-open") is None
    assert venue.resolution("KX-yes") is True and venue.resolution("KX-no") is False


def test_polymarket_resolution_reads_outcome_prices_of_a_closed_row():
    rows = {
        "0xlive": [{"closed": False, "outcomePrices": '["0.48","0.52"]', "outcomes": '["Up","Down"]'}],
        "0xup": [{"closed": True, "outcomePrices": '["1","0"]', "outcomes": '["Up","Down"]'}],
        "0xdown": [{"closed": True, "outcomePrices": '["0","1"]', "outcomes": '["Up","Down"]'}],
    }

    def handler(request):
        cid = request.url.params.get("condition_ids")
        # a closed market only answers when asked with closed=true
        if cid != "0xlive" and request.url.params.get("closed") != "true":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=rows[cid])

    venue = Polymarket(http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert venue.resolution("0xlive") is None
    assert venue.resolution("0xup") is True and venue.resolution("0xdown") is False
