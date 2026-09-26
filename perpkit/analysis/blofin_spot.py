"""BloFin's spot market endpoints, which the BloFin SDK does not expose.

`blofin-sdk-python` covers futures only: `rest_market.getInstruments()` returns
487 SWAP contracts and no spot pairs, and there is no spot module at all. The
endpoints exist — they simply are not wrapped:

    /api/v1/spot/market/instruments?instType=SPOT     242 pairs
    /api/v1/spot/market/tickers?instType=SPOT         best bid/ask per pair

Both are public and unauthenticated, like their futures counterparts. This is
a thin wrapper over the SDK's own `Client`, not a reimplementation of it, so
signing, base URLs and demo-mode routing keep working the way they already do.

Why this matters: a delta-neutral funding carry is long spot and short perp,
so without spot access the only way to harvest funding on this venue is to
short the perp unhedged — which is a directional bet wearing a carry costume.
With it, the trade is available on one venue with one set of credentials.

One asymmetry that shapes everything downstream: **you can only be long spot.**
Shorting it needs margin or a borrow, which these endpoints do not offer. So
only POSITIVE funding is harvestable this way (receive funding by being short
the perp, hedged long the spot). An instrument whose funding is negative —
BTC-USDT at the time of writing — cannot be carried in this structure at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SPOT_INSTRUMENTS = "/api/v1/spot/market/instruments"
SPOT_TICKERS = "/api/v1/spot/market/tickers"

# Both endpoints reject an absent instType with code 152001, which is what
# makes them look missing if you probe them bare.
SPOT = "SPOT"


def _rows(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    return data if isinstance(data, list) else []


def fetch_spot_instruments(client) -> Dict[str, Dict[str, Any]]:
    """Every spot pair, by instId. Same id shape as the perps (`BTC-USDT`).

    The shared id shape is what makes the perp/spot join trivial, and it is
    worth not taking for granted: `contractValue`, `maxLeverage` and
    `contractType` all come back null here, which is the reliable way to tell
    a spot row from a swap row if the two are ever mixed.
    """
    payload = client.get(SPOT_INSTRUMENTS, params={"instType": SPOT}, sign=False)
    return {row["instId"]: row for row in _rows(payload) if row.get("instId")}


def fetch_spot_tickers(client) -> Dict[str, Dict[str, Any]]:
    """Best bid/ask and 24h volume per spot pair, by instId."""
    payload = client.get(SPOT_TICKERS, params={"instType": SPOT}, sign=False)
    return {row["instId"]: row for row in _rows(payload) if row.get("instId")}


def top_of_book(ticker: Optional[Dict[str, Any]]) -> Optional[tuple]:
    """(bid, ask, mid, spread_bps), or None if the quote is unusable.

    A crossed or one-sided book is not a spread measurement. Returning None
    rather than a negative number keeps the caller from quietly averaging in
    a quote that would flatter the cost of trading.
    """
    if not ticker:
        return None
    try:
        bid = float(ticker.get("bidPrice") or 0)
        ask = float(ticker.get("askPrice") or 0)
    except (TypeError, ValueError):
        return None
    if not (bid > 0 and ask > 0 and ask > bid):
        return None
    mid = (bid + ask) / 2.0
    return bid, ask, mid, (ask - bid) / mid * 10_000.0
