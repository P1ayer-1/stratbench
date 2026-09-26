"""Kalshi: one REST API for everything, RSA-PSS signed when it writes.

Endpoints, checked live 2026-09-12 against the public book:
  REST  https://api.elections.kalshi.com/trade-api/v2
        /markets, /markets/<ticker>, /markets/<ticker>/orderbook, /markets/trades
        /portfolio/orders, /portfolio/fills            (signed)
  WS    wss://api.elections.kalshi.com/trade-api/ws/v2  (signed handshake)

Read live 2026-09-13: the v2 API now serves DOLLAR STRINGS and fractional
counts. The orderbook is `{"orderbook_fp": {"yes_dollars": [["0.5100",
"1158.14"], ...], "no_dollars": [...]}}`, both lists of BIDS, ascending,
best last; trades carry `yes_price_dollars`, `no_price_dollars`,
`count_fp`, `taker_side`; markets carry `yes_bid_dollars`, a
`price_level_structure` of `tapered_deci_cent` with explicit
`price_ranges` (step 0.001 below 0.10 and above 0.90, 0.01 between), and
a `close_time` that IS the settlement time for the 15-minute markets
(`expiration_time` sits a week later). The older cents shape (`orderbook.yes`
as [[cents, count]], `yes_price`, `count`) is still parsed, because the
websocket snapshot/delta shape was not observed live and may be either.

A NO bid at 0.52 is a YES ask at 0.48, and that conversion happens here so
the rest of the package sees one YES-denominated book.

The order body sends `yes_price_dollars` / `no_price_dollars` and
`count_fp` to match the read side; that write shape was NOT verified live
(no key was used when it was written). A wrong field name is a 4xx, not a wrong
order.

Signing: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP (ms) and
KALSHI-ACCESS-SIGNATURE = base64(RSA-PSS-SHA256(timestamp + METHOD + path))
where `path` is the URL path without the query. The private key is loaded
by the caller from `predkit.keys` and handed in as a `Signer`; this module
never reads a key file. Without a signer the adapter reads public data and
refuses to write; the websocket also needs the signature, so recording
without one falls back to polling the public book (channel `book_poll`).

Fee tier is `default` for every market: Kalshi charges takers on the 0.07
curve and makers nothing on most markets. Known limitation: this adapter
does not read the series' `fee_type`, so a series that charges makers
(`quadratic_with_maker_fees` in `fees.py`) is labelled `default` here; set
`fee_tier` on the contract yourself for those series.

Resolution source is `kalshi:<series>` (KXBTC15M resolves on Kalshi's own
BTC index per its rules); the market's `rules_primary` text is kept on the
contract's question for the human.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Tuple

import httpx

from predkit.schema import (Book, Contract, Fill, Level, OrderIntent, Outcome, Side, Trade,
                            no_to_yes, to_decimal)
from predkit.venues import NotSignable

BASE = "https://api.elections.kalshi.com/trade-api/v2"
WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
CENTS = Decimal(100)


class Signer(Protocol):
    key_id: str

    def sign(self, message: bytes) -> bytes: ...


class RsaPssSigner:
    """RSA-PSS over SHA-256, as Kalshi's docs specify. Holds the private key
    object in memory for the life of the process and nowhere else."""

    def __init__(self, key_id: str, private_key_pem: bytes):
        from cryptography.hazmat.primitives import serialization

        self.key_id = key_id
        self._key = serialization.load_pem_private_key(private_key_pem, password=None)

    def sign(self, message: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        return self._key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )


def auth_headers(signer: Signer, method: str, path: str, *, now_ms: Optional[int] = None) -> Dict[str, str]:
    stamp = str(now_ms if now_ms is not None else int(time.time() * 1000))
    signature = signer.sign(f"{stamp}{method.upper()}{path}".encode())
    return {
        "KALSHI-ACCESS-KEY": signer.key_id,
        "KALSHI-ACCESS-TIMESTAMP": stamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


def _parse_iso(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def contract_from_market(row: Dict[str, Any]) -> Contract:
    ticker = row["ticker"]
    series = row.get("series_ticker") or ticker.split("-")[0]
    # close_time is when trading stops and, for the short markets, when the
    # settlement value is read; expiration_time is the latest the venue may
    # take to settle (a week later on KXBTC15M, read live 2026-09-13).
    resolves = row.get("close_time") or row.get("expiration_time")
    if not resolves:
        raise ValueError(f"kalshi market {ticker} has no close_time or expiration_time")
    ranges = tuple((to_decimal(r["start"]), to_decimal(r["end"]), to_decimal(r["step"]))
                   for r in row.get("price_ranges") or [])
    extra = {k: row[k] for k in ("rules_primary", "expiration_time", "settlement_timer_seconds",
                                 "price_level_structure", "status") if k in row}
    return Contract(
        venue="kalshi",
        market_id=ticker,
        question=row.get("title") or row.get("subtitle") or ticker,
        resolves_at=_parse_iso(resolves),
        resolution_source=f"kalshi:{series}",
        fee_tier="default",
        tick=Decimal("0.01"),
        min_size=Decimal("1"),
        price_ranges=ranges,
        extra=extra,
    )


class Kalshi:
    name = "kalshi"

    def __init__(self, *, http: Optional[httpx.Client] = None, signer: Optional[Signer] = None,
                 base_url: str = BASE, ws_url: str = WS, poll_interval_s: float = 1.0,
                 trades_interval_s: float = 2.0):
        self.http = http or httpx.Client(timeout=20.0, headers={"User-Agent": "predkit/0.1"})
        self.signer = signer
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self.poll_interval_s = poll_interval_s
        self.trades_interval_s = trades_interval_s
        self._contracts: Dict[str, Contract] = {}

    # ---- transport -------------------------------------------------------

    def _path(self, route: str) -> str:
        return httpx.URL(self.base_url).path + route

    def _request(self, method: str, route: str, *, signed: bool = False, **kwargs) -> Any:
        headers = dict(kwargs.pop("headers", {}) or {})
        if signed:
            if self.signer is None:
                raise NotSignable("kalshi: no signer. Load the key with predkit.keys and pass "
                                  "an RsaPssSigner; this adapter never reads a key file.")
            headers.update(auth_headers(self.signer, method, self._path(route)))
        response = self.http.request(method, self.base_url + route, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    # ---- read ------------------------------------------------------------

    def list_markets(self, query: Optional[str] = None, *, limit: int = 50) -> List[Contract]:
        params: Dict[str, Any] = {"status": "open", "limit": limit}
        if query:
            params["series_ticker"] = query.upper()
        rows = self._request("GET", "/markets", params=params).get("markets", [])
        out = []
        for row in rows:
            try:
                contract = contract_from_market(row)
            except (ValueError, KeyError):
                continue
            self._contracts[contract.market_id] = contract
            out.append(contract)
        return out

    def market(self, market_id: str) -> Contract:
        if market_id in self._contracts:
            return self._contracts[market_id]
        row = self._request("GET", f"/markets/{market_id}").get("market")
        if not row:
            raise KeyError(f"kalshi: no market {market_id}")
        contract = contract_from_market(row)
        self._contracts[market_id] = contract
        return contract

    def book(self, market_id: str) -> Book:
        raw = self._request("GET", f"/markets/{market_id}/orderbook")
        return parse_orderbook(raw, market_id)

    def trades(self, market_id: str, *, limit: int = 100) -> List[Trade]:
        raw = self._request("GET", "/markets/trades", params={"ticker": market_id, "limit": limit})
        return [parse_trade(row, market_id) for row in raw.get("trades", [])]

    async def stream(self, market_id: str) -> AsyncIterator[Tuple[str, Any]]:
        if self.signer is None:
            async for item in self._poll(market_id):
                yield item
            return
        import websockets

        headers = auth_headers(self.signer, "GET", "/trade-api/ws/v2")
        async with websockets.connect(self.ws_url, additional_headers=headers,
                                      ping_interval=10, ping_timeout=10) as socket:
            await socket.send(json.dumps({"id": 1, "cmd": "subscribe", "params": {
                "channels": ["orderbook_delta", "trade"], "market_tickers": [market_id]}}))
            async for raw in socket:
                message = json.loads(raw)
                yield (str(message.get("type", "unknown")), message)

    async def _poll(self, market_id: str) -> AsyncIterator[Tuple[str, Any]]:
        """No signer, no websocket: the public book once a second and the
        public trade tape every two, both verbatim.

        The book is a sample; the tape is complete. `/markets/trades` is
        newest-first, cursor-paged, up to 1,000 rows a page and accepts
        `min_ts` (read live 2026-09-13; a live KXBTC15M window printed 200
        trades in 7 s mid-window), so each poll asks from a second before the
        newest trade already seen and follows the cursor until it meets a
        seen id, then archives the unseen rows oldest-first under
        `trade_poll`. De-duplication is by `trade_id` over the last 20,000.
        Cost per window: 1.5 requests a second.
        """
        queue: "asyncio.Queue[Tuple[str, Any]]" = asyncio.Queue()

        async def book_loop() -> None:
            while True:
                try:
                    raw = await asyncio.to_thread(self._request, "GET", f"/markets/{market_id}/orderbook")
                    raw["_polled_at_ms"] = int(time.time() * 1000)
                    await queue.put(("book_poll", raw))
                except Exception as exc:
                    await queue.put(("_error", exc))
                await asyncio.sleep(self.poll_interval_s)

        async def trades_loop() -> None:
            seen: "collections.OrderedDict[str, None]" = collections.OrderedDict()
            newest_ts = 0
            while True:
                try:
                    rows = await asyncio.to_thread(self._trades_since, market_id, newest_ts, seen)
                    for row in rows:                     # oldest first
                        row["_polled_at_ms"] = int(time.time() * 1000)
                        await queue.put(("trade_poll", row))
                        stamp = row.get("created_time")
                        if stamp:
                            newest_ts = max(newest_ts, int(_parse_iso(stamp).timestamp()))
                except Exception as exc:
                    await queue.put(("_error", exc))
                await asyncio.sleep(self.trades_interval_s)

        tasks = [asyncio.create_task(book_loop()), asyncio.create_task(trades_loop())]
        try:
            while True:
                channel, item = await queue.get()
                if channel == "_error":
                    raise item
                yield (channel, item)
        finally:
            for task in tasks:
                task.cancel()

    def _trades_since(self, market_id: str, newest_ts: int, seen: "collections.OrderedDict[str, None]",
                      *, page: int = 1000, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Unseen trades newer than `newest_ts`, oldest first. Follows the
        cursor until a page holds a seen id or `max_pages` is reached."""
        params: Dict[str, Any] = {"ticker": market_id, "limit": page}
        if newest_ts:
            params["min_ts"] = newest_ts - 1
        fresh: List[Dict[str, Any]] = []
        cursor = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            payload = self._request("GET", "/markets/trades", params=params)
            rows = payload.get("trades", [])
            hit_seen = False
            for row in rows:
                trade_id = str(row.get("trade_id", ""))
                if trade_id in seen:
                    hit_seen = True
                    continue
                fresh.append(row)
            cursor = payload.get("cursor")
            if hit_seen or not cursor or len(rows) < page:
                break
        for row in fresh:
            seen[str(row.get("trade_id", ""))] = None
        while len(seen) > 20_000:
            seen.popitem(last=False)
        fresh.reverse()
        return fresh

    def resolution(self, market_id: str) -> Optional[bool]:
        """True/False once the venue has finalized the market, else None.
        The venue's own result, which is the only label the backtest takes."""
        row = self._request("GET", f"/markets/{market_id}").get("market") or {}
        if row.get("status") != "finalized":
            return None
        result = str(row.get("result", "")).lower()
        if result not in ("yes", "no"):
            return None
        return result == "yes"

    # ---- write -----------------------------------------------------------

    def place_order(self, intent: OrderIntent) -> str:
        body: Dict[str, Any] = {
            "ticker": intent.contract.market_id,
            "action": intent.side.value,
            "side": intent.outcome.value,
            "type": "limit",
            "count_fp": f"{intent.size:.2f}",
            "post_only": intent.post_only,
        }
        field = "yes_price_dollars" if intent.outcome is Outcome.YES else "no_price_dollars"
        body[field] = f"{intent.price:.4f}"
        if intent.client_id:
            body["client_order_id"] = intent.client_id
        result = self._request("POST", "/portfolio/orders", signed=True, json=body)
        order_id = (result.get("order") or {}).get("order_id")
        if not order_id:
            raise RuntimeError(f"kalshi rejected the order: {result}")
        return str(order_id)

    def cancel(self, order_id: str) -> None:
        self._request("DELETE", f"/portfolio/orders/{order_id}", signed=True)

    def open_orders(self, market_id: Optional[str] = None) -> List[dict]:
        params = {"status": "resting"}
        if market_id:
            params["ticker"] = market_id
        return list(self._request("GET", "/portfolio/orders", signed=True, params=params)
                    .get("orders", []))

    def fills(self, market_id: Optional[str] = None) -> List[Fill]:
        params = {"ticker": market_id} if market_id else {}
        rows = self._request("GET", "/portfolio/fills", signed=True, params=params).get("fills", [])
        return [fill_from_row(row) for row in rows]


# ---- parsing, usable by replay without a client ---------------------------

def _cents(value: Any) -> Decimal:
    return to_decimal(value) / CENTS


def _price(row: Dict[str, Any], dollars_key: str, cents_key: str) -> Decimal:
    """Dollar-string field if present (v2, read live 2026-09-13), else the
    older integer-cents field."""
    if dollars_key in row:
        return to_decimal(row[dollars_key])
    return _cents(row[cents_key])


def _count(row: Dict[str, Any]) -> Decimal:
    return to_decimal(row["count_fp"] if "count_fp" in row else row["count"])


def parse_orderbook(raw: Dict[str, Any], market_id: str, *, ts_ms: int = 0) -> Book:
    """REST `/orderbook` (`orderbook_fp` with `yes_dollars`/`no_dollars`, or
    the older `orderbook` with `yes`/`no` in cents) or a WS snapshot under
    `msg`. Every list is BIDS; NO bids become YES asks at 1 - price."""
    if "orderbook_fp" in raw:
        book = raw["orderbook_fp"]
        bids = [Level(to_decimal(p), q) for p, q in book.get("yes_dollars") or []]
        asks = [Level(no_to_yes(to_decimal(p)), q) for p, q in book.get("no_dollars") or []]
    else:
        book = raw.get("orderbook", raw.get("msg", raw))
        if "yes_dollars" in book or "no_dollars" in book:
            bids = [Level(to_decimal(p), q) for p, q in book.get("yes_dollars") or []]
            asks = [Level(no_to_yes(to_decimal(p)), q) for p, q in book.get("no_dollars") or []]
        else:
            bids = [Level(_cents(p), q) for p, q in book.get("yes") or []]
            asks = [Level(no_to_yes(_cents(p)), q) for p, q in book.get("no") or []]
    return Book(market_id=market_id, bids=bids, asks=asks,
                ts_ms=ts_ms or int(raw.get("_polled_at_ms", 0)))


def parse_trade(row: Dict[str, Any], market_id: str) -> Trade:
    taker = str(row.get("taker_side", "yes")).lower()
    created = row.get("created_time")
    ts_ms = int(_parse_iso(created).timestamp() * 1000) if created else int(row.get("ts", 0)) * 1000
    return Trade(market_id=market_id, price=_price(row, "yes_price_dollars", "yes_price"),
                 size=_count(row), aggressor=Side.BUY if taker == "yes" else Side.SELL, ts_ms=ts_ms,
                 trade_id=str(row.get("trade_id", "")))


def fill_from_row(row: Dict[str, Any]) -> Fill:
    outcome = Outcome(str(row.get("side", "yes")).lower())
    price = (_price(row, "yes_price_dollars", "yes_price") if outcome is Outcome.YES
             else _price(row, "no_price_dollars", "no_price"))
    created = row.get("created_time")
    if "fee_dollars" in row:
        fee = to_decimal(row["fee_dollars"])
    elif "fee_cents" in row:
        fee = _cents(row["fee_cents"])
    else:
        fee = to_decimal(row.get("fee", 0))
    return Fill(
        venue="kalshi", market_id=row["ticker"], order_id=str(row.get("order_id", "")),
        side=Side(str(row.get("action", "buy")).lower()), price=price, size=_count(row), fee=fee,
        ts_ms=int(_parse_iso(created).timestamp() * 1000) if created else 0,
        outcome=outcome, fill_id=str(row.get("trade_id", "")), tx_hash=None, builder_code=None,
    )


__all__ = ["Kalshi", "RsaPssSigner", "Signer", "auth_headers", "contract_from_market",
           "fill_from_row", "parse_orderbook", "parse_trade"]
