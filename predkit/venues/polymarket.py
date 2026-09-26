"""Polymarket (global): Gamma for listings, the CLOB for books and orders.

Endpoints, checked live 2026-09-12:
  Gamma      https://gamma-api.polymarket.com/markets      listings, one row per market
  CLOB       https://clob.polymarket.com                    /book, /markets/<condition>, orders
  Data API   https://data-api.polymarket.com/trades         public trade prints
  WS         wss://ws-subscriptions-clob.polymarket.com/ws/market

A market is a `conditionId` with two ERC-1155 tokens, one per outcome, each
with its own book. The YES token's book is already YES-denominated; the NO
token's book is converted here. Tick is 1 cent on the crypto markets (some
markets step to 0.001 near the extremes; the CLOB reports it per token) and
the minimum order is 5 contracts.

Orders are signed EIP-712 by `py_clob_client` with the user's own key, which
never leaves this process; the adapter refuses to send without a client.
Builder attribution is `py_clob_client`'s `BuilderConfig`: the Builder
Program's API credentials, passed at construction. Without them, orders are
unattributed, which is fine for the user.

Fee tier is Gamma's own `feeType` string (`crypto_fees_v2` on the 5- and
15-minute BTC markets, `economics_fees` on a Fed market; read live
2026-09-13), copied verbatim onto the contract so `fees.py` can refuse by
name anything it has not transcribed. A row with no `feeType` gets the
tier `unknown`, which is absent there.

Resolution: the 5-minute BTC market row carries `resolutionSource`
"https://data.chain.link/streams/btc-usd-twap-60s-streams" (read live
2026-09-13); rows without one get `polymarket:uma:<conditionId>`. Either
way it is the venue's source, never a spot exchange.

The rolling windows are slugs: `btc-updown-5m-<unix start aligned to
300 s>` and `btc-updown-15m-<aligned to 900 s>`, looked up with
`/markets?slug=`; windows are listed hours ahead. `record_series.py` uses
that to find the current and next windows.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

from predkit.schema import (Book, Contract, Fill, Level, OrderIntent, Outcome, Side, Trade,
                            check_price, no_to_yes, to_decimal)
from predkit.venues import NotSignable

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
WS_MARKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_DATA_PRICE_QUANTUM = Decimal("0.0001")     # the finest CLOB tick
_DATA_SIZE_QUANTUM = Decimal("0.000001")


def _parse_iso(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _json_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value or [])


def contract_from_gamma(row: Dict[str, Any]) -> Contract:
    tokens = _json_list(row.get("clobTokenIds"))
    outcomes = [o.lower() for o in _json_list(row.get("outcomes"))]
    yes_token = no_token = None
    if len(tokens) == 2 and len(outcomes) == 2:
        yes_index = outcomes.index("yes") if "yes" in outcomes else 0
        yes_token, no_token = tokens[yes_index], tokens[1 - yes_index]
    resolves = row.get("endDate") or row.get("endDateIso") or row.get("end_date_iso")
    if not resolves:
        raise ValueError(f"gamma market {row.get('conditionId')} has no endDate")
    condition = row.get("conditionId") or row.get("condition_id")
    # Raw fee fields, kept beside the tier so a change in the venue's
    # schedule is visible in the archive's contract.json.
    extra = {k: row[k] for k in ("takerBaseFee", "makerBaseFee", "feeType", "feeSchedule",
                                 "resolvedBy", "negRisk", "slug") if row.get(k) is not None}
    return Contract(
        venue="polymarket",
        market_id=condition,
        question=row.get("question", ""),
        resolves_at=_parse_iso(resolves),
        resolution_source=(row.get("resolutionSource") or "").strip()
        or f"polymarket:uma:{condition}",
        fee_tier=str(row.get("feeType") or "unknown"),
        tick=to_decimal(row.get("orderPriceMinTickSize") or "0.01"),
        min_size=to_decimal(row.get("orderMinSize") or "5"),
        yes_token=yes_token,
        no_token=no_token,
        extra=extra,
    )


class Polymarket:
    name = "polymarket"

    def __init__(self, *, http: Optional[httpx.Client] = None, clob: Any = None,
                 gamma_url: str = GAMMA, clob_url: str = CLOB, data_url: str = DATA_API,
                 ws_url: str = WS_MARKET):
        self.http = http or httpx.Client(timeout=20.0, headers={"User-Agent": "predkit/0.1"})
        self.clob = clob            # a py_clob_client.ClobClient with the user's key, or None
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.data_url = data_url
        self.ws_url = ws_url
        self._contracts: Dict[str, Contract] = {}

    # ---- read ------------------------------------------------------------

    def _get(self, url: str, **params) -> Any:
        response = self.http.get(url, params=params or None)
        response.raise_for_status()
        return response.json()

    def list_markets(self, query: Optional[str] = None, *, limit: int = 50) -> List[Contract]:
        rows = self._get(f"{self.gamma_url}/markets", active="true", closed="false",
                         limit=limit, order="volume24hr", ascending="false")
        contracts = []
        for row in rows:
            if query and query.lower() not in (row.get("question", "") + row.get("slug", "")).lower():
                continue
            try:
                contract = contract_from_gamma(row)
            except (ValueError, KeyError):
                continue
            self._contracts[contract.market_id] = contract
            contracts.append(contract)
        return contracts

    def _row(self, market_id: str) -> Dict[str, Any]:
        """The Gamma row, live or closed: a closed market is not returned by
        the default listing (read live 2026-09-13), so ask again with
        `closed=true` before giving up."""
        rows = self._get(f"{self.gamma_url}/markets", condition_ids=market_id)
        if not rows:
            rows = self._get(f"{self.gamma_url}/markets", condition_ids=market_id, closed="true")
        if not rows:
            raise KeyError(f"polymarket: no market with conditionId {market_id}")
        return rows[0]

    def market(self, market_id: str) -> Contract:
        if market_id in self._contracts:
            return self._contracts[market_id]
        contract = contract_from_gamma(self._row(market_id))
        self._contracts[market_id] = contract
        return contract

    def resolution(self, market_id: str) -> Optional[bool]:
        """True/False once the market is closed with a settled outcome, else
        None. Read from `outcomePrices`, which a resolved row sets to
        `["1","0"]` or `["0","1"]` in `outcomes` order (read live 2026-09-13
        on a settled 5-minute window)."""
        row = self._row(market_id)
        if not row.get("closed"):
            return None
        prices = [str(p) for p in _json_list(row.get("outcomePrices"))]
        outcomes = [o.lower() for o in _json_list(row.get("outcomes"))]
        if len(prices) != 2 or sorted(prices) != ["0", "1"]:
            return None
        yes_index = outcomes.index("yes") if "yes" in outcomes else 0
        return prices[yes_index] == "1"

    def market_by_slug(self, slug: str) -> Optional[Contract]:
        """One Gamma row by slug, or None if the venue has not listed it."""
        rows = self._get(f"{self.gamma_url}/markets", slug=slug)
        if not rows:
            return None
        contract = contract_from_gamma(rows[0])
        self._contracts[contract.market_id] = contract
        return contract

    def book(self, market_id: str) -> Book:
        contract = self.market(market_id)
        if not contract.yes_token:
            raise ValueError(f"{market_id}: no YES token id on the listing")
        raw = self._get(f"{self.clob_url}/book", token_id=contract.yes_token)
        return parse_book(raw, market_id, is_yes_token=True)

    def trades(self, market_id: str, *, limit: int = 100) -> List[Trade]:
        contract = self.market(market_id)
        rows = self._get(f"{self.data_url}/trades", market=market_id, limit=limit)
        out = []
        for row in rows:
            # The data API serves floats (0.2099999969 for a 21c print, read
            # live 2026-09-13). Quantised to the finest CLOB tick, or the fill
            # model would read that print as strictly below a 0.21 bid.
            price = check_price(row["price"]).quantize(_DATA_PRICE_QUANTUM)
            is_yes = (row.get("asset") == contract.yes_token) or \
                     (str(row.get("outcome", "")).lower() == "yes")
            side = Side(row["side"].lower())
            if not is_yes:            # a NO print, expressed on YES
                price = no_to_yes(price)
                side = Side.SELL if side is Side.BUY else Side.BUY
            out.append(Trade(market_id=market_id, price=price,
                             size=to_decimal(row["size"]).quantize(_DATA_SIZE_QUANTUM), aggressor=side,
                             ts_ms=int(float(row.get("timestamp", 0)) * 1000),
                             trade_id=str(row.get("transactionHash") or row.get("id", ""))))
        return out

    async def stream(self, market_id: str, *, connections: int = 2) -> AsyncIterator[Tuple[str, Any]]:
        """Raw CLOB market-channel frames, tagged by `event_type`, from
        `connections` redundant sockets merged and de-duplicated.

        Why two sockets. In one 8.5 h recording the venue closed a live-window
        socket with 1013 "slow consumer: send buffer full" 17 times, then
        twice more in ten minutes after the reader was made trivially cheap
        (no parse on the loop, writer thread, receive queue 4096, loop lag
        under 25 ms, link at 15 of 300 Mbps). Two of the drops hit two
        connections within three seconds: a server-side event, not the client.
        Each reconnect cost 1.5 to 14 s of book updates. So every window
        holds two independent subscriptions; identical frames are dropped
        over the last 131,072 (`redundant_stream` says why that many), and
        a socket that dies is reopened at once while the other keeps
        streaming. Both tokens stay subscribed on each: YES-only saved 7% of
        bytes and lost the NO-token trade prints (144 of 257 in a minute).

        The one thing de-duplication can hide: two trades with identical
        price, size, side and millisecond on the same token would be one
        `last_trade_price` frame. Accepted; the data API has the full list.

        `max_queue` is the library's receive buffer (verified 4096/1024 high
        and low water); `ping_timeout` is the dead-socket detector and the
        recorder's stall deadline only a backstop behind it.
        """
        import websockets  # lazy: the package must import without it

        contract = self.market(market_id)
        assets = [t for t in (contract.yes_token, contract.no_token) if t]
        subscribe = json.dumps({"assets_ids": assets, "type": "market"})

        def connect():
            return websockets.connect(self.ws_url, ping_interval=10, ping_timeout=10, max_queue=4096)

        async for item in redundant_stream(connect, subscribe, connections=connections,
                                           label=f"polymarket:{market_id[:10]}"):
            yield item

    # ---- write -----------------------------------------------------------

    def _require_clob(self) -> Any:
        if self.clob is None:
            raise NotSignable("polymarket: no signing client. Build one from the user's own "
                              "key with predkit.keys and pass it as `clob`; this adapter never "
                              "reads a key itself.")
        return self.clob

    def place_order(self, intent: OrderIntent) -> str:
        from py_clob_client.clob_types import OrderArgs, OrderType

        clob = self._require_clob()
        token = intent.contract.yes_token if intent.outcome is Outcome.YES else intent.contract.no_token
        if not token:
            raise ValueError(f"{intent.contract.market_id}: no token id for {intent.outcome.value}")
        args = OrderArgs(token_id=token, price=float(intent.price), size=float(intent.size),
                         side=intent.side.value.upper())
        signed = clob.create_order(args)
        result = clob.post_order(signed, OrderType.GTC, post_only=intent.post_only)
        order_id = (result or {}).get("orderID") or (result or {}).get("orderId")
        if not order_id:
            raise RuntimeError(f"polymarket rejected the order: {result}")
        return str(order_id)

    def cancel(self, order_id: str) -> None:
        self._require_clob().cancel(order_id)

    def open_orders(self, market_id: Optional[str] = None) -> List[dict]:
        from py_clob_client.clob_types import OpenOrderParams

        params = OpenOrderParams(market=market_id) if market_id else None
        return list(self._require_clob().get_orders(params) or [])

    def fills(self, market_id: Optional[str] = None) -> List[Fill]:
        from py_clob_client.clob_types import TradeParams

        params = TradeParams(market=market_id) if market_id else None
        rows = self._require_clob().get_trades(params) or []
        return [fill_from_clob(row) for row in rows]


# ---- parsing, usable by replay without a client ---------------------------

def parse_book(raw: Dict[str, Any], market_id: str, *, is_yes_token: bool) -> Book:
    """A CLOB `/book` payload or a WS `book` event. On the NO token, bids
    become YES asks at 1 - price and vice versa."""
    bids = [Level(l["price"], l["size"]) for l in raw.get("bids", [])]
    asks = [Level(l["price"], l["size"]) for l in raw.get("asks", [])]
    if not is_yes_token:
        bids, asks = ([Level(no_to_yes(l.price), l.size) for l in asks],
                      [Level(no_to_yes(l.price), l.size) for l in bids])
    ts = raw.get("timestamp") or 0
    return Book(market_id=market_id, bids=bids, asks=asks, ts_ms=int(ts))


def parse_trade_event(message: Dict[str, Any], market_id: str, yes_token: Optional[str]) -> Trade:
    """A WS `last_trade_price` event."""
    price = check_price(message["price"])
    side = Side(str(message.get("side", "BUY")).lower())
    if yes_token and message.get("asset_id") != yes_token:
        price = no_to_yes(price)
        side = Side.SELL if side is Side.BUY else Side.BUY
    return Trade(market_id=market_id, price=price, size=message.get("size", "0"),
                 aggressor=side, ts_ms=int(message.get("timestamp", 0)))


def fill_from_clob(row: Dict[str, Any]) -> Fill:
    return Fill(
        venue="polymarket",
        market_id=str(row.get("market", "")),
        order_id=str(row.get("taker_order_id") or row.get("order_id", "")),
        side=Side(str(row.get("side", "BUY")).lower()),
        price=row["price"],
        size=row["size"],
        fee=row.get("fee_rate_bps", 0) and (to_decimal(row["price"]) * to_decimal(row["size"])
                                             * Decimal(row["fee_rate_bps"]) / Decimal(10000))
        or Decimal(0),
        ts_ms=int(float(row.get("match_time", 0)) * 1000),
        outcome=Outcome(str(row.get("outcome", "Yes")).lower()),
        fill_id=str(row.get("id", "")),
        tx_hash=row.get("transaction_hash"),
        builder_code=row.get("builder"),
    )


async def redundant_stream(connect: Callable[[], Any], subscribe: str, *, connections: int = 2,
                           label: str = "", dedupe_window: int = 131_072,
                           sleep: Callable[[float], Any] = asyncio.sleep) -> AsyncIterator[Tuple[str, Any]]:
    """Merge `connections` sockets opened by `connect()` (an async context
    manager yielding an async-iterable of text frames), each sent
    `subscribe` on open, into one de-duplicated stream of
    `(event_type, frame)`. A socket that raises is reopened after a short,
    growing pause; the merged stream only stops delivering if every socket
    is down at once. Drops are counted on `redundant_stream.drops[label]`
    and logged, so the venue's behaviour stays visible even though the
    recorder no longer sees them as feed errors.

    A frame is dropped when its 16-byte BLAKE2b digest is among the last
    `dedupe_window` delivered. The window was 4,096 frames of string
    equality at first, until the archive showed the lagging socket's
    copies landing 5-13 s after the first (about 4% of 5-minute frames);
    at 350 frames/s baseline and 1,400/s peaks 13 s is up to ~18,000 frames. 131,072 covers
    about 90 s at that peak. Digests keep it at ~18 MB a stream where the
    frames themselves would be ~80 MB; one costs 2.3 us (measured), a third
    of a millisecond a second at the peak. Replay drops the copies archived
    by an older, narrower window (`replay.iter_market_events`)."""
    queue: "asyncio.Queue[Tuple[int, Any]]" = asyncio.Queue()

    async def pump(index: int) -> None:
        backoff = 0.0
        while True:
            try:
                async with connect() as socket:
                    await socket.send(subscribe)
                    backoff = 0.0
                    async for raw in socket:
                        await queue.put((index, raw))
                await queue.put((index, ConnectionError("socket closed cleanly")))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await queue.put((index, exc))
            await sleep(backoff)
            backoff = min(backoff + 0.5, 5.0)

    tasks = [asyncio.create_task(pump(i), name=f"{label}#{i}") for i in range(connections)]
    seen: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()
    try:
        while True:
            index, item = await queue.get()
            if isinstance(item, BaseException):
                redundant_stream.drops[label] = redundant_stream.drops.get(label, 0) + 1
                print(f"  [{label}] connection {index} dropped: {type(item).__name__}: {str(item)[:120]}; "
                      f"{connections - 1} other(s) still streaming", flush=True)
                continue
            if item.startswith("["):
                # The connect-time batch: one array frame holding both tokens'
                # snapshots. One per (re)connection, never de-duplicated: a
                # snapshot is state and applying it twice is harmless.
                for message in _split_batch(item):
                    yield (str(message.get("event_type", "unknown")), message)
                continue
            digest = hashlib.blake2b(item.encode(), digest_size=16).digest()
            if digest in seen:
                continue
            seen[digest] = None
            if len(seen) > dedupe_window:
                seen.popitem(last=False)
            # Every other frame goes to the archive verbatim, tagged by a
            # substring scan: no json.loads on the loop. Measured on a live window:
            # ~350 frames/s baseline and 1,400/s peaks on a live 5-minute window.
            yield (_event_type(item), item)
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


redundant_stream.drops = {}   # type: ignore[attr-defined]


_EVENT_TYPE = re.compile(r'"event_type"\s*:\s*"([A-Za-z0-9_]+)"')


def _event_type(raw: str) -> str:
    """The frame's `event_type`, found by a regex scan, not a parse. About
    a microsecond on a 700-byte frame; tolerant of the venue's spacing."""
    found = _EVENT_TYPE.search(raw)
    return found.group(1) if found else "unknown"


def _split_batch(raw: str) -> List[Dict[str, Any]]:
    """The market channel sends single objects and, at connect, arrays."""
    payload = json.loads(raw)
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    return [payload] if isinstance(payload, dict) else []


__all__ = ["Polymarket", "contract_from_gamma", "fill_from_clob", "parse_book", "parse_trade_event",
           "redundant_stream"]
