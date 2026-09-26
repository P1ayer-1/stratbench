"""One adapter per venue: markets, book, trades, orders, fills, stream.

The adapters are the only modules that know a venue's wire format, and the
only modules with a path to an order endpoint. Everything they return is in
`predkit.schema` terms, priced as YES (a Kalshi NO bid at 52 cents comes
out as a YES ask at 0.48), so nothing above them handles a NO price.

Each adapter takes an injectable `httpx.Client`, so the unit tests run
against `httpx.MockTransport` with recorded payloads and never touch the
network. Streams import `websockets` lazily, inside the method, so the
package imports without it.

The contract is a Protocol, satisfied structurally. There is no venue base
class for the same reason there is no strategy base class: two venues is
one too few to know what they share.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from predkit.schema import Book, Contract, Fill, OrderIntent, Trade


@runtime_checkable
class Venue(Protocol):
    name: str

    def list_markets(self, query: Optional[str] = None, *, limit: int = 50) -> List[Contract]: ...
    def market(self, market_id: str) -> Contract: ...
    def book(self, market_id: str) -> Book: ...
    def trades(self, market_id: str, *, limit: int = 100) -> List[Trade]: ...
    def stream(self, market_id: str) -> AsyncIterator[Tuple[str, object]]: ...
    def place_order(self, intent: OrderIntent) -> str: ...
    def cancel(self, order_id: str) -> None: ...
    def open_orders(self, market_id: Optional[str] = None) -> List[dict]: ...
    def fills(self, market_id: Optional[str] = None) -> List[Fill]: ...


class NotSignable(RuntimeError):
    """The adapter has no signer: it can read the venue but not send to it."""


__all__ = ["NotSignable", "Venue"]
