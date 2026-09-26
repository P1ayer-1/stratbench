"""Records BloFin's mark and index price - the basis, which nothing else sees.

Why this exists
---------------
`ingest.py` records the order book, so it knows the *traded* price. It does
not know the **mark** price, and the gap between them is the basis: what the
perpetual is worth against its own index rather than against its book.

That gap is not decoration. `perpkit/analysis/funding_carry.py` prices it as
the `conv` column - the cost a carry pays if the basis converges while the
position is on - and without this recorder it would be the
one input to that number this repo cannot observe over time. The carry's
liquidation price is struck against the mark too, not against the book, so a
position can be liquidated at a price that never printed.

There is no websocket channel for it
------------------------------------
Measured 2026-09-10, not assumed. BloFin's public socket refuses every
plausible name with `60012 Invalid request`:

    mark-price   index-price   index-tickers
    open-interest   price-limit   liquidation-orders

The `tickers` channel exists but carries only top of book, last trade and 24h
stats - all of which `books` and `trades` already provide. So mark price is
REST-only, and polling is not a shortcut here, it is the only route.

What the endpoint actually does (measured, not assumed)
-------------------------------------------------------
  * `GET /api/v1/market/mark-price` returns `markPrice` AND `indexPrice`
    together, so one request covers both series.
  * Called with **no instId it returns every instrument** - 486 of them in one
    response, so 15 symbols cost one request, not 15.
  * Unlike open interest it updates **continuously** - the `ts` moved on every
    3-second probe. There is no minute boundary to align to, so the poll
    interval IS the sample rate, and the `ts` dedupe only protects against
    polling faster than the publisher.

Capture-or-lose, like everything else here
------------------------------------------
The endpoint returns a snapshot and there is no history for it. Every hour
recorded without this is an hour whose basis can never be reconstructed - not
from the book, which does not contain it, and not from anywhere else.
"""

from __future__ import annotations

from .openinterest import SnapshotPoller

CHANNEL = "mark-price"
MARK_PRICE_PATH = "/api/v1/market/mark-price"

# The series updates continuously rather than on a boundary, so this is a
# sampling decision rather than a "catch the publish" one. 10s is ~8,600 rows
# per instrument per day - negligible beside the book's ~140 MB - and fine
# enough to see basis move within a funding interval.
DEFAULT_POLL_SECONDS = 10.0


class MarkPricePoller(SnapshotPoller):
    """Mark and index price for a set of instruments, sampled and archived.

    Shares `SnapshotPoller`'s exclusive lock, which is named after the
    CHANNEL - so this runs safely alongside an open-interest poller on the
    same instrument, and still refuses a second mark-price poller on it.
    """

    CHANNEL = CHANNEL
    PATH = MARK_PRICE_PATH

    def __init__(self, instruments, *, poll_seconds: float = DEFAULT_POLL_SECONDS,
                 **kwargs):
        super().__init__(instruments, poll_seconds=poll_seconds, **kwargs)
