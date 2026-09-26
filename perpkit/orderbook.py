"""A maintained L2 limit order book, fed from BloFin's `books` websocket channel.

BloFin sends one full `snapshot` when you subscribe, then `update` messages
every ~100ms containing only the price levels that changed. This module
applies those updates and keeps a correctly sorted book:

    bids: descending  (best/highest bid first)
    asks: ascending   (best/lowest ask first)

The important, easy-to-get-wrong part is **sequence continuity**. Each message
carries `seqId` and `prevSeqId`. If an update's `prevSeqId` doesn't match the
`seqId` we last applied, we silently missed a message and our book is now
wrong. A wrong book produces wrong features, which produce wrong trades. So
instead of guessing, the book marks itself stale and refuses to serve data
until a fresh snapshot arrives.

Prices and sizes are floats here, not Decimal. That is deliberate: this is the
hot path (hundreds of updates/sec), every feature computed from it is a ratio
or a difference where float precision is irrelevant, and Decimal arithmetic is
~50x slower. Money math that must be exact (liquidation prices, position
sizing) lives in perpkit/risk.py and uses Decimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _to_float(value: Any) -> float:
    """BloFin is inconsistent about sending numbers vs numeric strings."""
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    """Live L2 book for a single instrument.

    Usage:
        book = OrderBook()
        book.apply(message)          # from the websocket
        if book.is_ready:
            bid, ask = book.best_bid_ask()

    `is_ready` is False before the first snapshot and after any detected gap,
    so callers never act on a book that has silently drifted.
    """

    # price -> size. Kept as dicts for O(1) updates; sorted on read.
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)

    seq_id: int = 0
    ts: int = 0
    ready: bool = False

    # Diagnostics — worth watching in production. A climbing resync_count
    # means the feed or the network is unhealthy.
    update_count: int = 0
    resync_count: int = 0
    last_gap_reason: Optional[str] = None

    # ---- state -----------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True only if we hold a book we believe is an accurate mirror."""
        return self.ready and bool(self.bids) and bool(self.asks)

    def mark_stale(self, reason: str) -> None:
        """Drop the book. The caller should resubscribe to force a snapshot."""
        self.ready = False
        self.last_gap_reason = reason
        self.resync_count += 1

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.seq_id = 0
        self.ts = 0
        self.ready = False

    # ---- ingestion -------------------------------------------------------

    def apply(self, message: Dict[str, Any]) -> bool:
        """Apply one raw websocket message. Returns True if the book changed.

        Accepts the whole message ({"arg":..., "action":..., "data":...}) so
        callers don't have to know BloFin's envelope shape.
        """
        action = message.get("action") or "snapshot"
        data = message.get("data")

        # `books` sends a dict; some feeds/tests wrap it in a single-item list.
        if isinstance(data, list):
            if not data:
                return False
            data = data[0]
        if not isinstance(data, dict):
            return False

        if action == "snapshot":
            return self._apply_snapshot(data)
        return self._apply_update(data)

    def _apply_snapshot(self, data: Dict[str, Any]) -> bool:
        self.bids = {
            price: size
            for price, size in (
                (_to_float(level[0]), _to_float(level[1]))
                for level in data.get("bids", [])
                if len(level) >= 2
            )
            if size > 0
        }
        self.asks = {
            price: size
            for price, size in (
                (_to_float(level[0]), _to_float(level[1]))
                for level in data.get("asks", [])
                if len(level) >= 2
            )
            if size > 0
        }
        self.seq_id = _to_int(data.get("seqId"))
        self.ts = _to_int(data.get("ts"))
        self.ready = True
        self.last_gap_reason = None
        self.update_count += 1
        return True

    def _apply_update(self, data: Dict[str, Any]) -> bool:
        # An update before any snapshot is useless — we have no base to apply
        # it to. Stay stale rather than building a partial, wrong book.
        if not self.ready:
            return False

        prev_seq = _to_int(data.get("prevSeqId"), -1)
        new_seq = _to_int(data.get("seqId"), -1)

        # The gap check. BloFin guarantees prevSeqId of an update equals the
        # seqId of the message before it. Anything else means we lost data.
        if prev_seq != -1 and prev_seq != self.seq_id:
            # A repeat of a message we already applied is harmless, just skip.
            if new_seq != -1 and new_seq <= self.seq_id:
                return False
            self.mark_stale(
                f"sequence gap: expected prevSeqId={self.seq_id}, got {prev_seq}"
            )
            return False

        for level in data.get("bids", []):
            if len(level) >= 2:
                self._set_level(self.bids, _to_float(level[0]), _to_float(level[1]))
        for level in data.get("asks", []):
            if len(level) >= 2:
                self._set_level(self.asks, _to_float(level[0]), _to_float(level[1]))

        if new_seq != -1:
            self.seq_id = new_seq
        self.ts = _to_int(data.get("ts"), self.ts)
        self.update_count += 1
        return True

    @staticmethod
    def _set_level(side: Dict[float, float], price: float, size: float) -> None:
        """size == 0 is BloFin's way of saying 'this level is gone'."""
        if size <= 0:
            side.pop(price, None)
        else:
            side[price] = size

    # ---- reads -----------------------------------------------------------

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.bids or not self.asks:
            return None, None
        return max(self.bids), min(self.asks)

    def best_bid_ask_size(self) -> Tuple[float, float, float, float]:
        """(bid_price, bid_size, ask_price, ask_size). Zeros if not ready."""
        bid, ask = self.best_bid_ask()
        if bid is None or ask is None:
            return 0.0, 0.0, 0.0, 0.0
        return bid, self.bids[bid], ask, self.asks[ask]

    def top(self, depth: int) -> Tuple[List[BookLevel], List[BookLevel]]:
        """The best `depth` levels per side, correctly ordered."""
        bids = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[:depth]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:depth]
        return (
            [BookLevel(price, size) for price, size in bids],
            [BookLevel(price, size) for price, size in asks],
        )

    def depth_volume(self, depth: int) -> Tuple[float, float]:
        """Total resting size within the top `depth` levels of each side."""
        bids, asks = self.top(depth)
        return sum(level.size for level in bids), sum(level.size for level in asks)

    def mid(self) -> Optional[float]:
        bid, ask = self.best_bid_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread(self) -> Optional[float]:
        bid, ask = self.best_bid_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def is_crossed(self) -> bool:
        """A crossed book (bid >= ask) means our state is corrupt or the feed
        is mid-update. Never trade on one."""
        bid, ask = self.best_bid_ask()
        if bid is None or ask is None:
            return False
        return bid >= ask
