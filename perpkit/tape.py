"""A rolling window of recent trades — the 'tape'.

BloFin's `trades` channel pushes one trade per message with a `side` field.
On a perpetual futures feed, `side` is the **aggressor's** side: "buy" means a
market buy lifted someone's ask. That is exactly what we want, because
aggressive flow is what moves price. It saves us from having to infer trade
direction with the Lee-Ready tick rule.

The tape keeps trades from the last N seconds and can summarise them into the
trade-flow features the signal layer needs.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class Trade:
    ts: int          # exchange fill time, milliseconds
    price: float
    size: float
    is_buy: bool     # True if the aggressor was a buyer


class TradeTape:
    """Fixed-duration rolling trade history.

    Trimming is by exchange timestamp, not arrival time, so a burst of
    backfilled trades doesn't silently evict live ones.
    """

    def __init__(self, window_seconds: float = 60.0, max_trades: int = 20_000):
        self.window_ms = int(window_seconds * 1000)
        self.trades: Deque[Trade] = deque(maxlen=max_trades)
        self.last_ts: int = 0

    def add_message(self, data: Any) -> int:
        """Ingest the `data` field of a trades message. Returns count added."""
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return 0

        added = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            price = _to_float(item.get("price"))
            size = _to_float(item.get("size"))
            if price <= 0 or size <= 0:
                continue
            try:
                ts = int(item.get("ts", 0))
            except (TypeError, ValueError):
                continue
            side = str(item.get("side", "")).lower()
            self.trades.append(Trade(ts, price, size, is_buy=(side == "buy")))
            self.last_ts = max(self.last_ts, ts)
            added += 1

        if added:
            self._trim()
        return added

    def _trim(self) -> None:
        cutoff = self.last_ts - self.window_ms
        while self.trades and self.trades[0].ts < cutoff:
            self.trades.popleft()

    # ---- summaries -------------------------------------------------------

    def window(self, seconds: float) -> Tuple[float, float, int]:
        """(buy_volume, sell_volume, trade_count) over the last `seconds`.

        Volumes are in contracts, split by aggressor side.
        """
        if not self.trades:
            return 0.0, 0.0, 0
        cutoff = self.last_ts - int(seconds * 1000)
        buy_volume = sell_volume = 0.0
        count = 0
        # Walk backwards; the tape is time-ordered so we can stop early.
        for trade in reversed(self.trades):
            if trade.ts < cutoff:
                break
            if trade.is_buy:
                buy_volume += trade.size
            else:
                sell_volume += trade.size
            count += 1
        return buy_volume, sell_volume, count

    def flow_imbalance(self, seconds: float) -> float:
        """(buys - sells) / (buys + sells) over the window. Range [-1, 1].

        +1 = every contract traded in the window was an aggressive buy.
        0 with no trades, which is correct: no flow means no flow signal.
        """
        buy_volume, sell_volume, _ = self.window(seconds)
        total = buy_volume + sell_volume
        if total <= 0:
            return 0.0
        return (buy_volume - sell_volume) / total

    def volume(self, seconds: float) -> float:
        buy_volume, sell_volume, _ = self.window(seconds)
        return buy_volume + sell_volume

    def trade_count(self, seconds: float) -> int:
        return self.window(seconds)[2]

    def vwap(self, seconds: float) -> Optional[float]:
        if not self.trades:
            return None
        cutoff = self.last_ts - int(seconds * 1000)
        notional = volume = 0.0
        for trade in reversed(self.trades):
            if trade.ts < cutoff:
                break
            notional += trade.price * trade.size
            volume += trade.size
        if volume <= 0:
            return None
        return notional / volume

    def last_price(self) -> Optional[float]:
        return self.trades[-1].price if self.trades else None

    def staleness_seconds(self, now_ms: Optional[int] = None) -> float:
        """Wall-clock seconds since the last trade we saw. A tape that stops
        updating while the book keeps moving is a red flag for the feed.

        `now_ms` replaces `time.time()` for a replay (see `FeatureEngine.compute`)."""
        if not self.last_ts:
            return float("inf")
        now = time.time() if now_ms is None else now_ms / 1000.0
        return max(0.0, now - self.last_ts / 1000.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "trades": len(self.trades),
            "lastPrice": self.last_price(),
            "lastTs": self.last_ts,
            "stalenessSeconds": round(self.staleness_seconds(), 2),
        }
