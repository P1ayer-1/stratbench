"""Microstructure feature computation.

This is the layer the plan's `S = w1*OBI + w2*OFI + ...` sits on top of, and
the layer a LightGBM model would eventually be trained against. It turns the
raw book + tape into a fixed-width numeric vector, recomputed on every event.

Features produced
-----------------
  mid, microprice, microprice_delta_bps
  spread, spread_bps
  obi_1 / obi_5 / obi_20        order-book imbalance at three depths
  ofi_1s / ofi_5s               order flow imbalance (Cont et al. 2014)
  tfi_1s / tfi_5s / tfi_30s     aggressive trade-flow imbalance
  ret_1s / ret_5s / ret_30s     mid log-returns, in basis points
  rv_10s / rv_60s               realized volatility, per-second stdev
  trade_count_1s, volume_1s     activity
  funding_rate, funding_ttl_s   carry (matters at high leverage)
  book_age_ms, tape_staleness_s freshness guards

Every feature is finite and defined even with no data (returns 0.0 or None
explicitly) — a NaN leaking into a position-sizing formula at 20x leverage is
not an acceptable failure mode.

Note on horizons: BloFin's `books` channel updates at ~100ms and `tickers` at
~1s, so sub-100ms horizons are not observable from this feed. The shortest
horizon here is 1s deliberately. Claiming "10ms momentum" off a 100ms feed
would be fabricating precision the data doesn't contain.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .orderbook import OrderBook
from .tape import TradeTape


@dataclass
class FeatureSnapshot:
    """One row of the feature matrix. Keys here become CSV/parquet columns."""

    # Exchange event time (the book's own `ts`), milliseconds. This is the
    # clock labels and horizons are measured against, NOT wall-clock arrival
    # time — so the same code labels a live feed and a replayed historical
    # file identically. Backtesting depends on this distinction.
    ts: int = 0
    # Local wall-clock time the snapshot was built. Only for latency and
    # staleness checks, never for labelling.
    received_ts: int = 0

    mid: Optional[float] = None
    microprice: Optional[float] = None
    microprice_delta_bps: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0

    obi_1: float = 0.0
    obi_5: float = 0.0
    obi_20: float = 0.0

    ofi_1s: float = 0.0
    ofi_5s: float = 0.0

    tfi_1s: float = 0.0
    tfi_5s: float = 0.0
    tfi_30s: float = 0.0

    ret_1s: float = 0.0
    ret_5s: float = 0.0
    ret_30s: float = 0.0

    rv_10s: float = 0.0
    rv_60s: float = 0.0

    bid_depth_20: float = 0.0
    ask_depth_20: float = 0.0
    trade_count_1s: int = 0
    volume_1s: float = 0.0

    funding_rate: float = 0.0
    funding_ttl_s: float = 0.0

    vol_regime: str = "unknown"
    # Seconds of mid-price history behind this row. Returns and volatility over
    # a horizon longer than this are reported as 0.0 because the anchor was
    # never observed — which is indistinguishable from a genuine zero unless
    # you check this field. Filter training rows on
    # `history_seconds >= max(horizon)` or the model learns from padding.
    history_seconds: float = 0.0
    book_age_ms: int = 0
    tape_staleness_s: float = 0.0
    is_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _Series:
    """Time-ordered samples supporting O(log n) windowed sums.

    Exists for one reason: performance. The obvious implementation of these
    features rescans the whole history on every event — `sum(x for ts, x in
    history if ts >= cutoff)`. With a 300-second window that is fine at
    BloFin's ~10 book updates/second, but it is quadratic in the event rate,
    and on a fast feed (Binance bookTicker runs at 200-400 updates/second, so
    the window holds ~100k samples) `compute()` degrades to ~1000 events/sec —
    slow enough that replaying one day takes hours.

    Instead: timestamps are appended in order, so `bisect` finds a window
    boundary in O(log n), and a running cumulative sum turns a windowed sum
    into one subtraction. Old samples are dropped in batches, because deleting
    from the front of a list one element at a time is itself O(n).
    """

    __slots__ = ("ts", "values", "_cumulative")

    # Only compact once there is a worthwhile amount to drop; each compaction
    # is O(n) and doing it per event would reintroduce the original problem.
    _COMPACT_THRESHOLD = 4096

    def __init__(self) -> None:
        self.ts: List[int] = []
        self.values: List[float] = []
        # _cumulative[i] == sum(values[:i]), so it is always one longer.
        self._cumulative: List[float] = [0.0]

    def __len__(self) -> int:
        return len(self.ts)

    def append(self, ts: int, value: float) -> None:
        self.ts.append(ts)
        self.values.append(value)
        self._cumulative.append(self._cumulative[-1] + value)

    def sum_since(self, cutoff_ts: int) -> float:
        """Sum of values with ts >= cutoff_ts."""
        index = bisect_left(self.ts, cutoff_ts)
        return self._cumulative[-1] - self._cumulative[index]

    def index_since(self, cutoff_ts: int) -> int:
        return bisect_left(self.ts, cutoff_ts)

    def value_at_or_before(self, target_ts: int) -> Optional[float]:
        """Most recent value at or before target_ts, or None if the history
        does not reach back that far."""
        index = bisect_right(self.ts, target_ts) - 1
        if index < 0:
            return None
        return self.values[index]

    def compact(self, cutoff_ts: int) -> None:
        index = bisect_left(self.ts, cutoff_ts)
        if index < self._COMPACT_THRESHOLD:
            return
        del self.ts[:index]
        del self.values[:index]
        base = self._cumulative[index]
        self._cumulative = [value - base for value in self._cumulative[index:]]

    def pairs(self) -> List[Tuple[int, float]]:
        """(ts, value) tuples. O(n) — for diagnostics and tests, not the hot
        path."""
        return list(zip(self.ts, self.values))


class FeatureEngine:
    """Stateful feature calculator.

    Call `on_book_event()` / `on_trade_event()` as messages arrive, then
    `compute()` to get a snapshot. Rolling state is kept incrementally, and
    every windowed query is O(log n) via `_Series` — so `compute()` costs the
    same whether the feed delivers 10 or 1000 updates per second.
    """

    def __init__(
        self,
        *,
        history_seconds: float = 300.0,
        ofi_window_seconds: float = 5.0,
        vol_percentile_window: int = 600,
    ):
        self.history_ms = int(history_seconds * 1000)

        # Mid prices, driving returns and realized volatility.
        self._mid = _Series()
        # Squared log returns between consecutive mids, kept alongside so that
        # realized variance over a window is a single cumulative subtraction
        # rather than a rescan.
        self._squared_returns = _Series()
        # OFI increments; OFI over a window is their sum.
        self._ofi = _Series()
        self.ofi_window_ms = int(ofi_window_seconds * 1000)

        # Previous top-of-book, needed for the OFI recursion.
        self._prev_bid: Optional[float] = None
        self._prev_bid_size: float = 0.0
        self._prev_ask: Optional[float] = None
        self._prev_ask_size: float = 0.0

        # Rolling realized-vol samples, for percentile-based regime tagging.
        self._rv_samples: Deque[float] = deque(maxlen=vol_percentile_window)

        self.funding_rate: float = 0.0
        self.funding_time_ms: int = 0
        self.last_book_ts: int = 0

    # ---- compatibility views --------------------------------------------
    # Both are O(n) and exist for tests and diagnostics. Never call them per
    # event — that is exactly the pattern `_Series` was written to remove.

    @property
    def mid_history(self) -> List[Tuple[int, float]]:
        return self._mid.pairs()

    @property
    def ofi_history(self) -> List[Tuple[int, float]]:
        return self._ofi.pairs()

    # ---- event handlers --------------------------------------------------

    def on_book_event(self, book: OrderBook) -> None:
        """Update OFI and the mid-price history from a changed book."""
        if not book.is_ready or book.is_crossed():
            return

        bid, bid_size, ask, ask_size = book.best_bid_ask_size()
        if bid <= 0 or ask <= 0:
            return

        ts = book.ts or int(time.time() * 1000)
        self.last_book_ts = ts

        increment = self._ofi_increment(bid, bid_size, ask, ask_size)
        self._ofi.append(ts, increment)

        self._prev_bid, self._prev_bid_size = bid, bid_size
        self._prev_ask, self._prev_ask_size = ask, ask_size

        mid = (bid + ask) / 2.0
        # Squared log return against the previous mid, accumulated here so
        # realized volatility never has to recompute it.
        previous = self._mid.values[-1] if self._mid.values else None
        if previous is not None and previous > 0 and mid > 0:
            self._squared_returns.append(ts, math.log(mid / previous) ** 2)
        else:
            self._squared_returns.append(ts, 0.0)
        self._mid.append(ts, mid)
        self._trim(ts)

    def _ofi_increment(
        self, bid: float, bid_size: float, ask: float, ask_size: float
    ) -> float:
        """Order Flow Imbalance for one book event (Cont, Kukanov & Stoikov).

        The intuition: OFI measures net *pressure* added at the touch, and it
        distinguishes a level being refilled from the price level itself
        moving. A bid that ticks up contributes its whole new size (new demand
        appeared); a bid that ticks down removes the old size (demand
        vanished); a bid at an unchanged price contributes only the change.

        This is why OFI outperforms a raw size-imbalance snapshot: it is a
        flow, not a level.
        """
        if self._prev_bid is None or self._prev_ask is None:
            return 0.0

        if bid > self._prev_bid:
            bid_term = bid_size
        elif bid == self._prev_bid:
            bid_term = bid_size - self._prev_bid_size
        else:
            bid_term = -self._prev_bid_size

        if ask < self._prev_ask:
            ask_term = -ask_size
        elif ask == self._prev_ask:
            ask_term = -(ask_size - self._prev_ask_size)
        else:
            ask_term = self._prev_ask_size

        return bid_term + ask_term

    def on_funding(self, data: Any) -> None:
        """Ingest the `data` field of a funding-rate message."""
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            return
        try:
            self.funding_rate = float(data.get("fundingRate", 0.0))
        except (TypeError, ValueError):
            pass
        try:
            self.funding_time_ms = int(data.get("fundingTime", 0))
        except (TypeError, ValueError):
            pass

    def _trim(self, now_ms: int) -> None:
        cutoff = now_ms - self.history_ms
        self._mid.compact(cutoff)
        self._squared_returns.compact(cutoff)
        self._ofi.compact(now_ms - max(self.ofi_window_ms, 60_000))

    # ---- primitives ------------------------------------------------------

    def _mid_at(self, ts: int, seconds: float) -> Optional[float]:
        """The most recent mid at or before `seconds` ago.

        Returns None if our history doesn't reach back that far — better an
        explicit None than a return computed against a bogus anchor.
        """
        target = ts - int(seconds * 1000)
        if not self._mid.ts or self._mid.ts[0] > target:
            return None
        return self._mid.value_at_or_before(target)

    def _return_bps(self, ts: int, current_mid: float, seconds: float) -> float:
        past = self._mid_at(ts, seconds)
        if past is None or past <= 0 or current_mid <= 0:
            return 0.0
        return math.log(current_mid / past) * 10_000.0

    def _realized_vol(self, ts: int, seconds: float) -> float:
        """Per-second realized volatility from summed squared log returns.

        sigma = sqrt( sum(r_i^2) / elapsed_seconds )

        This is the standard realized-variance estimator, and dividing by
        elapsed time (rather than sample count) makes it comparable across
        periods with different update rates — important because book update
        frequency itself spikes during volatility.
        """
        cutoff = ts - int(seconds * 1000)
        start = self._mid.index_since(cutoff)
        if len(self._mid.ts) - start < 3:
            return 0.0

        # Squared returns were accumulated at append time. The first sample in
        # the window carries the return from *before* it, which belongs to the
        # previous window, so sum from start + 1.
        total = self._squared_returns.sum_since(self._mid.ts[start + 1])
        elapsed = (self._mid.ts[-1] - self._mid.ts[start]) / 1000.0
        if elapsed <= 0:
            return 0.0
        return math.sqrt(total / elapsed)

    def _ofi_sum(self, ts: int, seconds: float) -> float:
        return self._ofi.sum_since(ts - int(seconds * 1000))

    @staticmethod
    def _obi(bid_volume: float, ask_volume: float) -> float:
        """Order book imbalance in [-1, 1]. +1 = all resting size is on the bid."""
        total = bid_volume + ask_volume
        if total <= 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    def _vol_regime(self, rv: float) -> str:
        """Rule-of-thumb volatility regime by percentile of recent history.

        This is a deliberate placeholder for the HMM in the plan. It is honest
        about what it is: a percentile bucket, not a fitted latent-state model.
        It needs a full window before it says anything.
        """
        if rv > 0:
            self._rv_samples.append(rv)
        if len(self._rv_samples) < 60:
            return "unknown"
        ordered = sorted(self._rv_samples)
        low = ordered[int(len(ordered) * 0.33)]
        high = ordered[int(len(ordered) * 0.67)]
        if rv <= low:
            return "low_vol"
        if rv >= high:
            return "high_vol"
        return "normal_vol"

    # ---- main entry point ------------------------------------------------

    def compute(self, book: OrderBook, tape: TradeTape,
                now_ms: Optional[int] = None) -> FeatureSnapshot:
        """Build the current feature vector.

        `is_valid` is the gate the rest of the system must respect: it is only
        True when the book is sequence-clean, uncrossed, and two-sided. Every
        consumer should refuse to trade on an invalid snapshot rather than
        treating the zeros as real values.

        `now_ms` is the wall clock behind `received_ts`, `book_age_ms` and
        `tape_staleness_s`, and nothing else; live leaves it to `time.time()`,
        and a replay passes the archived clock (`perpkit/analysis/replay.py --clock`).
        """
        now = int(time.time() * 1000) if now_ms is None else now_ms
        snapshot = FeatureSnapshot(ts=self.last_book_ts or now, received_ts=now)

        if not book.is_ready or book.is_crossed():
            snapshot.book_age_ms = (
                now - self.last_book_ts if self.last_book_ts else 0
            )
            snapshot.tape_staleness_s = round(tape.staleness_seconds(now), 3)
            return snapshot

        bid, bid_size, ask, ask_size = book.best_bid_ask_size()
        if bid <= 0 or ask <= 0:
            return snapshot

        ts = book.ts or now
        snapshot.ts = ts
        mid = (bid + ask) / 2.0

        # Microprice: the size-weighted touch price. Bid is weighted by *ask*
        # size — when the ask is thin and the bid is heavy, fair value sits
        # closer to the ask, because that thin ask is what will break first.
        total_touch = bid_size + ask_size
        microprice = (
            (bid * ask_size + ask * bid_size) / total_touch if total_touch > 0 else mid
        )

        snapshot.mid = mid
        snapshot.microprice = microprice
        snapshot.microprice_delta_bps = (microprice - mid) / mid * 10_000.0
        snapshot.spread = ask - bid
        snapshot.spread_bps = (ask - bid) / mid * 10_000.0

        bid_1, ask_1 = bid_size, ask_size
        bid_5, ask_5 = book.depth_volume(5)
        bid_20, ask_20 = book.depth_volume(20)
        snapshot.obi_1 = self._obi(bid_1, ask_1)
        snapshot.obi_5 = self._obi(bid_5, ask_5)
        snapshot.obi_20 = self._obi(bid_20, ask_20)
        snapshot.bid_depth_20 = bid_20
        snapshot.ask_depth_20 = ask_20

        snapshot.ofi_1s = self._ofi_sum(ts, 1.0)
        snapshot.ofi_5s = self._ofi_sum(ts, 5.0)

        snapshot.tfi_1s = tape.flow_imbalance(1.0)
        snapshot.tfi_5s = tape.flow_imbalance(5.0)
        snapshot.tfi_30s = tape.flow_imbalance(30.0)
        snapshot.trade_count_1s = tape.trade_count(1.0)
        snapshot.volume_1s = tape.volume(1.0)

        snapshot.ret_1s = self._return_bps(ts, mid, 1.0)
        snapshot.ret_5s = self._return_bps(ts, mid, 5.0)
        snapshot.ret_30s = self._return_bps(ts, mid, 30.0)

        snapshot.rv_10s = self._realized_vol(ts, 10.0)
        snapshot.rv_60s = self._realized_vol(ts, 60.0)
        snapshot.vol_regime = self._vol_regime(snapshot.rv_60s)

        snapshot.funding_rate = self.funding_rate
        if self.funding_time_ms:
            snapshot.funding_ttl_s = max(
                0.0, (self.funding_time_ms - snapshot.ts) / 1000.0
            )

        if self._mid.ts:
            snapshot.history_seconds = round((ts - self._mid.ts[0]) / 1000.0, 2)

        # Latency between the exchange stamping the book and us processing it.
        snapshot.book_age_ms = max(0, now - ts)
        snapshot.tape_staleness_s = round(tape.staleness_seconds(now), 3)
        snapshot.is_valid = True
        return snapshot


def feature_columns() -> List[str]:
    """Column order for the recorder. Stable across runs so appended CSV
    files from different sessions stay concatenable."""
    return list(FeatureSnapshot().to_dict().keys())
