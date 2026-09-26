"""The same daily panel schema, built from Hyperliquid.

    python -m perpkit.analysis.panel_hyperliquid --top 60
    python -m perpkit.analysis.factor_panel --panel data\\panel\\hyperliquid-daily.csv

Why another venue
-----------------
**A small venue gives a thin cross-section.** BloFin's 47 instruments give a
median of about 15 eligible names on a given day, so a cross-sectional study
there has few positions a side and wide intervals. Hyperliquid lists 234 perps
with 37 over $5M a day - a cross-section of the same order - over a different
user base, so it gives a second independent reading of a cross-sectional claim
rather than a wider cut of the first.

**Cross-venue comparisons need a second funding series.** Hyperliquid is
permissionless (an address is an account), and a panel in the same schema lets
its funding be compared day by day against any other venue's.

What Hyperliquid's data costs, and the one thing that makes it slow
------------------------------------------------------------------
`metaAndAssetCtxs` gives every perp with its 24h notional volume in one call.
`candleSnapshot` at `1d` returns the whole history in one call - 1,351 days
back to 2023-01-01.

`fundingHistory` is the expensive one. Hyperliquid funds **hourly**, and the
endpoint returns 500 records a call, which is 20.8 days: covering 2023-05 to
now takes about 59 calls per coin. Its weight is 20 against a 1,200-per-minute
IP budget, so the ceiling is one call a second and a 60-coin panel is about an
hour. Everything is cached per coin so that cost is paid once.

Hourly funding is not a detail to normalise away
------------------------------------------------
Twenty-four settlements a day against Binance's and BloFin's three. A daily
total is still the right unit - it is what a position actually paid that day,
whatever the cadence - and it is why every panel builder carries `funding_bps` per
day rather than a per-settlement rate. Comparing rates instead of daily totals
would read Hyperliquid as eight times cheaper than it is.

The same alignment rule holds: a settlement stamped T paid for the interval
ending at T, so it is bucketed just before T. `v` is base volume and `n` the
trade count, so unlike BloFin this panel can fill `trades`; there is still no
taker/maker split, and `rv_bps` is again the Parkinson high/low estimator.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


from perpkit.analysis.panel import write_panel
from perpkit.hyperliquid import fetch_universe, post_info

from perpkit.config import DATA_DIR
CACHE = DATA_DIR / "cache" / "hyperliquid"
DEFAULT_OUT = DATA_DIR / "panel" / "hyperliquid-daily.csv"

# The archive starts here; asking for earlier just returns the earliest page.
HISTORY_START_MS = 1_672_531_200_000          # 2023-01-01
FUNDING_PAGE = 500                            # records per fundingHistory call
HOUR_MS = 3_600_000
# fundingHistory is weight 20 against 1,200 per minute, so one call a second is
# the ceiling. Slightly over, because a 429 costs far more than the pause does.
DEFAULT_PAUSE = 1.05

TRUNCATED: set = set()


def _day_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d")


@dataclass
class Coin:
    name: str
    volume_usd: float
    max_leverage: float


def liquid_coins(*, top: int, min_volume: float,
                 opener: Optional[Callable[..., Any]] = None) -> List[Coin]:
    """The `top` perps by 24h notional volume.

    Same survivorship caveat as every other panel builder:
    ranking by volume today selects the contracts that survived and grew.
    """
    assets, contexts = fetch_universe(opener=opener)
    out: List[Coin] = []
    for name, context in contexts.items():
        try:
            volume = float(context.get("dayNtlVlm") or 0.0)
        except (TypeError, ValueError):
            continue
        if volume < min_volume:
            continue
        asset = assets.get(name) or {}
        if asset.get("isDelisted"):
            continue
        try:
            leverage = float(asset.get("maxLeverage") or 0.0)
        except (TypeError, ValueError):
            leverage = 0.0
        out.append(Coin(name, volume, leverage))
    out.sort(key=lambda item: item.volume_usd, reverse=True)
    return out[:top]


def _cache_path(coin: str, kind: str, cache: Path) -> Path:
    return cache / coin / (kind + ".json")


def _read_cache(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _write_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def fetch_candles(coin: str, *, cache: Path, refresh: bool, now_ms: int,
                  opener: Optional[Callable[..., Any]] = None,
                  ) -> List[Tuple[int, float, float, float, float, float, float]]:
    """`(day_start_ms, open, high, low, close, quote_volume, trades)`, oldest first.

    Hyperliquid stamps a candle with the start of its interval (`t`) and carries
    the end in `T`. The shared schema keys a row on the day it CLOSED in, and
    for a daily candle those are the same UTC date - but the distinction matters
    enough to be explicit, because reading `T` as the key would be right for a
    1-minute bar and wrong here by one day at the boundary.
    """
    path = _cache_path(coin, "candles-1d", cache)
    payload = None if refresh else _read_cache(path)
    if payload is None:
        payload = post_info({"type": "candleSnapshot",
                             "req": {"coin": coin, "interval": "1d",
                                     "startTime": HISTORY_START_MS,
                                     "endTime": now_ms}}, opener=opener)
        if not isinstance(payload, list):
            payload = []
        _write_cache(path, payload)

    rows = []
    for row in payload:
        try:
            close = float(row["c"])
            volume = float(row["v"])
            rows.append((int(row["t"]), float(row["o"]), float(row["h"]),
                         float(row["l"]), close, volume * close,
                         float(row.get("n") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort()
    # The final candle is the day in progress: its close is simply the current
    # price, so a return computed to it spans an unknown fraction of a day.
    if rows and rows[-1][0] + 86_400_000 > now_ms:
        rows.pop()
    return rows


def fetch_funding(coin: str, *, cache: Path, refresh: bool, now_ms: int,
                  pause: float, opener: Optional[Callable[..., Any]] = None,
                  max_pages: int = 80) -> Dict[int, float]:
    """`{settlement_ms: rate_bps}`, paging FORWARD from the start of history.

    The endpoint takes `startTime` and returns the next 500 records, so the
    cursor moves to one millisecond past the newest stamp seen. Pages are
    cached whole: a stored dict keyed by stamp makes a re-fetch that overlaps
    idempotent, where a list would count settlements twice and every duplicate
    is an hour of carry invented.
    """
    path = _cache_path(coin, "funding", cache)
    rates: Dict[int, float] = {}
    if not refresh:
        cached = _read_cache(path)
        if isinstance(cached, dict) and cached:
            rates = {int(key): float(value) for key, value in cached.items()}

    cursor = max(rates) + 1 if rates else HISTORY_START_MS
    for _ in range(max_pages):
        if cursor >= now_ms:
            break
        payload = None
        for attempt in range(4):
            try:
                payload = post_info({"type": "fundingHistory", "coin": coin,
                                     "startTime": cursor}, opener=opener)
                break
            except Exception:                       # noqa: BLE001
                if attempt == 3:
                    TRUNCATED.add(coin)
                else:
                    time.sleep(2.0 * (attempt + 1))
        if not isinstance(payload, list) or not payload:
            break

        newest = cursor
        for row in payload:
            try:
                ts = int(row["time"])
                rates[ts] = float(row["fundingRate"]) * 10_000.0
                newest = max(newest, ts)
            except (KeyError, TypeError, ValueError):
                continue
        if newest <= cursor:
            break
        cursor = newest + 1
        if len(payload) < FUNDING_PAGE:
            break
        if pause:
            time.sleep(pause)

    _write_cache(path, {str(key): value for key, value in rates.items()})
    return rates


def funding_by_day(rates: Dict[int, float]) -> Dict[str, Tuple[float, int]]:
    """`{date: (bps accrued during that UTC day, settlements)}`.

    Hourly here rather than 8-hourly, so a full day is 24 settlements. The rule
    is the one the shared schema in perpkit/analysis/panel.py fixes: a settlement stamped T paid for the
    interval ending at T, and the stamp is rounded to its nominal minute first
    because Hyperliquid's stamps carry tens of milliseconds of jitter (measured:
    `...600048` and `...200201` on consecutive BTC settlements).
    """
    out: Dict[str, List[float]] = {}
    for ts, bps in rates.items():
        nominal = int(round(ts / 60_000.0)) * 60_000
        bucket = out.setdefault(_day_key(nominal - 1), [0.0, 0.0])
        bucket[0] += bps
        bucket[1] += 1
    return {day: (value[0], int(value[1])) for day, value in out.items()}


PARKINSON = 1.0 / (2.0 * math.sqrt(math.log(2.0)))


def coin_rows(coin: str, candles: Sequence[Tuple[int, float, float, float, float,
                                                 float, float]],
              funding: Dict[str, Tuple[float, int]],
              *, min_settlements: int = 20) -> List[Dict[str, object]]:
    """Panel rows in the shared schema.

    A day with fewer than `min_settlements` of its 24 hourly funding periods is
    marked as having none, rather than reported as a small daily total. A
    partial day looks like cheap funding and is really missing data, and on this
    venue the difference is 24 settlements wide.
    """
    rows: List[Dict[str, object]] = []
    for ts, opn, high, low, close, quote, trades in candles:
        if not (close > 0 and high >= low > 0):
            continue
        date = _day_key(ts)
        bps, periods = funding.get(date, (float("nan"), 0))
        if periods < min_settlements:
            bps, periods = float("nan"), 0

        # `candleSnapshot` returns candles for days BEFORE a coin listed here,
        # carrying an OHLC from somewhere else with `v` and `n` both zero.
        # Measured on a 60-coin panel: ZEC and XMR had 999 such candles each
        # and 13% of the whole panel was one, concentrated in 2023 when the
        # venue was young. They are not thin days, they are days this venue did not trade
        # the coin at all - a return computed across one is a price move that
        # could not have been captured, and a universe filter reading their
        # volume as zero behaves erratically rather than excluding them.
        #
        # `minutes` is the field the shared loader already uses to mark a day
        # incomplete, and setting it to zero routes these through the same guard
        # the shared schema in perpkit/analysis/panel.py uses for an exchange
        # outage: excluded from the universe, and the history run-length counter resets so a coin cannot
        # appear to have years of history it did not have.
        traded = quote > 0 or trades > 0
        rows.append({
            "date": date,
            "symbol": coin,
            "open": opn, "high": high, "low": low, "close": close,
            "quote_volume": quote,
            "trades": trades,
            "taker_buy_frac": float("nan"),
            "minutes": 1440 if traded else 0,
            "rv_bps": math.log(high / low) * PARKINSON * 10_000.0,
            "funding_bps": bps,
            "funding_periods": periods,
        })
    return rows


def build(coins: Sequence[Coin], *, cache: Path, refresh: bool, pause: float,
          now_ms: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, coin in enumerate(coins, start=1):
        try:
            candles = fetch_candles(coin.name, cache=cache, refresh=refresh,
                                    now_ms=now_ms)
            rates = fetch_funding(coin.name, cache=cache, refresh=refresh,
                                  now_ms=now_ms, pause=pause)
        except Exception as exc:                    # noqa: BLE001
            print("\n  " + coin.name + ": skipped (" + str(exc) + ")")
            continue
        rows.extend(coin_rows(coin.name, candles, funding_by_day(rates)))
        print("\r  {}/{}  {:<12} {:,} days, {:,} settlements     ".format(
            index, len(coins), coin.name, len(candles), len(rates)),
            end="", flush=True)
    print()
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    return rows


def report(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        print("No rows.")
        return
    import numpy as np
    dates = sorted({str(row["date"]) for row in rows})
    symbols = sorted({str(row["symbol"]) for row in rows})
    with_funding = [row for row in rows if float(row["funding_periods"]) > 0]
    untraded = [row for row in rows if float(row["minutes"]) == 0]
    breadth: Dict[str, int] = {}
    for row in with_funding:
        breadth[str(row["date"])] = breadth.get(str(row["date"]), 0) + 1
    counts = sorted(breadth.values()) or [0]
    print()
    print("rows                {:,}".format(len(rows)))
    print("coins               {}".format(len(symbols)))
    print("dates               {}  ({} .. {})".format(len(dates), dates[0], dates[-1]))
    print("days with funding   {:,}  ({:.1f}%)".format(
        len(with_funding), 100.0 * len(with_funding) / len(rows)))
    print("pre-listing days     {:,}  ({:.1f}%)  marked minutes=0 and excluded "
          "from the universe".format(
              len(untraded), 100.0 * len(untraded) / len(rows)))
    print("coins per day       min {}  median {}  max {}".format(
        counts[0], counts[len(counts) // 2], counts[-1]))
    if with_funding:
        periods = [float(row["funding_periods"]) for row in with_funding]
        print("settlements per day median {:.0f}  (hourly venue: 24 is a full day)"
              .format(float(np.median(periods))))
    if TRUNCATED:
        print("TRUNCATED by request failures: " + ", ".join(sorted(TRUNCATED)))
        print("  short history for that reason, not because they listed late. "
              "Re-run with a larger --pause.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--min-volume", type=float, default=2e6)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    now_ms = int(time.time() * 1000)
    coins = liquid_coins(top=args.top, min_volume=args.min_volume)
    if not coins:
        raise SystemExit("No coins cleared the volume filter.")
    print("Building Hyperliquid panel for {} coins (>= ${:,.0f}/24h).".format(
        len(coins), args.min_volume))
    print("Funding is hourly and pages 500 at a time, so this is about "
          "{:.0f} minutes at {:.2f}s a call.".format(
              len(coins) * 55 * args.pause / 60.0, args.pause))

    rows = build(coins, cache=args.cache, refresh=args.refresh,
                 pause=args.pause, now_ms=now_ms)
    write_panel(rows, args.out)
    report(rows)
    print("\nwrote " + str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
