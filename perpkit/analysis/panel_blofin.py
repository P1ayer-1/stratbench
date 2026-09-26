"""The shared daily panel schema, built from BloFin.

    python -m perpkit.analysis.panel_blofin --top 200
    python -m perpkit.analysis.panel_blofin --top 200 --refresh
    python -m perpkit.analysis.factor_panel --panel data\\panel\\blofin-daily.csv

Why a panel per venue
---------------------
**Is an effect a property of perpetual markets or of one venue?** A
replication on a venue with a different user base, a different fee schedule
and a different funding formula is worth more than any further slicing of the
original sample, because the failure it tests for - that the effect is an
artifact of how one exchange sets its rates - is invisible from inside that
exchange.

**What does it cost on this venue?** BloFin's spreads are systematically wider
than Binance's where makers have room to choose the width (measured: 2.7x on
ADA, LTC, AVAX and DOGE, 1.0x on the tick-bound BTC), so a strategy priced at
Binance's costs is priced wrong here. The panel carries BloFin's own quoted
spread per instrument so the harness can be run at a cost measured on the
venue rather than assumed from another.

What BloFin gives, and what it does not
---------------------------------------
`getFundingRate()` returns every instrument's current rate AND its funding
interval in a single call - the interval matters, because BloFin runs some
instruments on 8h and others on 4h or 1h, and a rate is not comparable across
intervals until it is expressed per day. `getCandlesticks(bar="1D")` returns up
to 1,340 daily bars, which is 3.7 years. `getFundingRateHistory` pages 100
settlements at a time, oldest-last.

Columns in the shared schema (perpkit/analysis/panel.py) that BloFin cannot
fill the usual way:

- `trades` and `taker_buy_frac` are absent from BloFin's candle payload, so
  they are NaN and the two factors that use them simply do not score here.
- `rv_bps` is the **Parkinson** high/low estimator rather than a sum of squared
  1-minute returns, because fetching 3.7 years of 1m bars for 200 instruments
  over REST is not a reasonable thing to do. It is a different estimator of the
  same quantity and is marked as such; the vol-scaling a carry backtest
  uses the standard deviation of daily returns, which is computed
  from closes and is identical across panels.

The unconfirmed last candle is dropped. BloFin returns the day in progress with
`confirm = "0"`, and a partial day whose close is simply "now" would give the
newest row a return over an unknown fraction of a day - in the one place, the
end of the sample, where it would be least likely to be noticed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


from perpkit.analysis.panel import PANEL_COLUMNS, write_panel

from perpkit.config import DATA_DIR
CACHE = DATA_DIR / "cache" / "blofin"
DEFAULT_OUT = DATA_DIR / "panel" / "blofin-daily.csv"

CANDLE_LIMIT = 1440          # the endpoint caps out around 1,340
# Instruments whose funding history a rate limit cut short. Reported at the
# end rather than raised, because one truncated symbol is not a reason to
# throw away the other 199 - but an unreported one is a silent short history.
TRUNCATED: set = set()
FUNDING_PAGE = 100
HOURS_PER_DAY = 24.0


def market_api():
    from blofin.client import Client
    from blofin.rest_market import MarketAPI
    return MarketAPI(Client())


def _day_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


@dataclass
class Instrument:
    inst_id: str
    volume_usd: float
    spread_bps: float
    funding_interval_hours: float


def liquid_instruments(api, *, top: int, min_volume: float) -> List[Instrument]:
    """The `top` USDT perps by 24h quote volume, with spread and funding cadence.

    The usual survivorship caveat applies and is worth stating: ranking by volume TODAY selects the contracts
    that survived and grew, so the early years of the panel are populated by
    the winners of the later ones. Carry is the least exposed factor to that,
    because it ranks on a cash flow rather than on past price, but "least
    exposed" is not "unexposed".
    """
    tickers = (api.getTickers().get("data") or [])
    rates = {row["instId"]: row
             for row in (api.getFundingRate().get("data") or [])}

    out: List[Instrument] = []
    for row in tickers:
        inst_id = str(row.get("instId", ""))
        if not inst_id.endswith("-USDT"):
            continue
        try:
            # `getTickers` reports size in contracts (`vol24h`) and in base
            # units (`volCurrency24h`) but never in quote, so dollar volume is
            # derived. Using `vol24h` here would rank a 1000x-multiplier meme
            # contract against BTC on contract count, which is not a size.
            volume = float(row.get("volCurrency24h") or 0.0) * float(
                row.get("last") or 0.0)
            bid, ask = float(row.get("bidPrice") or 0), float(row.get("askPrice") or 0)
        except (TypeError, ValueError):
            continue
        if volume < min_volume or bid <= 0 or ask <= bid:
            continue
        mid = (bid + ask) / 2.0
        rate = rates.get(inst_id, {})
        try:
            interval = float(rate.get("fundingInterval") or 8.0)
        except (TypeError, ValueError):
            interval = 8.0
        if str(rate.get("fundingIntervalUnit", "hour")).lower().startswith("min"):
            interval /= 60.0
        out.append(Instrument(inst_id, volume, (ask - bid) / mid * 10_000.0,
                              interval))

    out.sort(key=lambda item: item.volume_usd, reverse=True)
    return out[:top]


# ---------------------------------------------------------------------------
# Candles and funding, cached per instrument
# ---------------------------------------------------------------------------


def _cache_path(inst_id: str, kind: str, cache: Path) -> Path:
    return cache / inst_id / (kind + ".json")


def _read_cache(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _write_cache(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def fetch_candles(api, inst_id: str, *, cache: Path, refresh: bool,
                  ) -> List[Tuple[int, float, float, float, float, float]]:
    """Daily bars as `(ts, open, high, low, close, quote_volume)`, oldest first.

    The unconfirmed bar is dropped: BloFin returns the day in progress with
    `confirm = "0"`, and its close is whatever the price is at the moment of
    the request.
    """
    path = _cache_path(inst_id, "candles-1d", cache)
    payload = None if refresh else _read_cache(path)
    if payload is None:
        payload = (api.getCandlesticks(inst_id, bar="1D",
                                       limit=str(CANDLE_LIMIT)).get("data") or [])
        _write_cache(path, payload)

    rows: List[Tuple[int, float, float, float, float, float]] = []
    for row in payload:
        try:
            if str(row[8]) == "0":                 # the day in progress
                continue
            rows.append((int(row[0]), float(row[1]), float(row[2]),
                         float(row[3]), float(row[4]), float(row[7])))
        except (IndexError, TypeError, ValueError):
            continue
    rows.sort()
    return rows


def fetch_funding_history(api, inst_id: str, *, cache: Path, refresh: bool,
                          pages: int) -> Dict[int, float]:
    """`{fundingTime_ms: rate_bps}`, paging backwards from now.

    The endpoint returns newest first and takes a `before`/`after` cursor; this
    walks backwards with `after` set to the oldest stamp seen, which is the
    same pagination `funding_carry.py` uses and for the same reason - asking
    for the newest page again loops forever collecting duplicates.
    """
    path = _cache_path(inst_id, "funding", cache)
    cached = None if refresh else _read_cache(path)
    if isinstance(cached, dict) and cached:
        return {int(key): float(value) for key, value in cached.items()}

    rates: Dict[int, float] = {}
    cursor: Optional[str] = None
    for _ in range(pages):
        # A rate limit that merely ended the loop would truncate this
        # instrument's history silently, and a short history is not visibly
        # different from an instrument that was listed late. Retry, then record
        # the truncation rather than swallowing it.
        payload = None
        for attempt in range(4):
            try:
                payload = api.getFundingRateHistory(inst_id, after=cursor,
                                                    limit=str(FUNDING_PAGE))
                break
            except Exception:                      # noqa: BLE001
                if attempt == 3:
                    TRUNCATED.add(inst_id)
                    break
                time.sleep(1.0 * (attempt + 1))
        if payload is None:
            break
        page = payload.get("data") or []
        fresh = {}
        for row in page:
            try:
                ts = int(row["fundingTime"])
                if ts in rates:
                    continue
                fresh[ts] = float(row["fundingRate"]) * 10_000.0
            except (KeyError, TypeError, ValueError):
                continue
        if not fresh:
            break
        rates.update(fresh)
        if len(page) < FUNDING_PAGE:
            break
        cursor = str(min(fresh))

    _write_cache(path, {str(key): value for key, value in rates.items()})
    return rates


def funding_by_day(rates: Dict[int, float]) -> Dict[str, Tuple[float, int]]:
    """`{date: (bps accrued during that UTC day, periods)}`.

    The rule the shared schema in perpkit/analysis/panel.py fixes: a settlement stamped T paid for the
    interval ending at T, so it is bucketed just before T, with the stamp
    rounded to its nominal minute first to absorb the milliseconds by which
    settlements print late.
    """
    out: Dict[str, List[float]] = {}
    for ts, bps in rates.items():
        nominal = int(round(ts / 60_000.0)) * 60_000
        bucket = out.setdefault(_day_key(nominal - 1), [0.0, 0.0])
        bucket[0] += bps
        bucket[1] += 1
    return {day: (value[0], int(value[1])) for day, value in out.items()}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


PARKINSON = 1.0 / (2.0 * math.sqrt(math.log(2.0)))


def instrument_rows(inst_id: str,
                    candles: Sequence[Tuple[int, float, float, float, float, float]],
                    funding: Dict[str, Tuple[float, int]]) -> List[Dict[str, object]]:
    """Daily panel rows in the shared schema in perpkit/analysis/panel.py, so
    one harness reads every venue's panel."""
    rows: List[Dict[str, object]] = []
    for ts, opn, high, low, close, quote in candles:
        if not (close > 0 and high >= low > 0):
            continue
        date = _day_key(ts)
        bps, periods = funding.get(date, (float("nan"), 0))
        rows.append({
            "date": date,
            "symbol": inst_id.replace("-", ""),
            "open": opn, "high": high, "low": low, "close": close,
            "quote_volume": quote,
            "trades": float("nan"),
            "taker_buy_frac": float("nan"),
            "minutes": 1440,
            "rv_bps": math.log(high / low) * PARKINSON * 10_000.0,
            "funding_bps": bps,
            "funding_periods": periods,
        })
    return rows


def build(api, instruments: Sequence[Instrument], *, cache: Path, refresh: bool,
          pages: int, pause: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        inst_id = instrument.inst_id
        try:
            candles = fetch_candles(api, inst_id, cache=cache, refresh=refresh)
            rates = fetch_funding_history(api, inst_id, cache=cache,
                                          refresh=refresh, pages=pages)
        except Exception as exc:                   # noqa: BLE001
            print("\n  " + inst_id + ": skipped (" + str(exc) + ")")
            continue
        rows.extend(instrument_rows(inst_id, candles, funding_by_day(rates)))
        print("\r  {}/{}  {:<18} {:,} days, {:,} settlements    ".format(
            index, total, inst_id, len(candles), len(rates)), end="", flush=True)
        if pause:
            time.sleep(pause)
    print()
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    return rows


def write_spreads(instruments: Sequence[Instrument], path: Path) -> None:
    """The venue's own quoted spread per instrument, beside the panel.

    This is what makes a BloFin-priced backtest possible rather than a
    Binance-priced one run on BloFin data, and it is a snapshot: one reading of
    the top of book on the day the panel was built, which is the same
    assumption `carry_backtest.py` makes and should be re-read before it is
    trusted at size.
    """
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "inst_id", "volume_usd_24h", "spread_bps",
                         "funding_interval_hours"])
        for item in instruments:
            writer.writerow([item.inst_id.replace("-", ""), item.inst_id,
                             "{:.0f}".format(item.volume_usd),
                             "{:.3f}".format(item.spread_bps),
                             "{:g}".format(item.funding_interval_hours)])


def report(rows: Sequence[Dict[str, object]],
           instruments: Sequence[Instrument]) -> None:
    if not rows:
        print("No rows.")
        return
    dates = sorted({str(row["date"]) for row in rows})
    symbols = sorted({str(row["symbol"]) for row in rows})
    with_funding = sum(1 for row in rows if float(row["funding_periods"]) > 0)
    spreads = sorted(item.spread_bps for item in instruments)
    intervals: Dict[float, int] = {}
    for item in instruments:
        intervals[item.funding_interval_hours] = (
            intervals.get(item.funding_interval_hours, 0) + 1)

    breadth: Dict[str, int] = {}
    for row in rows:
        breadth[str(row["date"])] = breadth.get(str(row["date"]), 0) + 1
    counts = sorted(breadth.values())

    print()
    print("rows                {:,}".format(len(rows)))
    print("instruments         {}".format(len(symbols)))
    print("dates               {}  ({} .. {})".format(len(dates), dates[0], dates[-1]))
    print("days with funding   {:,}  ({:.1f}%)".format(
        with_funding, 100.0 * with_funding / len(rows)))
    print("symbols per day     min {}  median {}  max {}".format(
        counts[0], counts[len(counts) // 2], counts[-1]))
    print("quoted spread bps   p25 {:.2f}  median {:.2f}  p75 {:.2f}".format(
        spreads[len(spreads) // 4], spreads[len(spreads) // 2],
        spreads[3 * len(spreads) // 4]))
    print("funding intervals   " + ", ".join(
        "{:g}h x{}".format(hours, count) for hours, count in sorted(intervals.items())))
    if TRUNCATED:
        print("TRUNCATED by rate limits: " + ", ".join(sorted(TRUNCATED)))
        print("  their funding history is short for that reason, not because "
              "they were listed late. Re-run with a larger --pause.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-volume", type=float, default=1e6)
    parser.add_argument("--pages", type=int, default=14,
                        help="funding pages per instrument; 100 settlements each")
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    api = market_api()
    instruments = liquid_instruments(api, top=args.top, min_volume=args.min_volume)
    if not instruments:
        raise SystemExit("No instruments cleared the volume filter.")
    print("Building BloFin panel for {} instruments (>= ${:,.0f}/24h)...".format(
        len(instruments), args.min_volume))

    rows = build(api, instruments, cache=args.cache, refresh=args.refresh,
                 pages=args.pages, pause=args.pause)
    write_panel(rows, args.out)
    write_spreads(instruments, args.out.with_name("blofin-spreads.csv"))
    report(rows, instruments)
    print("\nwrote " + str(args.out))
    print("wrote " + str(args.out.with_name("blofin-spreads.csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
