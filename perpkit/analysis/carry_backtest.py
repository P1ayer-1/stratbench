"""What would a carry position ACTUALLY have earned? Every entry, over 200 days.

    python -m perpkit.analysis.carry_backtest
    python -m perpkit.analysis.carry_backtest --hold-days 14 --instruments SUI-USDT
    python -m perpkit.analysis.carry_backtest --pages 6 --top 25

`funding_carry.py` answers "does the median funding rate cover the round trip?"
That is a screen, and it is the right first question, but it is not a P&L. It
assumes you enter today at today's spread and that funding behaves like its
median for the whole hold. Neither is a thing that happens.

This runs the position instead. For **every** entry point in the recorded
history it holds for `--hold-days`, collects the funding that actually printed
period by period, applies the basis move that actually happened, pays the
round trip, and reports the distribution of outcomes. A median that survives
that is a different claim from a median rate.

What is historical and what is not
----------------------------------
**Funding: actual.** Every 8h rate as printed, not a median.

**Basis: actual.** Spot and perp 8h closes, so the gap at entry and the gap at
exit are both measured. This matters more than it sounds - `funding_carry`
could only price full convergence as a worst case, whereas the pair's price
P&L is exactly `gap_exit - gap_entry`, and here both ends are known.

**Spreads: today's snapshot, held constant.** There is no historical top-of-
book for these markets, so entry and exit cost is the spread measured now. It
is the one assumption left, and it is the smaller one: the four-leg cost is
tens of bps against funding swings of hundreds.

Overlapping windows are not independent
---------------------------------------
600 periods with a 30-day hold gives ~510 windows, and adjacent ones share 89
of their 90 periods. They are very nearly the same observation. 200 days of
history contains about **7** independent 30-day holds, and the report says so
next to every distribution, because "profitable in 95% of 510 windows" sounds
like evidence and is mostly the same window counted 500 times.

The bar that matters
--------------------
Not the median. A carry is a position you hold through whatever arrives, so
the honest question is what the BAD entries did. The report leads with the 5th
percentile and the worst window, and the verdict is written against those.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


from perpkit.analysis.blofin_spot import (
    fetch_spot_instruments,
    fetch_spot_tickers,
    top_of_book,
)
from perpkit.analysis.funding_carry import (
    DEFAULT_SPOT_MAKER_BPS,
    DEFAULT_SPOT_TAKER_BPS,
    PERIODS_PER_DAY,
    PERP_MAKER_BPS,
    PERP_TAKER_BPS,
    TIER,
    fetch_funding_history,
)
from perpkit.analysis.stats import effective_sample_size

SPOT_CANDLES = "/api/v1/spot/market/candles"
BAR = "8H"                 # matches the funding period exactly
CANDLES_PER_PAGE = 300
PERIOD_SECONDS = 8 * 3600


@dataclass
class Backtest:
    """The distribution of outcomes for one instrument, or why there isn't one."""

    inst_id: str
    unavailable: Optional[str] = None

    windows: int = 0
    effective_n: int = 0
    span_days: float = float("nan")

    median_bps: float = float("nan")
    p5_bps: float = float("nan")
    worst_bps: float = float("nan")
    best_bps: float = float("nan")
    profitable_share: float = float("nan")

    median_funding_bps: float = float("nan")
    median_basis_bps: float = float("nan")
    round_trip_bps: float = float("nan")

    @property
    def usable(self) -> bool:
        return self.unavailable is None and self.windows > 0


def fetch_candles(client, inst_id: str, *, spot: bool, pages: int,
                  api=None) -> Dict[int, float]:
    """{timestamp_ms: close}, paged backwards. 8h bars, matching funding."""
    closes: Dict[int, float] = {}
    cursor: Optional[str] = None

    for _ in range(max(1, pages)):
        params = {"instId": inst_id, "bar": BAR, "limit": str(CANDLES_PER_PAGE)}
        if spot:
            params["instType"] = "SPOT"
        if cursor:
            params["after"] = cursor
        try:
            if spot:
                payload = client.get(SPOT_CANDLES, params=params, sign=False)
            else:
                payload = api.getCandlesticks(
                    inst_id, bar=BAR, limit=str(CANDLES_PER_PAGE), after=cursor)
        except Exception:  # noqa: BLE001 - a short history is still a history
            break

        rows = payload.get("data") or []
        fresh = [row for row in rows if int(row[0]) not in closes]
        if not fresh:
            break
        for row in fresh:
            try:
                closes[int(row[0])] = float(row[4])   # close
            except (TypeError, ValueError, IndexError):
                continue
        if len(rows) < CANDLES_PER_PAGE:
            break
        cursor = str(min(int(row[0]) for row in fresh))

    return closes


def align(funding: Sequence[dict], spot: Dict[int, float],
          perp: Dict[int, float]) -> Tuple[List[int], List[float], List[float]]:
    """(timestamps, funding bps, gap bps) over periods present in all three.

    `gap` is spot minus perp in bps. The pair's price P&L over a hold is
    exactly `gap_exit - gap_entry`: the long spot leg and the short perp leg
    cancel except for the change in their difference, so a persistent
    dislocation costs nothing and only a MOVE in it does.
    """
    rates: Dict[int, float] = {}
    for row in funding:
        try:
            rates[int(row["fundingTime"])] = float(row["fundingRate"]) * 10_000.0
        except (TypeError, ValueError, KeyError):
            continue

    # Funding stamps and candle stamps are both on the 8h grid but need not be
    # the same instant, so snap each funding time to the candle bucket it
    # falls in rather than requiring exact equality.
    bucket = PERIOD_SECONDS * 1000
    spot_by_bucket = {ts // bucket: price for ts, price in spot.items()}
    perp_by_bucket = {ts // bucket: price for ts, price in perp.items()}

    timestamps: List[int] = []
    funding_bps: List[float] = []
    gap_bps: List[float] = []
    for ts in sorted(rates):
        key = ts // bucket
        spot_price = spot_by_bucket.get(key)
        perp_price = perp_by_bucket.get(key)
        if not spot_price or not perp_price or spot_price <= 0:
            continue
        timestamps.append(ts)
        funding_bps.append(rates[ts])
        gap_bps.append((spot_price - perp_price) / spot_price * 10_000.0)

    return timestamps, funding_bps, gap_bps


def run(inst_id: str, funding: Sequence[dict], spot: Dict[int, float],
        perp: Dict[int, float], *, hold_days: float,
        round_trip_bps: float) -> Backtest:
    timestamps, funding_bps, gap_bps = align(funding, spot, perp)
    periods = int(round(hold_days * PERIODS_PER_DAY))

    if len(timestamps) < periods + 2:
        return Backtest(inst_id, unavailable=(
            f"only {len(timestamps)} aligned periods, need {periods + 2} for a "
            f"{hold_days:g}-day hold"))

    nets: List[float] = []
    fundings: List[float] = []
    bases: List[float] = []
    for entry in range(len(timestamps) - periods):
        exit_index = entry + periods
        # Funding accrues on the periods held THROUGH, not the entry instant.
        collected = sum(funding_bps[entry + 1:exit_index + 1])
        basis = gap_bps[exit_index] - gap_bps[entry]
        fundings.append(collected)
        bases.append(basis)
        nets.append(collected + basis - round_trip_bps)

    span_days = (timestamps[-1] - timestamps[0]) / 86_400_000.0
    ordered = sorted(nets)
    return Backtest(
        inst_id=inst_id,
        windows=len(nets),
        # Adjacent windows share all but one period, so the row count is not
        # the sample size. This is the same correction check_features applies
        # to overlapping labels, for the same reason.
        effective_n=effective_sample_size(
            len(nets), hold_days * 86_400.0, PERIOD_SECONDS),
        span_days=span_days,
        median_bps=statistics.median(nets),
        p5_bps=ordered[max(0, int(0.05 * (len(ordered) - 1)))],
        worst_bps=ordered[0],
        best_bps=ordered[-1],
        profitable_share=sum(1 for value in nets if value > 0) / len(nets),
        median_funding_bps=statistics.median(fundings),
        median_basis_bps=statistics.median(bases),
        round_trip_bps=round_trip_bps,
    )


def report(rows: Sequence[Backtest], *, hold_days: float, top: int,
           spot_taker: float) -> None:
    print("\n" + "=" * 100)
    print(f"CARRY BACKTEST  (long spot + short perp, VIP {TIER}, "
          f"{hold_days:g}-day hold, every entry)")
    print("=" * 100)
    print(f"  fees   perp {PERP_TAKER_BPS:.2f} / spot {spot_taker:.2f} bps "
          f"taker, four legs")
    print("  Funding and basis are HISTORICAL. Spreads are today's, held "
          "constant - there is no\n  historical top of book for these markets.")

    usable = [row for row in rows if row.usable]
    missing = [row for row in rows if not row.usable]
    if not usable:
        print("\n  Nothing measurable.")
        for row in missing[:10]:
            print(f"    {row.inst_id:<16} {row.unavailable}")
        return

    ranked = sorted(usable, key=lambda row: -row.p5_bps)
    shown = ranked[:top]

    print(f"\n  {'instrument':<15}{'median':>9}{'p5':>9}{'worst':>9}{'best':>9}"
          f"{'profit%':>9}{'fund':>9}{'basis':>9}{'cost':>8}{'windows':>9}{'eff N':>7}")
    print("  " + "-" * 96)
    for row in shown:
        print(f"  {row.inst_id:<15}{row.median_bps:>+9.1f}{row.p5_bps:>+9.1f}"
              f"{row.worst_bps:>+9.1f}{row.best_bps:>+9.1f}"
              f"{row.profitable_share:>9.0%}{row.median_funding_bps:>+9.1f}"
              f"{row.median_basis_bps:>+9.1f}{row.round_trip_bps:>8.1f}"
              f"{row.windows:>9,}{row.effective_n:>7}")
    if len(ranked) > len(shown):
        print(f"  ... {len(ranked) - len(shown)} more")

    print(f"\n  Sorted by p5, not median. `eff N` is the count of INDEPENDENT "
          f"holds in the\n  history - adjacent windows share all but one of "
          f"their {int(hold_days * PERIODS_PER_DAY)} periods, so "
          f"`windows` is\n  the same observation counted many times and "
          f"`profit%` inherits that.")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)

    survives_p5 = [row for row in ranked if row.p5_bps > 0]
    survives_worst = [row for row in ranked if row.worst_bps > 0]
    median_only = [row for row in ranked if row.median_bps > 0 and row.p5_bps <= 0]

    if survives_worst:
        names = ", ".join(row.inst_id for row in survives_worst[:8])
        print(f"  {len(survives_worst)} instrument(s) were profitable from "
              f"EVERY entry in the history:\n    {names}")
        print("  That is the strongest statement this data can make, and it is "
              "still only\n  about the regime the history covers.")
    elif survives_p5:
        names = ", ".join(row.inst_id for row in survives_p5[:8])
        print(f"  {len(survives_p5)} instrument(s) were profitable at the 5th "
              f"percentile of entries:\n    {names}")
        print("  The worst entries lost money; the bad tail is real and priced "
              "above.")
    else:
        print("  NOTHING IS PROFITABLE AT THE 5TH PERCENTILE OF ENTRY POINTS.")
        print("  The screen looked at median funding and this looks at what "
              "actually happened\n  from every entry, and they disagree. Trust "
              "this one: a carry is a position\n  you hold through whatever "
              "arrives, not through the median.")

    if median_only:
        names = ", ".join(row.inst_id for row in median_only[:8])
        print(f"\n  Profitable on median but NOT at the 5th percentile: {names}")
        print("  These are the ones the screen would have sold you.")

    worst = min(ranked, key=lambda row: row.worst_bps)
    print(f"\n  Worst single hold in the sample: {worst.inst_id} at "
          f"{worst.worst_bps:+.1f} bps.")

    thin = [row for row in ranked[:top] if row.effective_n < 10]
    if thin:
        print(f"\n  {len(thin)} of the instruments shown have fewer than 10 "
              "independent holds in\n  the history. Every distribution above "
              "is drawn from that, not from the\n  window count beside it.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instruments", default=None,
                        help="Default: every pair with both markets.")
    parser.add_argument("--hold-days", type=float, default=30.0)
    parser.add_argument("--pages", type=int, default=6,
                        help="Pages of history, 100 funding periods each.")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-volume-usd", type=float, default=1e6)
    parser.add_argument("--spot-taker-bps", type=float,
                        default=DEFAULT_SPOT_TAKER_BPS)
    args = parser.parse_args(argv)

    if args.spot_taker_bps is None:
        raise SystemExit(
            f"No spot fee schedule for VIP {TIER}. Pass --spot-taker-bps, or "
            "add the tier to perpkit.fees.SPOT_VIP_TIERS.")

    from blofin.client import Client
    from blofin.rest_market import MarketAPI

    client = Client()
    api = MarketAPI(client)

    print("Fetching markets...")
    spot_instruments = fetch_spot_instruments(client)
    spot_tickers = fetch_spot_tickers(client)
    perps = {row["instId"] for row in (api.getInstruments().get("data") or [])
             if row.get("contractType") == "linear"}
    perp_tickers = {row["instId"]: row
                    for row in (api.getTickers().get("data") or [])}

    both = sorted(set(spot_instruments) & perps)
    wanted = ([part.strip() for part in args.instruments.split(",") if part.strip()]
              if args.instruments else both)
    print(f"  {len(both)} pairs with both markets; testing {len(wanted)}.")

    rows: List[Backtest] = []
    for index, inst_id in enumerate(wanted, start=1):
        spot_book = top_of_book(spot_tickers.get(inst_id))
        perp_book = top_of_book(perp_tickers.get(inst_id))
        if not spot_book or not perp_book:
            rows.append(Backtest(inst_id, unavailable="no usable quote"))
            continue
        if args.min_volume_usd > 0:
            try:
                volume = float(
                    spot_tickers[inst_id].get("volCurrency24h") or 0) * spot_book[2]
            except (TypeError, ValueError):
                volume = 0.0
            if volume < args.min_volume_usd:
                rows.append(Backtest(inst_id, unavailable="spot volume below floor"))
                continue

        funding = fetch_funding_history(api, inst_id, pages=args.pages)
        if not funding:
            rows.append(Backtest(inst_id, unavailable="no funding history"))
            continue
        candle_pages = max(1, args.pages // 3 + 1)
        spot_closes = fetch_candles(client, inst_id, spot=True, pages=candle_pages)
        perp_closes = fetch_candles(client, inst_id, spot=False,
                                    pages=candle_pages, api=api)

        round_trip = (spot_book[3] + perp_book[3]
                      + 2 * (args.spot_taker_bps + PERP_TAKER_BPS))
        rows.append(run(inst_id, funding, spot_closes, perp_closes,
                        hold_days=args.hold_days, round_trip_bps=round_trip))
        if index % 10 == 0:
            print(f"  {index}/{len(wanted)}...", flush=True)

    report(rows, hold_days=args.hold_days, top=args.top,
           spot_taker=args.spot_taker_bps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
