"""Does perp funding pay for the cost of hedging it? The delta-neutral carry gate.

    python -m perpkit.analysis.funding_carry
    python -m perpkit.analysis.funding_carry --hold-days 30 --top 25
    python -m perpkit.analysis.funding_carry --instruments ADA-USDT,SUI-USDT

Directional strategies on short horizons usually die on execution cost
against a forecast that was too small. Funding carry is attractive because it
needs no forecast at all: hold spot, short the perp against it, and collect
funding while the position is delta-neutral. The return is a transfer from
longs to shorts, not a prediction. That does not make it an edge: this tool
exists to show how much of the funding the costs eat, and it often eats all
of it.

The structure, and the one asymmetry that shapes it
---------------------------------------------------
    LONG spot  +  SHORT perp  ->  receives funding when funding is POSITIVE

You can only be long the spot leg: BloFin's spot endpoints offer no borrow,
so shorting it is not available. That makes NEGATIVE funding unharvestable in
this structure, including on instruments whose median funding is below zero.

What actually decides it
------------------------
Not the funding rate. **The spot spread.** A carry position crosses four legs
over its life (in and out, on both books) and the spot side of a venue can be
far wider than the perp side: tens of bps of spot spread against single-digit
perp spreads on mid-cap coins is common. Against funding of a few bps per day,
a round trip like that is days or weeks of carry before the position is even
flat.

So the gate is a break-even in days:

    round trip cost bps / funding bps per day  =  days before this earns

and an instrument only makes sense if you would actually hold it that long,
through funding that is not guaranteed to stay positive for it.

What this does NOT model
------------------------
**Basis convergence.** You enter at some spot-perp basis and exit at another,
and the difference is P&L this tool does not forecast. It reports the current
basis so the size of that exposure is visible, but a carry entered at a rich
basis and exited at a cheap one can lose more than the funding it collected.

**Funding persistence.** The history is what it is: one regime, a few months.
The report shows the fraction of periods that were positive and the worst
cumulative drawdown precisely because the median is the number most likely to
mislead here.

**Liquidation.** The short perp leg needs margin, and a violent rally moves
against it. Delta-neutral is not risk-neutral when the legs margin separately.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


from perpkit.analysis.blofin_spot import (
    fetch_spot_instruments,
    fetch_spot_tickers,
    top_of_book,
)

try:  # Single source of truth for fees: perpkit/fees.py.
    from perpkit.config import (
        MAKER_FEE_BPS,
        SPOT_MAKER_FEE_BPS,
        SPOT_TAKER_FEE_BPS,
        TAKER_FEE_BPS,
        VIP_TIER,
    )

    PERP_MAKER_BPS = float(MAKER_FEE_BPS)
    PERP_TAKER_BPS = float(TAKER_FEE_BPS)
    TIER = VIP_TIER
    # None on a tier whose spot rates were never confirmed. Left as None so
    # the caller has to supply them rather than inherit a guess.
    DEFAULT_SPOT_MAKER_BPS = (float(SPOT_MAKER_FEE_BPS)
                              if SPOT_MAKER_FEE_BPS is not None else None)
    DEFAULT_SPOT_TAKER_BPS = (float(SPOT_TAKER_FEE_BPS)
                              if SPOT_TAKER_FEE_BPS is not None else None)
except Exception:  # pragma: no cover - keeps the analysis tools standalone
    PERP_MAKER_BPS, PERP_TAKER_BPS, TIER = 0.6, 5.0, 1
    DEFAULT_SPOT_MAKER_BPS, DEFAULT_SPOT_TAKER_BPS = None, None

# Funding is paid every 8 hours on BloFin, so three periods a day.
PERIODS_PER_DAY = 3

# Hold this long by default when annualising. A month is long enough for the
# round-trip cost to amortise and short enough to be a decision someone would
# actually make.
DEFAULT_HOLD_DAYS = 30.0


@dataclass
class Carry:
    """One instrument's carry arithmetic, or why there isn't any."""

    inst_id: str
    unavailable: Optional[str] = None

    funding_median_bps: float = float("nan")   # per 8h period
    funding_mean_bps: float = float("nan")
    positive_share: float = float("nan")
    worst_period_bps: float = float("nan")
    max_drawdown_bps: float = float("nan")     # of cumulative funding
    periods: int = 0

    spot_spread_bps: float = float("nan")
    perp_spread_bps: float = float("nan")
    basis_bps: float = float("nan")

    @property
    def funding_per_day_bps(self) -> float:
        return self.funding_median_bps * PERIODS_PER_DAY

    def round_trip_bps(self, *, spot_fee: float, perp_fee: float,
                       cross: bool) -> float:
        """Cost of opening and closing the pair.

        Crossing pays the whole spread on each leg, once in and once out.
        Quoting pays no spread but takes fill risk this tool cannot price, so
        both are reported and neither is presented as the answer.
        """
        spreads = (self.spot_spread_bps + self.perp_spread_bps) if cross else 0.0
        return spreads + 2 * (spot_fee + perp_fee)

    def breakeven_days(self, **kwargs) -> float:
        daily = self.funding_per_day_bps
        if not (daily > 0):
            return float("inf")
        return self.round_trip_bps(**kwargs) / daily

    def net_bps_over(self, days: float, **kwargs) -> float:
        return self.funding_per_day_bps * days - self.round_trip_bps(**kwargs)

    @property
    def convergence_cost_bps(self) -> float:
        """What full basis convergence would cost this position.

        The pair's price P&L is `gap_exit - gap_entry`, where `gap` is spot
        minus perp: the basis cancels exactly if it is unchanged, so a
        persistent dislocation is not itself a cost. Convergence is.

        In the BloFin samples checked when this was written the perp traded
        BELOW spot on every instrument, the unfavourable direction here - the hedge buys the
        expensive leg and shorts the cheap one, so the gap closing takes money
        out. A widening gap would pay, and this deliberately does not count
        that: an uncontrolled exposure is a risk, not a source of return.
        """
        return max(0.0, -self.basis_bps)

    def net_after_convergence(self, days: float, **kwargs) -> float:
        return self.net_bps_over(days, **kwargs) - self.convergence_cost_bps

    @property
    def harvestable(self) -> bool:
        """Positive funding only. The spot leg cannot be shorted."""
        return self.funding_median_bps > 0


def fetch_funding_history(api, inst_id: str, *, pages: int = 1,
                          per_page: int = 100) -> List[dict]:
    """Funding history, paged backwards through time.

    The endpoint caps at 100 records — 33 days at three periods a day, which
    is one regime and the binding unknown behind every carry number here. Its
    `after` parameter returns records OLDER than a timestamp, so walking it
    backwards buys months instead.

    Stops early when a page comes back short or repeats a timestamp already
    seen, because an endpoint that ignores an out-of-range cursor by returning
    the newest page again would otherwise loop forever collecting duplicates.
    """
    collected: List[dict] = []
    seen = set()
    cursor: Optional[str] = None

    for _ in range(max(1, pages)):
        try:
            payload = api.getFundingRateHistory(
                inst_id, after=cursor, limit=str(per_page))
        except Exception:  # noqa: BLE001 - a short history is still a history
            break
        page = payload.get("data") or []
        fresh = [row for row in page
                 if row.get("fundingTime") and row["fundingTime"] not in seen]
        if not fresh:
            break
        seen.update(row["fundingTime"] for row in fresh)
        collected.extend(fresh)
        if len(page) < per_page:
            break
        cursor = min(row["fundingTime"] for row in fresh)

    return collected


def funding_stats(history: Sequence[dict]) -> Dict[str, float]:
    """Median, mean, positive share, worst period, worst cumulative drawdown.

    The drawdown is over the CUMULATIVE funding curve, which is the thing a
    carry position actually experiences: a run of negative periods is money
    paid out, and it is the reason the median alone is not a plan.
    """
    rates = []
    for row in history:
        value = row.get("fundingRate")
        if value in (None, ""):
            continue
        try:
            rates.append(float(value) * 10_000.0)
        except (TypeError, ValueError):
            continue
    if not rates:
        return {}

    # The endpoint returns newest first; carry accrues oldest to newest.
    rates = list(reversed(rates))
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for rate in rates:
        cumulative += rate
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)

    return {
        "median": statistics.median(rates),
        "mean": statistics.fmean(rates),
        "positive_share": sum(1 for rate in rates if rate > 0) / len(rates),
        "worst": min(rates),
        "drawdown": drawdown,
        "periods": len(rates),
    }


def build(inst_id: str, history: Sequence[dict], spot_ticker, perp_ticker) -> Carry:
    spot = top_of_book(spot_ticker)
    perp = top_of_book(perp_ticker)
    if spot is None:
        return Carry(inst_id, unavailable="no usable spot quote")
    if perp is None:
        return Carry(inst_id, unavailable="no usable perp quote")

    stats = funding_stats(history)
    if not stats:
        return Carry(inst_id, unavailable="no funding history")

    _, _, spot_mid, spot_spread = spot
    _, _, perp_mid, perp_spread = perp
    return Carry(
        inst_id=inst_id,
        funding_median_bps=stats["median"],
        funding_mean_bps=stats["mean"],
        positive_share=stats["positive_share"],
        worst_period_bps=stats["worst"],
        max_drawdown_bps=stats["drawdown"],
        periods=int(stats["periods"]),
        spot_spread_bps=spot_spread,
        perp_spread_bps=perp_spread,
        basis_bps=(perp_mid - spot_mid) / spot_mid * 10_000.0,
    )


def report(rows: Sequence[Carry], *, spot_maker: float, spot_taker: float,
           hold_days: float, top: int, spot_fees_confirmed: bool) -> None:
    print("\n" + "=" * 94)
    print(f"FUNDING CARRY  (long spot + short perp, VIP {TIER}, "
          f"{hold_days:g}-day hold)")
    print("=" * 94)
    print(f"  perp fees   maker {PERP_MAKER_BPS:.2f} / taker {PERP_TAKER_BPS:.2f} bps")
    print(f"  spot fees   maker {spot_maker:.2f} / taker {spot_taker:.2f} bps", end="")
    if not spot_fees_confirmed:
        print("   <- NOT CONFIRMED, defaulted to the futures schedule")
    else:
        print()
    print("  Only positive funding is harvestable: the spot leg cannot be "
          "shorted, so an\n  instrument that pays longs is not carryable here "
          "at all.\n")

    usable = [row for row in rows if row.unavailable is None]
    missing = [row for row in rows if row.unavailable is not None]
    if not usable:
        print("  Nothing measurable.")
        for row in missing:
            print(f"    {row.inst_id:<16} {row.unavailable}")
        return

    cross = {"spot_fee": spot_taker, "perp_fee": PERP_TAKER_BPS, "cross": True}
    quote = {"spot_fee": spot_maker, "perp_fee": PERP_MAKER_BPS, "cross": False}

    ranked = sorted(usable, key=lambda row: -row.net_bps_over(hold_days, **cross))
    shown = ranked[:top]

    label = f"net {int(hold_days)}d"
    print(f"  {'instrument':<15}{'fund/day':>10}{'pos%':>7}{'spot spr':>10}"
          f"{'perp spr':>10}{'cross RT':>10}{'b/e days':>10}"
          f"{label:>10}{'basis':>8}{'conv':>7}{'net-conv':>10}")
    print("  " + "-" * 107)
    for row in shown:
        breakeven = row.breakeven_days(**cross)
        breakeven_text = "never" if breakeven == float("inf") else f"{breakeven:.1f}"
        print(f"  {row.inst_id:<15}{row.funding_per_day_bps:>10.2f}"
              f"{row.positive_share:>7.0%}{row.spot_spread_bps:>10.2f}"
              f"{row.perp_spread_bps:>10.2f}"
              f"{row.round_trip_bps(**cross):>10.2f}{breakeven_text:>10}"
              f"{row.net_bps_over(hold_days, **cross):>+10.1f}"
              f"{row.basis_bps:>8.2f}{row.convergence_cost_bps:>7.2f}"
              f"{row.net_after_convergence(hold_days, **cross):>+10.1f}")
    if len(ranked) > len(shown):
        print(f"  ... {len(ranked) - len(shown)} more")

    if missing:
        print("\n  not measured:")
        for row in missing[:8]:
            print(f"    {row.inst_id:<16} {row.unavailable}")

    print("\n" + "=" * 94)
    print("VERDICT")
    print("=" * 94)

    unharvestable = [row for row in usable if not row.harvestable]
    profitable_cross = [row for row in usable
                        if row.harvestable and row.net_bps_over(hold_days, **cross) > 0]
    profitable_quote = [row for row in usable
                        if row.harvestable and row.net_bps_over(hold_days, **quote) > 0]

    if unharvestable:
        names = ", ".join(row.inst_id for row in unharvestable[:6])
        print(f"  {len(unharvestable)} instrument(s) pay longs on median, so "
              f"they cannot be carried\n  in this structure: {names}")

    if not profitable_cross and not profitable_quote:
        print("\n  NOTHING CLEARS ITS OWN COST, EVEN QUOTING BOTH LEGS.")
        print("  The spot spread is what does the damage. Funding on this venue "
              "is a few bps\n  per day and the spot book is tens of bps wide, "
              "so the round trip is weeks of\n  carry before the position is "
              "flat.")
        return

    if profitable_cross:
        best = max(profitable_cross,
                   key=lambda row: row.net_bps_over(hold_days, **cross))
        print(f"\n  {len(profitable_cross)} instrument(s) clear their cost "
              f"CROSSING every leg over {hold_days:g} days.")
        print(f"  Best: {best.inst_id} at "
              f"{best.net_bps_over(hold_days, **cross):+.1f} bps "
              f"({best.breakeven_days(**cross):.1f} days to break even).")
    elif profitable_quote:
        print(f"\n  Nothing clears its cost crossing, but "
              f"{len(profitable_quote)} instrument(s) do if BOTH legs are\n  "
              f"quoted passively over {hold_days:g} days.")
        print("  That is a fill-rate assumption, not a result — and quoting the "
              "spot leg of a\n  hedge means sitting delta-exposed until it "
              "fills. Measure a real fill rate before\n  believing it: that assumption is the "
              "expensive one to get wrong.")

    risky = [row for row in (profitable_cross or profitable_quote)
             if row.positive_share < 0.8 or row.max_drawdown_bps < -20]
    if risky:
        print("\n  Read the distribution before believing any of it:")
        for row in risky[:5]:
            print(f"    {row.inst_id:<15} positive {row.positive_share:.0%} of "
                  f"{row.periods} periods, worst period "
                  f"{row.worst_period_bps:+.2f} bps, worst cumulative "
                  f"drawdown {row.max_drawdown_bps:+.1f} bps")

    print("\n  `conv` is what FULL basis convergence would cost. The perp "
          "trades BELOW spot on\n  every instrument measured here, so the "
          "hedge buys the expensive leg and shorts\n  the cheap one, and the "
          "gap closing takes money out. `net-conv` is the result\n  if that "
          "happens in full. A widening gap would pay instead, and is "
          "deliberately\n  not counted: an uncontrolled exposure is a risk, "
          "not a source of return.")
    print("\n  Returns are on NOTIONAL, not on capital. The spot leg is fully "
          "funded and the\n  perp short needs margin on top, so the capital "
          "deployed exceeds the notional\n  these basis points are measured "
          "against. Divide accordingly - and note that\n  leverage on the perp "
          "leg is what makes a delta-neutral position liquidatable.")
    print("\n  Liquidation is not modelled at all.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instruments", default=None,
                        help="Comma-separated instrument ids. Default: every "
                             "pair with both a spot and a linear perp market.")
    parser.add_argument("--hold-days", type=float, default=DEFAULT_HOLD_DAYS)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--funding-pages", type=int, default=1,
                        help="Pages of funding history to walk back, 100 "
                             "periods (~33 days) each. The default single "
                             "page is one regime; 6 is most of a year.")
    parser.add_argument("--min-volume-usd", type=float, default=1e6,
                        help="Drop thin spot books; a carry you cannot size "
                             "is not an opportunity.")
    parser.add_argument("--spot-maker-bps", type=float, default=None)
    parser.add_argument("--spot-taker-bps", type=float, default=None)
    args = parser.parse_args(argv)

    from blofin.client import Client
    from blofin.rest_market import MarketAPI

    client = Client()
    api = MarketAPI(client)

    print("Fetching markets...")
    spot_instruments = fetch_spot_instruments(client)
    spot_tickers = fetch_spot_tickers(client)
    perp_tickers = {row["instId"]: row for row in
                    (api.getInstruments().get("data") or [])
                    if row.get("contractType") == "linear"}
    live_perp_tickers = {row["instId"]: row
                         for row in (api.getTickers().get("data") or [])}

    both = sorted(set(spot_instruments) & set(perp_tickers))
    print(f"  {len(spot_instruments)} spot, {len(perp_tickers)} linear perps, "
          f"{len(both)} with both.")

    if args.instruments:
        wanted = [part.strip() for part in args.instruments.split(",")
                  if part.strip()]
    else:
        wanted = both

    rows: List[Carry] = []
    for index, inst_id in enumerate(wanted, start=1):
        if inst_id not in spot_instruments:
            rows.append(Carry(inst_id, unavailable="no spot market"))
            continue
        spot_ticker = spot_tickers.get(inst_id)
        book = top_of_book(spot_ticker)
        if book and args.min_volume_usd > 0:
            try:
                volume = float(spot_ticker.get("volCurrency24h") or 0) * book[2]
            except (TypeError, ValueError):
                volume = 0.0
            if volume < args.min_volume_usd:
                rows.append(Carry(inst_id, unavailable="spot volume below floor"))
                continue
        history = fetch_funding_history(api, inst_id, pages=args.funding_pages)
        if not history:
            rows.append(Carry(inst_id, unavailable="no funding history"))
            continue
        rows.append(build(inst_id, history, spot_ticker,
                          live_perp_tickers.get(inst_id)))
        if index % 20 == 0:
            print(f"  {index}/{len(wanted)}...", flush=True)

    spot_maker = (args.spot_maker_bps if args.spot_maker_bps is not None
                  else DEFAULT_SPOT_MAKER_BPS)
    spot_taker = (args.spot_taker_bps if args.spot_taker_bps is not None
                  else DEFAULT_SPOT_TAKER_BPS)
    if spot_maker is None or spot_taker is None:
        raise SystemExit(
            f"No spot fee schedule for VIP {TIER}. perpkit.fees.SPOT_VIP_TIERS only "
            "holds tiers read off an\naccount, and guessing the rest would "
            "move every verdict below silently.\nPass --spot-maker-bps and "
            "--spot-taker-bps, or add the tier to perpkit/fees.py."
        )

    report(rows, spot_maker=spot_maker, spot_taker=spot_taker,
           hold_days=args.hold_days, top=args.top,
           spot_fees_confirmed=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
