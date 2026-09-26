"""Fees decide every verdict. Hand-transcribed, dated, checked at import.

Both venues charge takers on the same curve, per contract:

    fee = c * p * (1 - p)        c = 0.07 on the standard tier

which is 0 at either extreme and peaks at 1.75 cents on a 50 cent contract:
exactly where a fair-value bot wants to trade. Makers pay nothing on most
tiers transcribed here; on Polymarket they are paid a share of the taker fee,
which is NOT modelled (the share is recorded per row, not credited).

The rule: unknown tiers are ABSENT, never interpolated. A guessed fee silently
moves every backtest and every plan downstream, and many apparent edges on
these markets disappear once the cost assumption is made honest. Asking for a
tier that is not here is an error that names the tiers that are.

Provenance, per row (dates are when the row was transcribed)
------------------------------------------------------------
kalshi / default             0.07 curve, taker only (1.75% at 50 cents).
                             Kalshi's published schedule rounds the fee UP to
                             the next cent per order; that rounding was
                             transcribed from the schedule, not checked
                             against a fill, and the row is marked unverified.
                             Some Kalshi markets charge makers; those are a
                             different tier (next row).
kalshi / quadratic_with_maker_fees
                             Kalshi `/series/<s>` `fee_type` string, read live
                             2026-09-24 on KXCPI, KXCPIYOY, KXLLM1 and
                             KXSUPERBOWLHEADLINE (fee_multiplier 1 on each).
                             Curve from the Kalshi fee schedule PDF "Last
                             updated and effective: July 7, 2026": taker round
                             up(M x 0.07 x C x P x (1-P)); maker round up(M x
                             0.0175 x C x P x (1-P)), charged only when a
                             resting order is executed. The PDF's non-standard
                             table lists maker 1 / taker 1 for these series,
                             so M = 1 here. Rounding: the PDF's text says to a
                             centicent, its own worked table to the cent per
                             order; the cent (stricter) is transcribed. Not
                             verified against a fill.
kalshi / quadratic           Kalshi `/series/<s>` `fee_type` string on the
                             "mention" series, read live 2026-09-25 on
                             KXTRUMPMENTION, KXFEDMENTION, KXVANCEMENTION,
                             KXMAMDANIMENTION, KXLEAVITTMENTION and KXTRUMPSAY
                             (fee_multiplier 1 on each). Same fee schedule PDF
                             as the row above: taker round up(0.07 x C x P x
                             (1-P)) per order, no maker fee (the maker line
                             applies only to the non-standard table's series).
                             Curve identical to `default`, which it names in
                             `same_curve_as`. Hand check: 10 at 0.50 =
                             ceil(0.175) = $0.18; 100 at 0.85 = ceil(0.8925) =
                             $0.90. Not verified against a fill.
polymarket / crypto_fees_v2  Gamma's own `feeType` on the 5- and 15-minute
                             BTC markets, read live 2026-09-13, with
                             `feeSchedule {"rate": 0.07, "exponent": 1,
                             "takerOnly": true, "rebateRate": 0.2}`. The
                             transcription ASSUMES `rate` is the `c` of this
                             curve (it matches the 1.75%-at-50c the venue's
                             docs describe); the 0.2 rebate to makers is
                             recorded, not modelled. A January 2026 press
                             report of ~3.15% at 50c on the 15-minute markets
                             (c = 0.126) disagrees with the listing and is
                             dropped in favour of the venue's own field.
polymarket / economics_fees  Gamma `feeType` on a Fed market, read live
                             2026-09-13: rate 0.05, exponent 1, taker only,
                             rebate 0.25.
polymarket / weather_fees    Gamma `feeType` on every market of
                             `highest-temperature-in-nyc-on-september-17-2026`,
                             read live 2026-09-24: rate 0.05, exponent 1,
                             taker only, rebate 0.25; rate again ASSUMED to be
                             the curve coefficient. Its curve is identical to
                             `economics_fees`, which the duplicate check below
                             reads as a mis-named reading. Both strings were
                             read live under different `feeType`s, so they are
                             two real tiers with the same numbers; the row says
                             so explicitly with `same_curve_as`, and a
                             duplicate pair neither row names is still refused.

Fees change. Every row is a snapshot on its date; re-read the venue's
schedule before trusting a verdict, and add a new row (with a date and a
source) rather than editing an old one in place.

Polymarket tiers are keyed by Gamma's `feeType` string, which the adapter
copies onto `Contract.fee_tier` verbatim. A market whose `feeType` is not
here fails at fee time by name, which is the design: nothing is guessed
from the question text. Nothing here is verified against a fill.

Builder fees (Polymarket only) are charged on notional, on top of the venue
fee, capped at 100 bps taker and 50 bps maker by the Builder Program. The
default here is ZERO; the caps are asserted at import so a misconfigured
environment cannot exceed them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Dict, List, Optional, Tuple

from predkit.schema import CENT, ONE, ZERO, check_price, to_decimal

Role = str   # "maker" | "taker"


@dataclass(frozen=True)
class Curve:
    """fee per contract = coefficient * p * (1 - p), optionally rounded up
    to the cent per order (Kalshi)."""

    coefficient: Decimal
    round_up_to_cent: bool = False

    def per_contract(self, price: Decimal) -> Decimal:
        return self.coefficient * price * (ONE - price)

    def for_order(self, price: Decimal, size: Decimal) -> Decimal:
        total = self.per_contract(price) * size
        if self.round_up_to_cent:
            total = total.quantize(CENT, rounding=ROUND_CEILING)
        return total


@dataclass(frozen=True)
class Tier:
    venue: str
    name: str
    maker: Curve
    taker: Curve
    dated: str
    source: str
    verified: bool
    rebate_rate: Decimal = ZERO     # share of the taker fee paid to makers; recorded, not modelled
    # Other tiers of this venue this row KNOWINGLY shares a curve with (both read
    # live under different feeType strings). Without it an identical curve is
    # refused at import as a mis-named reading.
    same_curve_as: Tuple[str, ...] = ()

    def curve(self, role: Role) -> Curve:
        if role == "maker":
            return self.maker
        if role == "taker":
            return self.taker
        raise ValueError(f"role must be 'maker' or 'taker', not {role!r}")


NO_FEE = Curve(ZERO)
STANDARD = Curve(Decimal("0.07"))

TABLES: Dict[Tuple[str, str], Tier] = {
    ("kalshi", "default"): Tier(
        venue="kalshi", name="default",
        maker=NO_FEE,
        taker=Curve(Decimal("0.07"), round_up_to_cent=True),
        dated="2026-09-12",
        source="Kalshi published fee schedule (0.07 x C x P x (1-P), 1.75% at 50c); cent "
               "rounding per order from the same schedule, not checked against a fill",
        verified=False,
    ),
    ("kalshi", "quadratic_with_maker_fees"): Tier(
        venue="kalshi", name="quadratic_with_maker_fees",
        maker=Curve(Decimal("0.0175"), round_up_to_cent=True),
        taker=Curve(Decimal("0.07"), round_up_to_cent=True),
        dated="2026-09-24",
        source="Kalshi fee schedule PDF effective 2026-07-07 (maker 0.0175 x M x C x P x (1-P), "
               "M = 1 per its non-standard table for KXCPI, KXCPIYOY, KXLLM1, "
               "KXSUPERBOWLHEADLINE); fee_type read live from /series 2026-09-24; cent rounding "
               "per order from the PDF's worked table",
        verified=False,
    ),
    ("kalshi", "quadratic"): Tier(
        venue="kalshi", name="quadratic",
        maker=NO_FEE,
        taker=Curve(Decimal("0.07"), round_up_to_cent=True),
        dated="2026-09-25",
        source="Kalshi /series fee_type 'quadratic', fee_multiplier 1, read live 2026-09-25 on the "
               "mention series (KXTRUMPMENTION, KXFEDMENTION, KXVANCEMENTION, KXMAMDANIMENTION, "
               "KXLEAVITTMENTION, KXTRUMPSAY); curve and cent round-up per order from the fee schedule "
               "PDF effective 2026-07-07",
        verified=False,
        same_curve_as=("default",),
    ),
    ("polymarket", "crypto_fees_v2"): Tier(
        venue="polymarket", name="crypto_fees_v2",
        maker=NO_FEE,
        taker=Curve(Decimal("0.07")),
        dated="2026-09-13",
        source="Gamma feeSchedule on btc-updown-5m and -15m markets: rate 0.07, exponent 1, "
               "takerOnly, rebateRate 0.2; rate assumed to be the curve coefficient",
        verified=False,
        rebate_rate=Decimal("0.2"),
    ),
    ("polymarket", "economics_fees"): Tier(
        venue="polymarket", name="economics_fees",
        maker=NO_FEE,
        taker=Curve(Decimal("0.05")),
        dated="2026-09-13",
        source="Gamma feeSchedule on a Fed market: rate 0.05, exponent 1, takerOnly, "
               "rebateRate 0.25; rate assumed to be the curve coefficient",
        verified=False,
        rebate_rate=Decimal("0.25"),
    ),
    ("polymarket", "weather_fees"): Tier(
        venue="polymarket", name="weather_fees",
        maker=NO_FEE,
        taker=Curve(Decimal("0.05")),
        dated="2026-09-24",
        source="Gamma feeType weather_fees, feeSchedule on every market of "
               "highest-temperature-in-nyc-on-september-17-2026: rate 0.05, exponent 1, "
               "takerOnly, rebateRate 0.25; rate assumed to be the curve coefficient",
        verified=False,
        rebate_rate=Decimal("0.25"),
        same_curve_as=("economics_fees",),
    ),
}


def tiers_for(venue: str) -> Tuple[str, ...]:
    return tuple(sorted(name for (v, name) in TABLES if v == venue))


def tier(venue: str, name: str) -> Tier:
    try:
        return TABLES[(venue, name)]
    except KeyError:
        known = tiers_for(venue)
        raise KeyError(
            f"no fee table for venue={venue!r} tier={name!r}. Known tiers for "
            f"{venue!r}: {list(known) or 'none'}. Tiers are transcribed by hand and an "
            "unknown one is absent rather than guessed; add it to fees.TABLES with a "
            "date and a source.") from None


def fee(venue: str, tier_name: str, role: Role, price, size) -> Decimal:
    """Venue fee for one order, in dollars. `role` is 'maker' or 'taker'."""
    price = check_price(price)
    size = to_decimal(size)
    if size < ZERO:
        raise ValueError(f"size {size} is negative")
    return tier(venue, tier_name).curve(role).for_order(price, size)


# --- Builder attribution fee (Polymarket) ----------------------------------

BUILDER_TAKER_CAP_BPS = Decimal("100")
BUILDER_MAKER_CAP_BPS = Decimal("50")
BPS = Decimal("10000")

BUILDER_TAKER_BPS = Decimal(os.getenv("PREDKIT_BUILDER_TAKER_BPS", "0"))
BUILDER_MAKER_BPS = Decimal(os.getenv("PREDKIT_BUILDER_MAKER_BPS", "0"))


def builder_fee(role: Role, price, size, *, taker_bps: Optional[Decimal] = None,
                maker_bps: Optional[Decimal] = None) -> Decimal:
    """The toolkit's own fee on notional, in dollars. Zero by default."""
    price = check_price(price)
    size = to_decimal(size)
    if role == "taker":
        rate = BUILDER_TAKER_BPS if taker_bps is None else taker_bps
        cap = BUILDER_TAKER_CAP_BPS
    elif role == "maker":
        rate = BUILDER_MAKER_BPS if maker_bps is None else maker_bps
        cap = BUILDER_MAKER_CAP_BPS
    else:
        raise ValueError(f"role must be 'maker' or 'taker', not {role!r}")
    if rate < ZERO or rate > cap:
        raise ValueError(f"builder {role} fee {rate} bps is outside [0, {cap}] bps")
    return price * size * rate / BPS


def total_fee(venue: str, tier_name: str, role: Role, price, size) -> Decimal:
    """Venue fee plus builder fee: what a fill actually costs the user."""
    return fee(venue, tier_name, role, price, size) + (
        builder_fee(role, price, size) if venue == "polymarket" else ZERO)


# --- Import-time checks ------------------------------------------------------

_GRID = [Decimal(n) / 100 for n in range(0, 101)]


def _check_tables() -> None:
    """A transcription error has a shape, and this catches the common ones.

    Per tier: the taker never pays less than the maker at any price; nothing
    is negative; the curve is zero at both extremes, symmetric about 50 cents
    and peaks there. Per venue: no two tiers on identical curves, which is
    what a mis-named reading looks like, unless one of the two rows names
    the other in `same_curve_as` (weather_fees and economics_fees, both read
    live, 2026-09-24). Builder rates within the program's
    caps, because an environment variable set to 150 must fail here, not on
    the first live fill.
    """
    for (venue, name), row in TABLES.items():
        if (row.venue, row.name) != (venue, name):
            raise SystemExit(f"fees.TABLES key {(venue, name)} disagrees with its row")
        for role in ("maker", "taker"):
            curve = row.curve(role)
            if curve.coefficient < ZERO:
                raise SystemExit(f"{venue}/{name} {role} coefficient is negative")
            values = [curve.per_contract(p) for p in _GRID]
            if values[0] != ZERO or values[-1] != ZERO:
                raise SystemExit(f"{venue}/{name} {role} fee is not zero at 0 or 1")
            if values != values[::-1]:
                raise SystemExit(f"{venue}/{name} {role} fee is not symmetric about 50c")
            if max(values) != values[50]:
                raise SystemExit(f"{venue}/{name} {role} fee does not peak at 50c")
        for p in _GRID:
            if row.taker.per_contract(p) < row.maker.per_contract(p):
                raise SystemExit(f"{venue}/{name}: maker fee exceeds taker fee at {p}; "
                                 "one of them is transcribed wrong")
    # A `same_curve_as` naming no tier of the same venue is a stale or mistyped
    # acknowledgement; it must fail here, not silently excuse nothing.
    stale = [f"{venue}/{name} same_curve_as {other!r}"
             for (venue, name), row in TABLES.items()
             for other in row.same_curve_as
             if other == name or (venue, other) not in TABLES]
    if stale:
        raise SystemExit("fees.TABLES same_curve_as names no other tier of the same venue: "
                         + "; ".join(stale) + ". Fix the name or drop the entry.")
    by_venue: Dict[str, Dict[Tuple[Decimal, bool, Decimal, bool], List[str]]] = {}
    for (venue, name), row in TABLES.items():
        shape = (row.maker.coefficient, row.maker.round_up_to_cent,
                 row.taker.coefficient, row.taker.round_up_to_cent)
        seen = by_venue.setdefault(venue, {}).setdefault(shape, [])
        for other in seen:
            other_row = TABLES[(venue, other)]
            if other in row.same_curve_as or name in other_row.same_curve_as:
                continue
            raise SystemExit(f"{venue}: tiers {other!r} and {name!r} have identical "
                             "curves. That is what a mis-named reading looks like; confirm "
                             "which is real and drop the other, or, if both were read live "
                             "under different feeType strings, name the other tier in "
                             "same_curve_as.")
        seen.append(name)
    for rate, cap, label in ((BUILDER_TAKER_BPS, BUILDER_TAKER_CAP_BPS, "TAKER"),
                             (BUILDER_MAKER_BPS, BUILDER_MAKER_CAP_BPS, "MAKER")):
        if rate < ZERO or rate > cap:
            raise SystemExit(f"PREDKIT_BUILDER_{label}_BPS={rate} is outside the Builder "
                             f"Program cap of {cap} bps")


_check_tables()

__all__ = ["BUILDER_MAKER_BPS", "BUILDER_TAKER_BPS", "Curve", "TABLES", "Tier",
           "builder_fee", "fee", "tier", "tiers_for", "total_fee"]
