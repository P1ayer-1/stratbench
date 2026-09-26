"""BloFin fee ladder and the round-trip cost model everything else prices with.

THE most load-bearing numbers in this package. Every "is there an edge?"
question is really "is the predicted move bigger than these?", so they are
defined once, here, and everything else imports them.

Provenance, because these decide everything
-------------------------------------------
Futures rows were transcribed on 2026-09-07 from BloFin's published fee
schedule and fee-structure pages. blofin.com serves 403 to automated fetches,
so tiers 0/1/2/5 were first read from search results quoting those pages and
then corroborated by an account's own fee display on 2026-09-09, which also
supplied tier 3. VIP 4 could not be confirmed and is deliberately **absent
rather than interpolated**: a guessed number here silently moves every
verdict downstream. None of the rows has been verified against a real fill.
**Confirm against your own account before sizing anything on this.** The tier
that bills you is the one you are actually on.

There is no maker rebate at any tier. The floor is 0.0000% maker at the top
tier, so no BloFin schedule ever PAYS you to provide liquidity; the best case
is that providing it becomes free. The consequence is easy to miss: the fee
can go to zero but adverse selection cannot. A passive fill that is picked off
costs a fraction of a basis point on a liquid perp, so a passive round trip
needs roughly a basis point of spread at ANY tier, and tier improvements buy
less and less as they approach that floor.

Qualification (as published at transcription time) is whichever of three
thresholds is hit first, refreshed daily:

  VIP 1   50,000 USDT held  |  10,000,000 USDT 30d futures  |  1,000,000 USDT 30d spot
  VIP 2                     |                               |  2,000,000 USDT 30d spot
  VIP 3+  volume-based; thresholds not transcribed here

The volume route is not free: 30-day futures volume is earned by trading at
your CURRENT tier's fees, so if a strategy loses at VIP 1 the path to VIP 3 is
paid for in losses, and that cost belongs in the decision.

Spot is a different schedule entirely. Its only row (VIP 1) was read off an
account on 2026-09-09: the spot maker fee is 3.5 bps against 0.6 on futures,
nearly six times as much. A delta-neutral funding carry crosses FOUR legs (two
perp, two spot), and pricing all four at futures rates understates half the
trade. Other spot tiers are absent, not guessed.

Default tier is VIP 0, deliberately: an account that has never traded is VIP
0, and a cost assumption that is optimistic is the most common way a
backtest lies. Set `BLOFIN_VIP_TIER` once the account actually qualifies,
not in anticipation of qualifying.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Dict, Optional, Tuple

# Transcription dates, shown by `python -m perpkit.fees` and in the README.
FUTURES_TRANSCRIBED = "2026-09-07 (tier 3: 2026-09-09)"
SPOT_TRANSCRIBED = "2026-09-09"
VERIFIED_AGAINST_A_FILL = False

VIP_TIERS: Dict[int, Tuple[Decimal, Decimal]] = {
    # tier: (maker, taker), fractions of notional per side
    0: (Decimal("0.00020"), Decimal("0.00060")),   # 0.0200% / 0.0600%
    1: (Decimal("0.00006"), Decimal("0.00050")),   # 0.0060% / 0.0500%
    2: (Decimal("0.00004"), Decimal("0.00045")),   # 0.0040% / 0.0450%
    3: (Decimal("0.00002"), Decimal("0.000425")),  # 0.0020% / 0.0425%
    5: (Decimal("0.00000"), Decimal("0.00035")),   # 0.0000% / 0.0350%
}

SPOT_VIP_TIERS: Dict[int, Tuple[Decimal, Decimal]] = {
    # tier: (maker, taker)
    1: (Decimal("0.00035"), Decimal("0.00060")),   # 0.0350% / 0.0600%
}


def check_tier_ladder(tiers: Dict[int, Tuple[Decimal, Decimal]]) -> None:
    """A fee schedule must not get worse as the tier improves.

    Every number above was transcribed by hand from somewhere else, and a
    transcription error silently moves every gate and verdict downstream. The
    one invariant that catches a whole class of them is asserted at import:
    fees never rise with the tier, and two tiers never share identical rates
    (which is what a mis-numbered reading looks like).
    """
    ordered = sorted(tiers.items())
    for (low_tier, (low_maker, low_taker)), (high_tier, (high_maker, high_taker)) in zip(
        ordered, ordered[1:]
    ):
        if high_maker > low_maker or high_taker > low_taker:
            raise SystemExit(
                f"fee ladder is not monotonic: tier {high_tier} "
                f"({high_maker}/{high_taker}) is worse than tier {low_tier} "
                f"({low_maker}/{low_taker}). One of them is transcribed wrong."
            )
        if (high_maker, high_taker) == (low_maker, low_taker):
            raise SystemExit(
                f"fee ladder has tier {low_tier} and tier {high_tier} on "
                f"identical rates ({low_maker}/{low_taker}). That is what a "
                "mis-numbered reading looks like; confirm which tier is real "
                "and drop the other."
            )
        if high_maker < 0 or high_taker < high_maker:
            raise SystemExit(
                f"tier {high_tier}: taker {high_taker} below maker {high_maker}, "
                "or a negative maker fee. BloFin publishes neither.")


check_tier_ladder(VIP_TIERS)
check_tier_ladder(SPOT_VIP_TIERS)

VIP_TIER = int(os.getenv("BLOFIN_VIP_TIER", "0"))
if VIP_TIER not in VIP_TIERS:
    raise SystemExit(
        f"BLOFIN_VIP_TIER={VIP_TIER} is not a tier this file has rates for. "
        f"Known: {sorted(VIP_TIERS)}. VIP 4 exists on BloFin but its rates "
        "were never confirmed, so they are not guessed at here."
    )

_TIER_MAKER, _TIER_TAKER = VIP_TIERS[VIP_TIER]

MAKER_FEE_RATE = Decimal(os.getenv("BLOFIN_MAKER_FEE_RATE", str(_TIER_MAKER)))
TAKER_FEE_RATE = Decimal(os.getenv("BLOFIN_TAKER_FEE_RATE", str(_TIER_TAKER)))


def bps(rate: Decimal) -> Decimal:
    """A fraction of notional as basis points."""
    return rate * Decimal("10000")


# Round-trip cost in bps for the three ways a position can open and close.
#
# The spread is not in these, and on a small-tick instrument like BTC-USDT
# it is a rounding error (hundredths of a bp). On a low-priced coin the
# minimum tick is a large fraction of the price and the median spread can be
# several bps, at which point it is the entire revenue of a passive strategy.
# Measure it per instrument before assuming it away.
#
# Comments show bps at VIP 0, the default.
MAKER_FEE_BPS = bps(MAKER_FEE_RATE)                     # 2.0
TAKER_FEE_BPS = bps(TAKER_FEE_RATE)                     # 6.0
COST_MAKER_MAKER_BPS = MAKER_FEE_BPS * 2                # 4.0  passive in, passive out
COST_MAKER_TAKER_BPS = MAKER_FEE_BPS + TAKER_FEE_BPS    # 8.0  passive in, market out
COST_TAKER_TAKER_BPS = TAKER_FEE_BPS * 2                # 12.0 crossing both ways

_SPOT_RATES = SPOT_VIP_TIERS.get(VIP_TIER)
# None rather than a fallback, so a caller on an unconfirmed tier has to say
# what it is assuming instead of inheriting a number that was never checked.
SPOT_MAKER_FEE_BPS: Optional[Decimal] = bps(_SPOT_RATES[0]) if _SPOT_RATES else None
SPOT_TAKER_FEE_BPS: Optional[Decimal] = bps(_SPOT_RATES[1]) if _SPOT_RATES else None

# The cost the risk engine's edge gate and the analysis tooling assume by
# default. Taker/taker deliberately: it is a VETO threshold, and the
# conservative assumption is that you cross the spread on both sides. Set
# BLOFIN_ROUND_TRIP_COST_BPS lower only once a passive execution engine exists
# and its fill rate has been measured, not before.
ROUND_TRIP_COST_BPS = Decimal(
    os.getenv("BLOFIN_ROUND_TRIP_COST_BPS", str(COST_TAKER_TAKER_BPS))
)


def round_trip_bps(entry: str, exit: str, *, tier: Optional[int] = None) -> Decimal:
    """Round-trip fee in bps for `entry`/`exit` roles ("maker" or "taker").

    An unknown tier is an error naming the known ones: fees are never guessed.
    """
    chosen = VIP_TIER if tier is None else tier
    if chosen not in VIP_TIERS:
        raise KeyError(f"no BloFin futures rates for VIP {chosen}; known: {sorted(VIP_TIERS)}")
    maker, taker = VIP_TIERS[chosen]
    rates = {"maker": maker, "taker": taker}
    for role in (entry, exit):
        if role not in rates:
            raise ValueError(f"role must be 'maker' or 'taker', not {role!r}")
    return bps(rates[entry] + rates[exit])


def main() -> int:
    print(f"BloFin futures (transcribed {FUTURES_TRANSCRIBED}; verified against a fill: "
          f"{VERIFIED_AGAINST_A_FILL})")
    for tier, (maker, taker) in sorted(VIP_TIERS.items()):
        mark = "  <- BLOFIN_VIP_TIER" if tier == VIP_TIER else ""
        print(f"  VIP {tier}  maker {bps(maker):>5.2f} bps  taker {bps(taker):>5.2f} bps{mark}")
    print(f"BloFin spot (transcribed {SPOT_TRANSCRIBED})")
    for tier, (maker, taker) in sorted(SPOT_VIP_TIERS.items()):
        print(f"  VIP {tier}  maker {bps(maker):>5.2f} bps  taker {bps(taker):>5.2f} bps")
    print(f"Round trip at VIP {VIP_TIER}: maker/maker {COST_MAKER_MAKER_BPS} bps, "
          f"maker/taker {COST_MAKER_TAKER_BPS}, taker/taker {COST_TAKER_TAKER_BPS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
