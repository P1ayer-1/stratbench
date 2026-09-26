"""Where Hyperliquid positions liquidate - read off the accounts, not modelled.

Why this is a different kind of object from anything else here
---------------------------------------------------------------
On a venue like BloFin, which does not publish other accounts' positions, a
liquidation map would have to be a *reconstruction*. Bucket the open interest added at each price by an
assumed leverage mix, roll it through the MMR formula, and call the result a
heat-shaped prior - because leverage, margin mode, top-ups and partial closes
are all unobservable there.

Hyperliquid is a public ledger. Every account's positions are readable by
anyone, and `clearinghouseState` returns the exchange's own `liquidationPx`
for each one. So nothing on this page is assumed: every level is the sum of
positions whose liquidation price the exchange itself reported. What IS
uncertain is different in kind, and it is reported beside the map rather
than folded into it:

  * **Coverage.** Accounts are discovered from the trade feed, so the map
    only contains positions whose owners have traded since the tracker
    started. Tracked size against the exchange's open interest says how much
    of the market that is. A map at 20% coverage is a sample, and says so.
  * **Positions with no liquidation price at all.** In one live sample, 64
    of 121 positions sampled on BTC, ETH, SOL and HYPE reported
    `liquidationPx: null` - over half. Those are accounts collateralised well
    enough that no price of that coin alone liquidates them. Dropping them
    silently would make the map look like it covers the positions it
    ignores, so they are counted.

    This is a LONG-side phenomenon, structurally. Price can fall no further
    than zero, so a long can be over-collateralised past it; a price can rise
    without limit, so every short has a liquidation price somewhere. The
    first live maps showed 36-89% of tracked long size unpriced and 0% of
    short size on all four coins, which is that asymmetry and not a parsing
    bug. Read the down-side bands as covering fewer of the longs than the
    up-side bands cover of the shorts.
  * **Age.** A cross-margin liquidation price is computed at the moment of
    the read, holding every OTHER position the account has at its mark. BTC
    and ETH do not hold still for each other, so a reading drifts. Each map
    carries the notional-weighted age of what it is built from.
  * **Already crossed.** A long whose reported liquidation price sits above
    the current mark is either a stale reading (the account has since added
    margin, or its other positions moved) or a liquidation in progress. It
    belongs in neither band, so it is counted on its own.

Floats, not Decimal, deliberately: this is an aggregate description of other
people's positions, in the same family as the feature engine, and nothing
here is compared against a margin engine the way `risk.py` is.

No I/O. `hyperliquid.py` feeds it; the tests feed it by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Band width as a fraction of mark. 0.5% is ~$385 on BTC at $77k: fine enough
# to separate a cluster at -2% from one at -3%, coarse enough that a band is a
# cluster rather than one account.
DEFAULT_BUCKET_PCT = 0.005

# Beyond this distance the levels stop being bands and become one tail total.
# Measured short liquidation prices run to 3x the mark (209,806 on BTC at
# 77,143), and a thousand empty 0.5% bands between here and there is noise
# rather than information.
DEFAULT_MAX_DISTANCE_PCT = 0.25

# The distances a summary line quotes: "how much liquidates on a 1% move".
DEFAULT_WITHIN_PCT = (0.01, 0.02, 0.05, 0.10)


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@dataclass(frozen=True)
class TrackedPosition:
    """One account's position in one coin, as the exchange reported it."""

    user: str
    coin: str
    size: float                     # signed base units: + long, - short
    entry_px: float
    liquidation_px: Optional[float]  # None = no price of this coin alone
    position_value: float           # |notional| at the mark when read, USD
    leverage_type: str              # "cross" | "isolated"
    leverage: float
    observed_ms: int                # exchange time of the read

    @property
    def is_long(self) -> bool:
        return self.size > 0


def positions_from_state(
    user: str,
    state: Dict[str, Any],
    *,
    observed_ms: int,
    coins: Optional[Iterable[str]] = None,
) -> List[TrackedPosition]:
    """Parse a `clearinghouseState` response into positions.

    `coins` restricts the result; the archive keeps the whole response
    regardless, because a cross account's other positions are what move its
    liquidation price.
    """
    wanted = set(coins) if coins is not None else None
    out: List[TrackedPosition] = []
    for item in state.get("assetPositions") or []:
        position = (item or {}).get("position") or {}
        coin = position.get("coin")
        if not coin or (wanted is not None and coin not in wanted):
            continue
        size = _float(position.get("szi"))
        if not size:
            continue
        liquidation = _float(position.get("liquidationPx"))
        if liquidation is not None and liquidation <= 0:
            liquidation = None
        leverage = position.get("leverage") or {}
        out.append(TrackedPosition(
            user=user,
            coin=coin,
            size=size,
            entry_px=_float(position.get("entryPx")) or 0.0,
            liquidation_px=liquidation,
            position_value=abs(_float(position.get("positionValue")) or 0.0),
            leverage_type=str(leverage.get("type") or ""),
            leverage=_float(leverage.get("value")) or 0.0,
            observed_ms=observed_ms,
        ))
    return out


@dataclass
class Level:
    """Positions liquidating within one band of distance from the mark."""

    distance_low: float    # fraction of mark, inclusive
    distance_high: float   # fraction of mark, exclusive
    price_near: float      # band edge nearest the mark
    price_far: float
    size: float            # base units that hit the book in this band
    notional: float        # size at each position's liquidation price, USD
    positions: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_pct": round(self.distance_low * 100, 4),
            "to_pct": round(self.distance_high * 100, 4),
            "price_near": self.price_near,
            "price_far": self.price_far,
            "size": self.size,
            "notional": round(self.notional, 2),
            "positions": self.positions,
        }


@dataclass
class SideSummary:
    """Everything tracked on one side, including what is not in the bands."""

    size: float = 0.0             # all tracked positions on this side
    accounts: int = 0
    no_liquidation_size: float = 0.0
    crossed_size: float = 0.0     # liquidation price already past the mark
    beyond_size: float = 0.0      # further than max_distance_pct
    beyond_notional: float = 0.0
    within: Dict[float, float] = field(default_factory=dict)  # pct -> notional
    levels: List[Level] = field(default_factory=list)          # nearest first

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "accounts": self.accounts,
            "no_liquidation_size": self.no_liquidation_size,
            "crossed_size": self.crossed_size,
            "beyond_size": self.beyond_size,
            "beyond_notional": round(self.beyond_notional, 2),
            "within_notional": {f"{pct * 100:g}%": round(value, 2)
                                for pct, value in self.within.items()},
            "levels": [level.to_dict() for level in self.levels],
        }


@dataclass
class LiquidationMap:
    coin: str
    mark_px: float
    as_of_ms: int
    bucket_pct: float
    max_distance_pct: float
    open_interest: Optional[float]
    longs: SideSummary      # liquidate on a FALL
    shorts: SideSummary     # liquidate on a RISE
    accounts: int
    weighted_age_s: Optional[float]

    def coverage(self) -> Dict[str, Optional[float]]:
        """Tracked size as a fraction of the exchange's open interest.

        Assumes `openInterest` counts ONE side - total longs, which in a perp
        equal total shorts. Hyperliquid's docs do not say. If it counted both,
        neither side could ever exceed 50%, so a coverage above 0.5 on either
        side is the observation that settles it.
        """
        if not self.open_interest or self.open_interest <= 0:
            return {"long": None, "short": None}
        return {
            "long": self.longs.size / self.open_interest,
            "short": self.shorts.size / self.open_interest,
        }

    def to_dict(self) -> Dict[str, Any]:
        coverage = self.coverage()
        return {
            "coin": self.coin,
            "as_of_ms": self.as_of_ms,
            "mark_px": self.mark_px,
            "open_interest": self.open_interest,
            "coverage": {side: (None if value is None else round(value, 4))
                         for side, value in coverage.items()},
            "accounts": self.accounts,
            "weighted_age_s": (None if self.weighted_age_s is None
                               else round(self.weighted_age_s, 1)),
            "bucket_pct": self.bucket_pct,
            "max_distance_pct": self.max_distance_pct,
            "longs": self.longs.to_dict(),
            "shorts": self.shorts.to_dict(),
        }


def _distance(position: TrackedPosition, mark: float) -> Optional[float]:
    """Adverse move, as a fraction of mark, that reaches the liquidation price.

    Positive when the price is still ahead of the mark; zero or negative when
    the mark has already crossed it.
    """
    if position.liquidation_px is None:
        return None
    if position.is_long:
        return (mark - position.liquidation_px) / mark
    return (position.liquidation_px - mark) / mark


def build_map(
    coin: str,
    positions: Iterable[TrackedPosition],
    *,
    mark_px: float,
    as_of_ms: int,
    open_interest: Optional[float] = None,
    bucket_pct: float = DEFAULT_BUCKET_PCT,
    max_distance_pct: float = DEFAULT_MAX_DISTANCE_PCT,
    within_pct: Sequence[float] = DEFAULT_WITHIN_PCT,
) -> LiquidationMap:
    """Band every tracked position in `coin` by its distance to liquidation."""
    if mark_px <= 0:
        raise ValueError(f"mark price must be positive, got {mark_px}")
    if bucket_pct <= 0:
        raise ValueError(f"bucket_pct must be positive, got {bucket_pct}")

    sides = {True: SideSummary(), False: SideSummary()}
    for summary in sides.values():
        summary.within = {pct: 0.0 for pct in within_pct}
    bands: Dict[bool, Dict[int, Level]] = {True: {}, False: {}}
    side_accounts: Dict[bool, set] = {True: set(), False: set()}
    accounts = set()
    age_weighted = 0.0
    age_weight = 0.0

    for position in positions:
        if position.coin != coin or not position.size:
            continue
        long = position.is_long
        summary = sides[long]
        size = abs(position.size)
        summary.size += size
        side_accounts[long].add(position.user)
        accounts.add(position.user)

        weight = position.position_value or size * mark_px
        age_weighted += max(0, as_of_ms - position.observed_ms) / 1000 * weight
        age_weight += weight

        distance = _distance(position, mark_px)
        if distance is None:
            summary.no_liquidation_size += size
            continue
        if distance <= 0:
            summary.crossed_size += size
            continue

        notional = size * position.liquidation_px
        for pct in within_pct:
            if distance <= pct:
                summary.within[pct] += notional
        if distance >= max_distance_pct:
            summary.beyond_size += size
            summary.beyond_notional += notional
            continue

        index = int(distance // bucket_pct)
        level = bands[long].get(index)
        if level is None:
            low, high = index * bucket_pct, (index + 1) * bucket_pct
            sign = -1.0 if long else 1.0
            level = Level(
                distance_low=low,
                distance_high=high,
                price_near=mark_px * (1 + sign * low),
                price_far=mark_px * (1 + sign * high),
                size=0.0,
                notional=0.0,
                positions=0,
            )
            bands[long][index] = level
        level.size += size
        level.notional += notional
        level.positions += 1

    for long, summary in sides.items():
        summary.accounts = len(side_accounts[long])
        summary.levels = [bands[long][index] for index in sorted(bands[long])]

    return LiquidationMap(
        coin=coin,
        mark_px=mark_px,
        as_of_ms=as_of_ms,
        bucket_pct=bucket_pct,
        max_distance_pct=max_distance_pct,
        open_interest=open_interest,
        longs=sides[True],
        shorts=sides[False],
        accounts=len(accounts),
        weighted_age_s=(age_weighted / age_weight) if age_weight else None,
    )
