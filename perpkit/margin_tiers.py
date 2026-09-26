"""The maintenance margin rate for a position, looked up rather than assumed.

`risk.liquidation_price()` takes an MMR and the docstring has always said to
use "the *tier* that applies to your intended size, not the base tier". Until
now nothing could: the rate was only observable from a position that already
existed, so the planner carried a measured constant (0.005, read off a real
SOL position) and hoped it generalised.

It does not generalise, in two directions at once.

**MMR is tiered by size.** BTC-USDT alone publishes 101 tiers. A position that
outgrows its tier is charged the next one, and the liquidation price moves
with it.

**MMR is per environment.** Measured 2026-09-09, SUI-USDT isolated:

    production   tier 1   0 - 4,000    MMR 0.0050   maxLev 100
    demo         tier 1   0 - 30,000   MMR 0.0065   maxLev  50

A demo carry was charged 0.0065 while the production table said 0.0050 - a 30%
error, in the direction that makes liquidation look further away than it is.
Nothing was wrong with the formula; the input came from the wrong host. So
tiers are fetched from the SAME host the account lives on, and a planner that
cannot reach that host does not get to guess.

The demo/production gap is worth stating plainly for its own sake: demo margin
is stricter than production here, so a demo result is conservative rather than
representative. That is the safe direction for a test and the wrong direction
for quoting a number.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

ZERO = Decimal("0")


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None


class Tier:
    """One row of the exchange's margin schedule."""

    __slots__ = ("min_size", "max_size", "mmr", "max_leverage")

    def __init__(self, min_size: Decimal, max_size: Decimal, mmr: Decimal,
                 max_leverage: Decimal):
        self.min_size = min_size
        self.max_size = max_size
        self.mmr = mmr
        self.max_leverage = max_leverage

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Tier({self.min_size}-{self.max_size}, mmr={self.mmr}, "
                f"maxLev={self.max_leverage})")

    def covers(self, contracts: Decimal) -> bool:
        return self.min_size <= contracts <= self.max_size


def parse_tiers(rows: Sequence[Dict[str, Any]]) -> List[Tier]:
    """Exchange rows -> tiers, sorted by size and with junk dropped."""
    tiers: List[Tier] = []
    for row in rows or []:
        minimum = _decimal(row.get("minSize"))
        maximum = _decimal(row.get("maxSize"))
        mmr = _decimal(row.get("maintenanceMarginRate"))
        leverage = _decimal(row.get("maxLeverage"))
        if minimum is None or maximum is None or mmr is None:
            continue
        tiers.append(Tier(minimum, maximum, mmr,
                          leverage if leverage is not None else ZERO))
    tiers.sort(key=lambda tier: tier.min_size)
    return tiers


def tier_for(tiers: Sequence[Tier], contracts: Decimal) -> Optional[Tier]:
    """The tier a position of this size falls in.

    Above the last tier returns the last one rather than None: an oversized
    position is charged the strictest published rate, and refusing to answer
    would push the caller back onto a default, which is the thing this module
    exists to remove.
    """
    if not tiers:
        return None
    size = abs(contracts)
    for tier in tiers:
        if tier.covers(size):
            return tier
    return tiers[-1] if size > tiers[-1].max_size else tiers[0]


def fetch_tiers(client, inst_id: str, *,
                margin_mode: str = "isolated") -> List[Tier]:
    """Published margin schedule for one instrument, from `client`'s host.

    The host matters: pass the client pointed at the same environment the
    account is on, or the rate will describe a different exchange's rules.
    """
    payload = client.get("/api/v1/market/position-tiers",
                         params={"instId": inst_id, "marginMode": margin_mode},
                         sign=False)
    if str(payload.get("code")) != "0":
        return []
    return parse_tiers(payload.get("data") or [])


def maintenance_margin_rate(client, inst_id: str, contracts: Decimal, *,
                            margin_mode: str = "isolated") -> Optional[Decimal]:
    """MMR for this instrument at this size, or None if it cannot be read.

    None rather than a fallback, deliberately. A liquidation price computed
    from a guessed MMR is a number that looks measured and is not, and this
    project has already spent a session finding out what those cost.
    """
    tier = tier_for(fetch_tiers(client, inst_id, margin_mode=margin_mode),
                    contracts)
    return tier.mmr if tier else None
