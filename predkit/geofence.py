"""Who this software may not face, by declared residency. Checked before live.

Venue and regulatory facts, reduced to a table (as transcribed 2026-09; these
change, so re-read the sources before relying on them):

  Canada, all venues        Canadian Securities Administrators Multilateral
                            Instrument 91-102 prohibits advertising, offering
                            or selling a binary option with a term under 30
                            days to an individual. Every market this kit
                            trades is under 30 days. Polymarket reached a
                            settlement with a Canadian securities regulator
                            over access (2025-04-17), and Kalshi does not
                            accept Canadian sign-ups.
  United States, Polymarket The global Polymarket exchange geoblocks US
                            persons (CFTC order against Polymarket, January
                            2022). US residents use Kalshi (a CFTC-designated
                            contract market) or Polymarket US, which is a
                            separate exchange and would need a separate
                            adapter (not included).

This table is a FLOOR, not a whitelist. Both venues publish longer
restricted-jurisdiction lists in their terms of service, and local law can
be stricter than any venue's list. Passing this check does not mean trading
is lawful for you; that is the user's responsibility to establish.

This is a declared-residency check, ISO 3166-1 alpha-2. It is not an IP
geofence. The live runner refuses without a residency, and refuses the
combinations above with every reason listed. Paper mode is not gated:
nothing is offered or traded on paper.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

BLOCKED_EVERYWHERE: FrozenSet[str] = frozenset({"CA"})
BLOCKED_BY_VENUE: Dict[str, FrozenSet[str]] = {
    "polymarket": frozenset({"US"}),
}
KNOWN_VENUES = ("polymarket", "kalshi")


def refusals(venue: str, residency: str) -> List[str]:
    """Every reason not to face this user on this venue live. Empty means go."""
    reasons: List[str] = []
    code = (residency or "").strip().upper()
    if len(code) != 2 or not code.isalpha():
        reasons.append("residency must be an ISO 3166-1 alpha-2 country code, declared by "
                       f"the user; got {residency!r}")
        return reasons
    if venue not in KNOWN_VENUES:
        reasons.append(f"unknown venue {venue!r}; known: {list(KNOWN_VENUES)}")
    if code in BLOCKED_EVERYWHERE:
        reasons.append(f"residency {code}: binary options under 30 days may not be offered "
                       "to individuals (CSA Multilateral Instrument 91-102); refused on every venue")
    if code in BLOCKED_BY_VENUE.get(venue, frozenset()):
        reasons.append(f"residency {code} on {venue}: the global exchange geoblocks US "
                       "persons (CFTC order, 2022); use kalshi instead")
    return reasons


__all__ = ["BLOCKED_BY_VENUE", "BLOCKED_EVERYWHERE", "KNOWN_VENUES", "refusals"]
