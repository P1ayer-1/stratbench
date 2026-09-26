"""Maintenance margin, looked up instead of assumed.

The constant this replaces was measured off a real position and was still
wrong twice over: MMR is tiered by size, and the tables differ between demo
and production. A carry planned with production's SUI rate (0.0050) opened on
demo was charged 0.0065 - a 30% error in the direction that makes liquidation
look further away than it is.
"""

from decimal import Decimal

import pytest

from perpkit.margin_tiers import (
    fetch_tiers,
    maintenance_margin_rate,
    parse_tiers,
    tier_for,
)

# SUI-USDT isolated, as read from each host. The two hosts genuinely disagree.
PRODUCTION_ROWS = [
    {"minSize": "0", "maxSize": "4000", "maintenanceMarginRate": "0.005",
     "maxLeverage": "100"},
    {"minSize": "4001", "maxSize": "20000", "maintenanceMarginRate": "0.01",
     "maxLeverage": "75"},
]
DEMO_ROWS = [
    {"minSize": "0", "maxSize": "30000", "maintenanceMarginRate": "0.0065",
     "maxLeverage": "50"},
    {"minSize": "30001", "maxSize": "60000", "maintenanceMarginRate": "0.01",
     "maxLeverage": "40"},
]


class FakeClient:
    def __init__(self, rows, code="0"):
        self.rows = rows
        self.code = code
        self.calls = []

    def get(self, path, params=None, sign=False):
        self.calls.append((path, params))
        return {"code": self.code, "data": self.rows}


def test_the_tier_matching_the_size_is_chosen():
    tiers = parse_tiers(PRODUCTION_ROWS)
    assert tier_for(tiers, Decimal("254")).mmr == Decimal("0.005")
    assert tier_for(tiers, Decimal("10000")).mmr == Decimal("0.01")


def test_the_two_environments_disagree_and_that_is_the_point():
    """The bug this module exists for, as a test.

    Same instrument, same size, same margin mode - different answer per host.
    """
    production = tier_for(parse_tiers(PRODUCTION_ROWS), Decimal("254")).mmr
    demo = tier_for(parse_tiers(DEMO_ROWS), Decimal("254")).mmr

    assert production == Decimal("0.005")
    assert demo == Decimal("0.0065")
    assert demo > production, "demo is stricter, so a demo test is conservative"


def test_a_short_position_size_is_matched_on_its_magnitude():
    """Positions come back negative; tiers are quoted on size."""
    tiers = parse_tiers(PRODUCTION_ROWS)
    assert tier_for(tiers, Decimal("-254")).mmr == Decimal("0.005")


def test_a_size_above_every_tier_gets_the_strictest_rate():
    """Refusing would push the caller back onto a default, which is the thing
    this module exists to remove."""
    tiers = parse_tiers(PRODUCTION_ROWS)
    assert tier_for(tiers, Decimal("999999")).mmr == Decimal("0.01")


def test_a_gap_between_tiers_does_not_fall_through():
    """maxSize 4000 and the next minSize 4001 leaves 4000.5 uncovered."""
    tiers = parse_tiers(PRODUCTION_ROWS)
    assert tier_for(tiers, Decimal("4000.5")) is not None


def test_rows_are_sorted_regardless_of_the_order_sent():
    tiers = parse_tiers(list(reversed(PRODUCTION_ROWS)))
    assert tiers[0].min_size < tiers[1].min_size
    assert tier_for(tiers, Decimal("254")).mmr == Decimal("0.005")


def test_unparseable_rows_are_dropped_not_guessed():
    rows = PRODUCTION_ROWS + [{"minSize": "x", "maxSize": "y"}]
    assert len(parse_tiers(rows)) == 2


def test_no_tiers_means_no_rate_rather_than_a_fallback():
    """A liquidation price from a guessed MMR looks measured and is not."""
    assert tier_for([], Decimal("254")) is None
    assert maintenance_margin_rate(FakeClient([]), "SUI-USDT",
                                   Decimal("254")) is None


def test_an_api_error_yields_no_tiers():
    client = FakeClient(PRODUCTION_ROWS, code="152001")
    assert fetch_tiers(client, "SUI-USDT") == []


def test_the_margin_mode_is_passed_through():
    """Isolated and cross publish different schedules."""
    client = FakeClient(PRODUCTION_ROWS)
    fetch_tiers(client, "SUI-USDT", margin_mode="cross")
    assert client.calls[0][1]["marginMode"] == "cross"


def test_the_rate_comes_from_the_client_it_is_given():
    """Which is how the host gets to be the account's host."""
    demo = maintenance_margin_rate(FakeClient(DEMO_ROWS), "SUI-USDT",
                                   Decimal("254"))
    production = maintenance_margin_rate(FakeClient(PRODUCTION_ROWS),
                                         "SUI-USDT", Decimal("254"))
    assert demo == Decimal("0.0065")
    assert production == Decimal("0.005")
