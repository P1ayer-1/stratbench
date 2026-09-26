"""The BloFin fee ladder: hand-transcribed, so checked at import and here."""

from decimal import Decimal

import pytest

from perpkit import fees


def test_round_trips_at_vip0_by_hand():
    # VIP 0: maker 0.020%, taker 0.060% of notional per side.
    assert fees.round_trip_bps("maker", "maker", tier=0) == Decimal("4.0")
    assert fees.round_trip_bps("maker", "taker", tier=0) == Decimal("8.0")
    assert fees.round_trip_bps("taker", "taker", tier=0) == Decimal("12.0")


def test_the_top_tier_is_free_to_make_but_never_pays_a_rebate():
    maker, taker = fees.VIP_TIERS[5]
    assert maker == 0 and taker == Decimal("0.00035")
    assert all(m >= 0 for m, _ in fees.VIP_TIERS.values())


def test_an_unconfirmed_tier_is_an_error_not_a_guess():
    assert 4 not in fees.VIP_TIERS
    with pytest.raises(KeyError, match="known"):
        fees.round_trip_bps("taker", "taker", tier=4)


def test_a_role_that_is_neither_maker_nor_taker_is_refused():
    with pytest.raises(ValueError):
        fees.round_trip_bps("maker", "market", tier=0)


def test_the_ladder_check_catches_a_tier_that_gets_worse():
    bad = {0: (Decimal("0.0002"), Decimal("0.0006")),
           1: (Decimal("0.0003"), Decimal("0.0005"))}
    with pytest.raises(SystemExit, match="not monotonic"):
        fees.check_tier_ladder(bad)


def test_the_ladder_check_catches_two_tiers_on_identical_rates():
    bad = {0: (Decimal("0.0002"), Decimal("0.0006")),
           1: (Decimal("0.0002"), Decimal("0.0006"))}
    with pytest.raises(SystemExit, match="identical"):
        fees.check_tier_ladder(bad)


def test_every_row_is_unverified_until_a_fill_says_otherwise():
    assert fees.VERIFIED_AGAINST_A_FILL is False
    assert fees.FUTURES_TRANSCRIBED.startswith("2026-09-07")
