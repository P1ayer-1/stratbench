"""Fee values are hand-computed from the curve, not read off the code."""

from decimal import Decimal

import pytest

from predkit import fees
from predkit.fees import Curve, Tier

D = Decimal


def test_the_crypto_curve_peaks_at_one_point_seven_five_cents():
    # 0.07 * 0.5 * 0.5 = 0.0175 per contract
    assert fees.fee("polymarket", "crypto_fees_v2", "taker", "0.5", 1) == D("0.0175")
    # 0.07 * 0.9 * 0.1 = 0.0063
    assert fees.fee("polymarket", "crypto_fees_v2", "taker", "0.9", 1) == D("0.0063")
    assert fees.fee("polymarket", "crypto_fees_v2", "taker", "0", 1) == D("0")


def test_economics_markets_pay_five_cents_of_curve():
    # 0.05 * 0.5 * 0.5 = 0.0125
    assert fees.fee("polymarket", "economics_fees", "taker", "0.5", 1) == D("0.0125")
    assert fees.tier("polymarket", "economics_fees").rebate_rate == D("0.25")


def test_kalshi_rounds_the_order_fee_up_to_the_cent():
    # 0.07 * 0.5 * 0.5 * 3 = 0.0525 -> 0.06
    assert fees.fee("kalshi", "default", "taker", "0.5", 3) == D("0.06")
    # 0.07 * 0.1 * 0.9 * 1 = 0.0063 -> 0.01
    assert fees.fee("kalshi", "default", "taker", "0.1", 1) == D("0.01")


def test_kalshi_maker_fee_series_charge_the_resting_side():
    """Read live 2026-09-24: KXCPI, KXCPIYOY, KXLLM1, KXSUPERBOWLHEADLINE carry `fee_type`
    quadratic_with_maker_fees. Guards a maker-fee series being scored as free (the default
    tier's maker curve is zero) and the maker coefficient drifting from the PDF's 0.0175."""
    # 0.0175 * 0.5 * 0.5 * 100 = 0.4375 -> ceil to the cent -> 0.44
    assert fees.fee("kalshi", "quadratic_with_maker_fees", "maker", "0.5", 100) == D("0.44")
    # 0.0175 * 0.1 * 0.9 * 1 = 0.001575 -> 0.01
    assert fees.fee("kalshi", "quadratic_with_maker_fees", "maker", "0.1", 1) == D("0.01")
    # taker side is the standard 0.07 curve: 0.07 * 0.5 * 0.5 * 100 = 1.75
    assert fees.fee("kalshi", "quadratic_with_maker_fees", "taker", "0.5", 100) == D("1.75")
    # the exact per-contract curve before rounding: 0.0175 * 0.25 = 0.004375
    assert fees.tier("kalshi", "quadratic_with_maker_fees").maker.per_contract(D("0.5")) == D("0.004375")


def test_kalshi_quadratic_mention_series_taker_fee_hand_values():
    """Read live 2026-09-25: the mention series' `fee_type` is `quadratic`. Guards the
    series being scored on an absent tier (KeyError at fee time) or without the per-order cent
    round-up, which is easy to forget when fees are computed per contract."""
    # 0.07 * 10 * 0.5 * 0.5 = 0.175 -> 0.18
    assert fees.fee("kalshi", "quadratic", "taker", "0.50", 10) == D("0.18")
    # 0.07 * 100 * 0.85 * 0.15 = 0.8925 -> 0.90
    assert fees.fee("kalshi", "quadratic", "taker", "0.85", 100) == D("0.90")
    assert fees.fee("kalshi", "quadratic", "maker", "0.50", 100) == D("0")


def test_makers_pay_nothing_on_transcribed_tiers():
    for venue, tier in (("kalshi", "default"), ("polymarket", "crypto_fees_v2"), ("polymarket", "economics_fees")):
        assert fees.fee(venue, tier, "maker", "0.5", 100) == D("0")


def test_nothing_is_verified_against_a_fill_yet():
    assert all(row.verified is False for row in fees.TABLES.values())


def test_an_unknown_tier_is_an_error_that_names_the_known_ones():
    """A Gamma feeType this file has not transcribed must fail by name."""
    with pytest.raises(KeyError) as excinfo:
        fees.fee("polymarket", "sports_fees", "taker", "0.5", 5)
    assert "crypto_fees_v2" in str(excinfo.value) and "economics_fees" in str(excinfo.value)
    with pytest.raises(KeyError):
        fees.fee("polymarket", "unknown", "taker", "0.5", 5)


def test_builder_fee_is_on_notional_and_capped():
    # 0.48 * 100 * 25 bps = 0.12
    assert fees.builder_fee("maker", "0.48", 100, maker_bps=D("25")) == D("0.12")
    with pytest.raises(ValueError):
        fees.builder_fee("maker", "0.5", 1, maker_bps=D("51"))
    with pytest.raises(ValueError):
        fees.builder_fee("taker", "0.5", 1, taker_bps=D("101"))
    assert fees.builder_fee("taker", "0.5", 1) == D("0")     # default is zero


def test_the_import_check_rejects_a_maker_fee_above_taker(monkeypatch):
    bad = Tier("kalshi", "bad", maker=Curve(D("0.08")), taker=Curve(D("0.07")), dated="", source="", verified=False)
    monkeypatch.setitem(fees.TABLES, ("kalshi", "bad"), bad)
    with pytest.raises(SystemExit):
        fees._check_tables()


def test_the_import_check_rejects_two_tiers_on_identical_curves(monkeypatch):
    dup = Tier("polymarket", "dup", maker=fees.NO_FEE, taker=fees.STANDARD, dated="", source="", verified=False)
    monkeypatch.setitem(fees.TABLES, ("polymarket", "dup"), dup)
    with pytest.raises(SystemExit):
        fees._check_tables()
