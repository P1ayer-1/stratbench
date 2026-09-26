"""The Hyperliquid liquidation map, checked against hand-built positions.

Every expected number here is worked out in the test, not read back from the
code. The ones that earn their place are about honesty rather than arithmetic:
a position with no liquidation price must be counted rather than dropped, a
crossed one must not land in a band, and coverage must be reported against
open interest so a sample cannot pass for the market.
"""

import pytest

from perpkit.liquidation_map import (
    TrackedPosition,
    build_map,
    positions_from_state,
)


def position(user, size, liq, *, coin="BTC", observed_ms=1_000_000,
             value=None, lev_type="cross"):
    return TrackedPosition(
        user=user, coin=coin, size=size, entry_px=100.0, liquidation_px=liq,
        position_value=value if value is not None else abs(size) * 100.0,
        leverage_type=lev_type, leverage=10.0, observed_ms=observed_ms,
    )


def state(*positions, time=1_789_151_132_727):
    return {
        "assetPositions": [{"type": "oneWay", "position": p} for p in positions],
        "time": time,
    }


def raw(coin, szi, liq, *, value="1000", lev_type="cross", lev=10):
    return {"coin": coin, "szi": szi, "entryPx": "100", "liquidationPx": liq,
            "positionValue": value, "leverage": {"type": lev_type, "value": lev}}


# ---------------------------------------------------------------------------
# Parsing the exchange's response
# ---------------------------------------------------------------------------


def test_a_null_liquidation_price_survives_parsing_as_none():
    """Measured: over half of sampled positions report null. They are real
    positions and must reach the map, where they are counted."""
    parsed = positions_from_state(
        "0xa", state(raw("BTC", "12.13623", None)), observed_ms=5)
    assert len(parsed) == 1
    assert parsed[0].liquidation_px is None
    assert parsed[0].size == pytest.approx(12.13623)


def test_short_size_keeps_its_sign():
    parsed = positions_from_state(
        "0xa", state(raw("BTC", "-0.4004", "209806.4493876494")), observed_ms=5)
    assert parsed[0].size == pytest.approx(-0.4004)
    assert not parsed[0].is_long
    assert parsed[0].liquidation_px == pytest.approx(209806.4493876494)


def test_coins_filter_restricts_the_parse():
    parsed = positions_from_state(
        "0xa", state(raw("BTC", "1", "90"), raw("ETH", "-2", "3000")),
        observed_ms=5, coins={"ETH"})
    assert [p.coin for p in parsed] == ["ETH"]


def test_a_zero_size_row_is_not_a_position():
    parsed = positions_from_state("0xa", state(raw("BTC", "0.0", "90")),
                                  observed_ms=5)
    assert parsed == []


def test_leverage_type_is_kept():
    parsed = positions_from_state(
        "0xa", state(raw("SOL", "5", "80", lev_type="isolated", lev=3)),
        observed_ms=5)
    assert parsed[0].leverage_type == "isolated"
    assert parsed[0].leverage == 3.0


def test_malformed_rows_are_skipped_not_raised():
    parsed = positions_from_state(
        "0xa", {"assetPositions": [None, {}, {"position": {"coin": "BTC"}}]},
        observed_ms=5)
    assert parsed == []


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------


def test_longs_band_below_the_mark_and_shorts_above():
    # Mark 100, 1% bands. Long liq at 98.5 is 1.5% away -> band [1%, 2%).
    # Short liq at 103.2 is 3.2% away -> band [3%, 4%).
    result = build_map(
        "BTC",
        [position("0xa", 2.0, 98.5), position("0xb", -3.0, 103.2)],
        mark_px=100.0, as_of_ms=1_000_000, bucket_pct=0.01,
    )
    (long_band,) = result.longs.levels
    assert long_band.distance_low == pytest.approx(0.01)
    assert long_band.price_near == pytest.approx(99.0)
    assert long_band.price_far == pytest.approx(98.0)
    assert long_band.size == pytest.approx(2.0)
    assert long_band.notional == pytest.approx(2.0 * 98.5)

    (short_band,) = result.shorts.levels
    assert short_band.distance_low == pytest.approx(0.03)
    assert short_band.price_near == pytest.approx(103.0)
    assert short_band.price_far == pytest.approx(104.0)
    assert short_band.notional == pytest.approx(3.0 * 103.2)


def test_levels_are_ordered_nearest_first():
    result = build_map(
        "BTC",
        [position("0xa", 1.0, 90.0), position("0xb", 1.0, 99.5),
         position("0xc", 1.0, 95.0)],
        mark_px=100.0, as_of_ms=0, bucket_pct=0.01,
    )
    assert [level.price_near for level in result.longs.levels] == \
        pytest.approx([100.0, 95.0, 90.0])


def test_positions_in_one_band_aggregate():
    result = build_map(
        "BTC",
        [position("0xa", 1.0, 98.9), position("0xb", 4.0, 98.1)],
        mark_px=100.0, as_of_ms=0, bucket_pct=0.01,
    )
    (band,) = result.longs.levels
    assert band.positions == 2
    assert band.size == pytest.approx(5.0)
    assert band.notional == pytest.approx(98.9 + 4 * 98.1)


def test_no_liquidation_price_is_counted_not_banded():
    result = build_map(
        "BTC",
        [position("0xa", 12.0, None), position("0xb", 1.0, 95.0)],
        mark_px=100.0, as_of_ms=0, bucket_pct=0.01,
    )
    assert result.longs.size == pytest.approx(13.0)
    assert result.longs.no_liquidation_size == pytest.approx(12.0)
    assert sum(level.size for level in result.longs.levels) == pytest.approx(1.0)


def test_a_crossed_liquidation_price_is_in_no_band():
    """A long liquidating ABOVE the mark is a stale read or a liquidation in
    progress. Banding it at distance zero would invent a cluster at the mark."""
    result = build_map(
        "BTC",
        [position("0xa", 1.0, 101.0), position("0xb", -1.0, 99.0)],
        mark_px=100.0, as_of_ms=0, bucket_pct=0.01,
    )
    assert result.longs.levels == [] and result.shorts.levels == []
    assert result.longs.crossed_size == pytest.approx(1.0)
    assert result.shorts.crossed_size == pytest.approx(1.0)


def test_far_positions_go_to_the_tail_not_a_thousand_bands():
    # Measured on BTC: a short liquidating at 209,806 against a 77,143 mark.
    result = build_map(
        "BTC", [position("0xa", -0.4004, 209806.45)],
        mark_px=77143.5, as_of_ms=0, bucket_pct=0.005, max_distance_pct=0.25,
    )
    assert result.shorts.levels == []
    assert result.shorts.beyond_size == pytest.approx(0.4004)
    assert result.shorts.beyond_notional == pytest.approx(0.4004 * 209806.45)


def test_within_is_exact_rather_than_band_rounded():
    result = build_map(
        "BTC",
        [position("0xa", 1.0, 99.2), position("0xb", 2.0, 98.5),
         position("0xc", 3.0, 93.0)],
        mark_px=100.0, as_of_ms=0, bucket_pct=0.05, within_pct=(0.01, 0.02, 0.10),
    )
    assert result.longs.within[0.01] == pytest.approx(99.2)
    assert result.longs.within[0.02] == pytest.approx(99.2 + 2 * 98.5)
    assert result.longs.within[0.10] == pytest.approx(99.2 + 2 * 98.5 + 3 * 93)


def test_other_coins_are_ignored():
    result = build_map(
        "BTC", [position("0xa", 1.0, 95.0, coin="ETH")],
        mark_px=100.0, as_of_ms=0,
    )
    assert result.accounts == 0
    assert result.longs.size == 0


# ---------------------------------------------------------------------------
# What the map says about itself
# ---------------------------------------------------------------------------


def test_coverage_is_tracked_size_over_open_interest():
    result = build_map(
        "BTC",
        [position("0xa", 30.0, 95.0), position("0xb", -10.0, 105.0)],
        mark_px=100.0, as_of_ms=0, open_interest=200.0,
    )
    assert result.coverage() == {"long": pytest.approx(0.15),
                                 "short": pytest.approx(0.05)}


def test_coverage_is_unknown_without_open_interest():
    result = build_map("BTC", [position("0xa", 1.0, 95.0)],
                       mark_px=100.0, as_of_ms=0)
    assert result.coverage() == {"long": None, "short": None}


def test_age_is_weighted_by_notional():
    # $1,000 read 10s ago and $3,000 read 2s ago -> (10*1000 + 2*3000)/4000 = 4s.
    result = build_map(
        "BTC",
        [position("0xa", 1.0, 95.0, observed_ms=90_000, value=1000.0),
         position("0xb", 1.0, 95.0, observed_ms=98_000, value=3000.0)],
        mark_px=100.0, as_of_ms=100_000,
    )
    assert result.weighted_age_s == pytest.approx(4.0)


def test_an_account_on_both_sides_counts_once_in_the_total():
    result = build_map(
        "BTC",
        [position("0xa", 1.0, 95.0), position("0xa", -1.0, 105.0)],
        mark_px=100.0, as_of_ms=0,
    )
    assert result.accounts == 1
    assert result.longs.accounts == 1 and result.shorts.accounts == 1


def test_to_dict_is_json_serialisable():
    import json

    result = build_map(
        "BTC",
        [position("0xa", 1.0, 95.0), position("0xb", -1.0, None)],
        mark_px=100.0, as_of_ms=0, open_interest=10.0,
    )
    decoded = json.loads(json.dumps(result.to_dict()))
    assert decoded["coverage"]["long"] == pytest.approx(0.1)
    assert decoded["shorts"]["no_liquidation_size"] == pytest.approx(1.0)


def test_a_non_positive_mark_is_refused():
    with pytest.raises(ValueError):
        build_map("BTC", [], mark_px=0.0, as_of_ms=0)
