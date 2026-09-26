from predkit.geofence import refusals


def test_canada_is_refused_everywhere():
    assert refusals("kalshi", "CA") and refusals("polymarket", "ca")


def test_us_is_refused_on_polymarket_only():
    assert refusals("polymarket", "US")
    assert refusals("kalshi", "US") == []


def test_a_missing_residency_is_a_refusal_not_a_default():
    assert refusals("kalshi", "") and refusals("kalshi", "Canada")


def test_residencies_not_in_the_table_pass_this_check():
    """The table is a floor, not a whitelist: passing it is necessary for
    live, not sufficient. The venue's own terms and local law still apply."""
    assert refusals("polymarket", "DE") == [] and refusals("kalshi", "GB") == []
