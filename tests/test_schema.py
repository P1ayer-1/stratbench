from decimal import Decimal

import pytest

from predkit.schema import Book, Contract, Level, OrderIntent, Outcome, Side, check_price, no_to_yes, utc

D = Decimal


def contract(**kw):
    args = dict(venue="kalshi", market_id="KXBTC15M-1", question="BTC up?", resolves_at=utc(2026, 9, 12, 14, 15),
                resolution_source="kalshi:KXBTC15M", fee_tier="default")
    args.update(kw)
    return Contract(**args)


def test_prices_are_probabilities():
    assert check_price("0.48") == D("0.48")
    for bad in ("-0.01", "1.01"):
        with pytest.raises(ValueError):
            check_price(bad)


def test_no_to_yes_is_its_own_inverse():
    assert no_to_yes("0.52") == D("0.48")
    assert no_to_yes(no_to_yes("0.37")) == D("0.37")


def test_a_naive_resolution_time_is_refused():
    from datetime import datetime
    with pytest.raises(ValueError):
        contract(resolves_at=datetime(2026, 9, 12, 14, 15))


def test_resolution_window_includes_after_resolution():
    c = contract()
    end = c.resolves_at_ms
    assert not c.in_resolution_window(end - 61_000, 60)
    assert c.in_resolution_window(end - 60_000, 60)
    assert c.in_resolution_window(end + 5_000, 60)


def test_an_intent_off_the_grid_or_under_minimum_cannot_exist():
    c = contract(venue="polymarket", min_size=D("5"))
    with pytest.raises(ValueError):
        OrderIntent(c, Side.BUY, D("0.485"), D("5"))
    with pytest.raises(ValueError):
        OrderIntent(c, Side.BUY, D("0.48"), D("4"))
    ok = OrderIntent(c, Side.BUY, D("0.48"), D("5"))
    assert ok.notional == D("2.40")


def test_yes_delta_signs():
    c = contract()
    assert OrderIntent(c, Side.BUY, D("0.5"), D("3"), Outcome.YES).yes_delta == D("3")
    assert OrderIntent(c, Side.BUY, D("0.5"), D("3"), Outcome.NO).yes_delta == D("-3")
    assert OrderIntent(c, Side.SELL, D("0.5"), D("3"), Outcome.NO).yes_delta == D("3")


def test_book_sorts_and_reports_the_touch():
    book = Book("M", bids=[Level("0.47", 10), Level("0.48", 5)], asks=[Level("0.51", 1), Level("0.50", 2)])
    assert (book.best_bid, book.best_ask, book.mid, book.spread) == (D("0.48"), D("0.50"), D("0.49"), D("0.02"))
    assert not book.is_crossed()
    assert book.size_at(Side.BUY, "0.47") == D("10")
