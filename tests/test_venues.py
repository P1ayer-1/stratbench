"""Venue adapters against recorded payload shapes, never the network."""

import base64
import json
from decimal import Decimal

import httpx
import pytest

from predkit.schema import Contract, OrderIntent, Outcome, Side, utc
from predkit.venues import NotSignable, Venue
from predkit.venues.kalshi import Kalshi, RsaPssSigner, auth_headers, contract_from_market
from predkit.venues.polymarket import Polymarket, contract_from_gamma

D = Decimal

GAMMA_ROW = {
    "conditionId": "0xabc", "question": "Bitcoin Up or Down - September 12, 2:05PM-2:10PM ET",
    "slug": "btc-updown-5m-1757700300", "endDate": "2026-09-12T18:10:00Z",
    "clobTokenIds": json.dumps(["111", "222"]), "outcomes": json.dumps(["Up", "Down"]),
    "orderPriceMinTickSize": 0.01, "orderMinSize": 5,
    "resolutionSource": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
    "feeType": "crypto_fees_v2", "takerBaseFee": 1000, "makerBaseFee": 1000,
    "feeSchedule": {"exponent": 1, "rate": 0.07, "takerOnly": True, "rebateRate": 0.2},
}
CLOB_BOOK = {"bids": [{"price": "0.48", "size": "100"}, {"price": "0.47", "size": "50"}],
             "asks": [{"price": "0.52", "size": "80"}], "timestamp": "1757700000000"}
# Shapes read live 2026-09-13: dollar strings, fractional counts, tapered ticks.
KALSHI_MARKET = {"ticker": "KXBTC15M-26SEP1218-T112000", "series_ticker": "KXBTC15M",
                 "title": "BTC price up in next 15 mins?", "close_time": "2026-09-12T12:18:00Z",
                 "expiration_time": "2026-09-19T12:18:00Z", "price_level_structure": "tapered_deci_cent",
                 "price_ranges": [{"start": "0.0000", "end": "0.1000", "step": "0.0010"},
                                  {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
                                  {"start": "0.9000", "end": "1.0000", "step": "0.0010"}],
                 "rules_primary": "If the simple average of the sixty seconds of CF Benchmarks' BRTI..."}
KALSHI_BOOK = {"orderbook_fp": {"yes_dollars": [["0.4900", "100.00"], ["0.5000", "2522.00"]],
                                "no_dollars": [["0.4900", "2574.00"]]}}
KALSHI_TRADES = {"trades": [{"trade_id": "t1", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000",
                             "count_fp": "3.00", "taker_side": "no",
                             "created_time": "2026-09-12T12:00:00.974575Z"}]}
KALSHI_BOOK_LEGACY = {"orderbook": {"yes": [[50, 2522], [49, 100]], "no": [[49, 2574]]}}


def poly_client(routes):
    def handler(request):
        for prefix, payload in routes.items():
            if request.url.path.startswith(prefix):
                return httpx.Response(200, json=payload)
        return httpx.Response(404)
    return Polymarket(http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_gamma_row_becomes_a_contract_keyed_on_the_venues_fee_type():
    c = contract_from_gamma(GAMMA_ROW)
    assert c.fee_tier == "crypto_fees_v2"                 # Gamma's feeType, verbatim
    assert (c.yes_token, c.no_token) == ("111", "222")   # Up is the YES-like first outcome
    assert c.resolution_source == "https://data.chain.link/streams/btc-usd-twap-60s-streams"
    assert c.min_size == D("5")


def test_a_row_without_a_fee_type_gets_the_absent_tier():
    row = {k: v for k, v in GAMMA_ROW.items() if k != "feeType"}
    assert contract_from_gamma(row).fee_tier == "unknown"


def test_polymarket_book_is_yes_denominated_and_typed():
    client = poly_client({"/markets": [GAMMA_ROW], "/book": CLOB_BOOK})
    book = client.book("0xabc")
    assert (book.best_bid, book.best_ask) == (D("0.48"), D("0.52"))
    assert isinstance(client, Venue)


def test_polymarket_data_api_floats_are_quantised_to_the_tick():
    """0.2099999969 is a 21c print (read live 2026-09-13), not a print below 0.21."""
    client = poly_client({"/markets": [GAMMA_ROW],
                          "/trades": [{"side": "BUY", "asset": "111", "outcome": "Up", "price": 0.2099999969,
                                       "size": 22.666667, "timestamp": 1789264516, "transactionHash": "0xt"}]})
    (trade,) = client.trades("0xabc")
    assert (trade.price, trade.size, trade.trade_id) == (D("0.2100"), D("22.666667"), "0xt")


def test_gamma_raw_fee_fields_ride_along_on_extra():
    c = contract_from_gamma(GAMMA_ROW)
    assert c.extra["feeSchedule"]["rate"] == 0.07 and c.extra["takerBaseFee"] == 1000


def test_polymarket_refuses_to_send_without_a_signing_client():
    client = poly_client({"/markets": [GAMMA_ROW]})
    c = client.market("0xabc")
    with pytest.raises(NotSignable):
        client.place_order(OrderIntent(c, Side.BUY, D("0.48"), D("5")))


def kalshi_client(signer=None):
    def handler(request):
        path = request.url.path
        if path.endswith("/orderbook"):
            return httpx.Response(200, json=KALSHI_BOOK)
        if path.endswith("/markets/trades"):
            return httpx.Response(200, json=KALSHI_TRADES)
        if path.endswith("/markets"):
            return httpx.Response(200, json={"markets": [KALSHI_MARKET]})
        if "/markets/" in path:
            return httpx.Response(200, json={"market": KALSHI_MARKET})
        if path.endswith("/portfolio/orders") and request.method == "POST":
            assert request.headers["KALSHI-ACCESS-KEY"] == "kid"
            body = json.loads(request.content)
            assert body["post_only"] is True and body["yes_price_dollars"] == "0.5000"
            assert body["count_fp"] == "1.00"
            return httpx.Response(200, json={"order": {"order_id": "ord-1"}})
        return httpx.Response(404)
    return Kalshi(http=httpx.Client(transport=httpx.MockTransport(handler)), signer=signer)


def test_kalshi_book_converts_no_bids_to_yes_asks():
    book = kalshi_client().book("KXBTC15M-26SEP1218-T112000")
    assert (book.best_bid, book.best_ask) == (D("0.50"), D("0.51"))
    assert book.size_at(Side.SELL, "0.51") == D("2574")


def test_kalshi_trades_are_typed_with_the_taker_side():
    (trade,) = kalshi_client().trades("KXBTC15M-26SEP1218-T112000")
    assert (trade.price, trade.size, trade.aggressor) == (D("0.50"), D("3"), Side.SELL)


def test_kalshi_contract_resolves_at_close_time_with_tapered_ticks():
    c = contract_from_market(KALSHI_MARKET)
    assert c.resolution_source == "kalshi:KXBTC15M" and c.fee_tier == "default"
    assert c.resolves_at == utc(2026, 9, 12, 12, 18)          # close_time, not expiration_time
    assert c.tick_at("0.05") == D("0.001") and c.tick_at("0.50") == D("0.01") and c.tick_at("0.95") == D("0.001")
    assert c.on_grid("0.055") and not c.on_grid("0.555") and c.on_grid("0.999")
    assert "BRTI" in c.extra["rules_primary"]


def test_kalshi_legacy_cents_book_still_parses():
    from predkit.venues.kalshi import parse_orderbook
    book = parse_orderbook(KALSHI_BOOK_LEGACY, "M")
    assert (book.best_bid, book.best_ask) == (D("0.50"), D("0.51"))


def test_kalshi_refuses_to_write_without_a_signer():
    client = kalshi_client()
    c = client.market("KXBTC15M-26SEP1218-T112000")
    with pytest.raises(NotSignable):
        client.place_order(OrderIntent(c, Side.BUY, D("0.50"), D("1")))


@pytest.fixture
def rsa_signer():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    return RsaPssSigner("kid", pem), key.public_key()


def test_auth_headers_sign_timestamp_method_and_path(rsa_signer):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    signer, public = rsa_signer
    headers = auth_headers(signer, "post", "/trade-api/v2/portfolio/orders", now_ms=1757700000000)
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1757700000000"
    public.verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
                  b"1757700000000POST/trade-api/v2/portfolio/orders",
                  padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                  hashes.SHA256())


def test_kalshi_places_a_signed_post_only_limit_in_cents(rsa_signer):
    client = kalshi_client(signer=rsa_signer[0])
    c = client.market("KXBTC15M-26SEP1218-T112000")
    assert client.place_order(OrderIntent(c, Side.BUY, D("0.50"), D("1"), Outcome.YES)) == "ord-1"
