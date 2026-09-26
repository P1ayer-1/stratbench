"""The one shape every venue is translated into.

Polymarket and Kalshi describe the same instrument in different words: a
Polymarket market has two ERC-1155 tokens (one per outcome) each with its own
order book, while a Kalshi market has one book with yes-bids and no-bids on
opposite sides. Both are a binary contract that pays $1 for one outcome and
$0 for the other, so everything above the venue adapters speaks in ONE
currency: the price of YES, between 0 and 1.

    a NO bid at q   is   a YES ask at 1 - q
    a NO ask at q   is   a YES bid at 1 - q

Adapters do that conversion once, in `venues/`, so the book, the fill model,
the fees, the risk veto and the strategies never see a NO price. A strategy
that wants NO exposure says so on the intent, and the adapter converts it
back. This is the boundary that keeps a sign error out of the backtest.

`resolves_at` and `resolution_source` are on the contract rather than on the
market listing because they decide two things nothing else may decide: the
risk veto refuses orders inside the resolution window, and a backtest row is
labelled only from the venue's own resolution source, never from a spot
exchange (see `backtest.label_row`).

Prices are `Decimal`. A price here is money compared against exchange values
and summed across venues to see whether it exceeds a dollar; 0.1 + 0.2 != 0.3
is not an argument to have with a cross-venue check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

ZERO = Decimal("0")
ONE = Decimal("1")
CENT = Decimal("0.01")

Number = Union[Decimal, int, str, float]


def to_decimal(value: Number) -> Decimal:
    """Money in, `Decimal` out. Floats go through `str` so 0.1 stays 0.1."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def check_price(value: Number) -> Decimal:
    """A contract price is a probability: 0 <= p <= 1, no exceptions."""
    price = to_decimal(value)
    if price < ZERO or price > ONE:
        raise ValueError(f"price {price} is outside [0, 1]; contracts are probabilities")
    return price


def no_to_yes(price: Number) -> Decimal:
    """The YES price a NO price implies, and vice versa. Its own inverse."""
    return ONE - check_price(price)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Outcome(str, Enum):
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class Contract:
    """One tradable binary contract, priced as YES.

    `fee_tier` names a row in `fees.TABLES` and is the venue's own tier name
    for this market family (Gamma's `feeType` on Polymarket, "default" on Kalshi). An unknown tier
    is an error at fee time, never a default: see fees.py for why.
    """

    venue: str
    market_id: str
    question: str
    resolves_at: datetime
    resolution_source: str
    fee_tier: str
    tick: Decimal = CENT
    min_size: Decimal = ONE
    yes_token: Optional[str] = None   # Polymarket: the YES token id the book is keyed by
    no_token: Optional[str] = None
    # Kalshi's "tapered_deci_cent" books step 0.001 below 10c and above 90c
    # and 0.01 between (read live 2026-09-13): (start, end, step) triples,
    # start inclusive. Empty means `tick` everywhere.
    price_ranges: Tuple[Tuple[Decimal, Decimal, Decimal], ...] = ()
    # The venue's raw fields this package does not interpret but must not
    # lose: rules text, raw fee fields, alternate timestamps.
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resolves_at.tzinfo is None:
            raise ValueError(f"{self.market_id}: resolves_at must be timezone-aware (UTC)")
        if self.tick <= ZERO or self.tick > ONE:
            raise ValueError(f"{self.market_id}: tick {self.tick} is not a price increment")
        if self.min_size <= ZERO:
            raise ValueError(f"{self.market_id}: min_size {self.min_size} must be positive")
        if not self.resolution_source:
            raise ValueError(f"{self.market_id}: resolution_source is required; "
                             "a label with no source cannot be checked")
        for start, end, step in self.price_ranges:
            if not (ZERO <= start < end <= ONE) or step <= ZERO:
                raise ValueError(f"{self.market_id}: bad price range {(start, end, step)}")

    def tick_at(self, price: Number) -> Decimal:
        """The increment in force at `price`."""
        price = check_price(price)
        for start, end, step in self.price_ranges:
            if start <= price < end or (price == ONE and end == ONE):
                return step
        return self.tick

    @property
    def resolves_at_ms(self) -> int:
        return int(self.resolves_at.timestamp() * 1000)

    def seconds_to_resolution(self, now_ms: int) -> float:
        return (self.resolves_at_ms - now_ms) / 1000.0

    def in_resolution_window(self, now_ms: int, window_s: float) -> bool:
        """True from `window_s` before resolution until resolution, and after.

        After resolution is included on purpose: a market that has resolved
        is not one to be sending orders to either.
        """
        return self.seconds_to_resolution(now_ms) <= window_s

    def on_grid(self, price: Number) -> bool:
        price = check_price(price)
        return (price / self.tick_at(price)) % 1 == 0


@dataclass(frozen=True)
class Level:
    price: Decimal
    size: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", check_price(self.price))
        object.__setattr__(self, "size", to_decimal(self.size))
        if self.size < ZERO:
            raise ValueError(f"level size {self.size} is negative")


@dataclass
class Book:
    """A YES-denominated order book. Bids descending, asks ascending."""

    market_id: str
    bids: List[Level] = field(default_factory=list)
    asks: List[Level] = field(default_factory=list)
    ts_ms: int = 0

    def __post_init__(self) -> None:
        self.bids = sorted(self.bids, key=lambda lvl: lvl.price, reverse=True)
        self.asks = sorted(self.asks, key=lambda lvl: lvl.price)

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[Decimal]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def is_crossed(self) -> bool:
        return (self.best_bid is not None and self.best_ask is not None
                and self.best_bid >= self.best_ask)

    def size_at(self, side: Side, price: Number) -> Decimal:
        price = check_price(price)
        levels = self.bids if side is Side.BUY else self.asks
        return sum((lvl.size for lvl in levels if lvl.price == price), ZERO)


@dataclass(frozen=True)
class Trade:
    market_id: str
    price: Decimal          # YES price
    size: Decimal
    aggressor: Side         # side of the taker, in YES terms
    ts_ms: int
    trade_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", check_price(self.price))
        object.__setattr__(self, "size", to_decimal(self.size))


@dataclass(frozen=True)
class OrderIntent:
    """What a strategy wants to send. Built by `plan`, sent by `execute`.

    `price` is the price of the OUTCOME named: buy YES at 0.48 pays 0.48 a
    contract, buy NO at 0.52 pays 0.52. `notional` is therefore always the
    dollars paid, and `yes_delta` is the signed YES-equivalent exposure, so
    the risk veto and the ledger never have to know which outcome it was. On the
    YES-denominated book a NO buy at q sits at the YES ask of 1 - q; the
    adapter does that conversion. Validated at construction so a plan cannot
    carry an unsendable order.
    """

    contract: Contract
    side: Side
    price: Decimal
    size: Decimal
    outcome: Outcome = Outcome.YES
    post_only: bool = True
    client_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", check_price(self.price))
        object.__setattr__(self, "size", to_decimal(self.size))
        if not self.contract.on_grid(self.price):
            raise ValueError(f"{self.contract.market_id}: price {self.price} is not on the "
                             f"{self.contract.tick} tick grid")
        if self.size < self.contract.min_size:
            raise ValueError(f"{self.contract.market_id}: size {self.size} is under the "
                             f"minimum {self.contract.min_size}")

    @property
    def notional(self) -> Decimal:
        return self.price * self.size

    @property
    def yes_delta(self) -> Decimal:
        """Signed YES-equivalent contracts this order adds when filled."""
        sign = ONE if self.side is Side.BUY else -ONE
        if self.outcome is Outcome.NO:
            sign = -sign
        return sign * self.size


@dataclass(frozen=True)
class Fill:
    """A fill as the VENUE reports it. `fill_id` is the venue's identifier and
    `tx_hash` the chain's (Polymarket settles on-chain). A fill that carries
    neither cannot be checked against the venue and should not be counted
    in a track record."""

    venue: str
    market_id: str
    order_id: str
    side: Side
    price: Decimal
    size: Decimal
    fee: Decimal
    ts_ms: int
    outcome: Outcome = Outcome.YES
    fill_id: str = ""
    tx_hash: Optional[str] = None
    builder_code: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", check_price(self.price))
        object.__setattr__(self, "size", to_decimal(self.size))
        object.__setattr__(self, "fee", to_decimal(self.fee))

    @property
    def notional(self) -> Decimal:
        return self.price * self.size

    @property
    def yes_delta(self) -> Decimal:
        sign = ONE if self.side is Side.BUY else -ONE
        if self.outcome is Outcome.NO:
            sign = -sign
        return sign * self.size


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0,
        second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


__all__ = [
    "Book", "CENT", "Contract", "Fill", "Level", "ONE", "OrderIntent", "Outcome",
    "Side", "Trade", "ZERO", "check_price", "no_to_yes", "to_decimal", "utc",
]
