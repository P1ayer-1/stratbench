"""perpkit: record, replay and backtest crypto perpetual futures, with a risk veto.

Layers, in dependency order (each imports only from the ones above it):

    orderbook.py  L2 book maintained from BloFin's `books` channel
    tape.py       rolling window of aggressive trades
    features.py   order-book imbalance / order-flow / microprice / volatility features
    recorder.py   feature rows + forward labels written to disk, only after the horizon closes
    ingest.py     the live BloFin websocket loop wiring the above together (needs the SDK)
    risk.py       liquidation math, sizing, hard limits, kill switch (imports nothing above)

`risk.py` deliberately has no imports from this package. It must stay able to
veto anything without depending on it.

Importing this package does **not** import the BloFin SDK. `ingest` is loaded
lazily on first access (PEP 562) so that `risk`, `features` and `orderbook`
stay importable, and unit-testable, with no SDK, no network and no
credentials.
"""

from typing import TYPE_CHECKING, Any

from .features import FeatureEngine, FeatureSnapshot, feature_columns
from .orderbook import OrderBook
from .recorder import FeatureRecorder, LabelConfig
from .risk import (
    AccountState,
    RiskDecision,
    RiskEngine,
    RiskLimits,
    Side,
    SizingResult,
    liquidation_distance_pct,
    liquidation_price,
    max_safe_leverage,
)
from .tape import TradeTape

if TYPE_CHECKING:  # for type checkers and IDEs only, not executed at runtime
    from .ingest import MicrostructureFeed


def __getattr__(name: str) -> Any:
    """Lazily import the websocket feed, which needs the BloFin SDK."""
    if name == "MicrostructureFeed":
        from .ingest import MicrostructureFeed as _MicrostructureFeed

        return _MicrostructureFeed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AccountState",
    "FeatureEngine",
    "FeatureRecorder",
    "FeatureSnapshot",
    "LabelConfig",
    "MicrostructureFeed",
    "OrderBook",
    "RiskDecision",
    "RiskEngine",
    "RiskLimits",
    "Side",
    "SizingResult",
    "TradeTape",
    "feature_columns",
    "liquidation_distance_pct",
    "liquidation_price",
    "max_safe_leverage",
]
