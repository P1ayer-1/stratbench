"""predkit: record, replay, backtest and run bots for prediction markets.

Every position here is a binary contract priced between 0 and 1 that resolves
to exactly one of two values at a known time. That one fact shapes the whole
package: max loss is the price paid, there is no leverage, no liquidation and
no funding, and a track record can be checked against the venue's own fills.

The package imports nothing that talks to a network at import time. Venue
adapters take an injectable opener, the recorder takes an injectable stream,
and `risk.py` imports nothing from this package at all.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
