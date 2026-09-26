"""Binance's public book ticker, as a REFERENCE feed. Never an account.

The 5- and 15-minute BTC markets resolve on the venue's own oracle, not on
Binance, so this is not a resolution source and `backtest.label_row` will
refuse it as one. It is the LEADER the maker template watches: when
Binance has already moved and the contract has not, the contract's stale
price is the opportunity. The feed is archived beside the market with its
receive time, because feed lag decides whether a lead-lag strategy pays at
all (an edge at 150 ms of lag can be gone at 500 ms) and it cannot be
reconstructed later.

    wss://stream.binance.com:9443/ws/<symbol>@bookTicker
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Tuple

WS = "wss://stream.binance.com:9443/ws"


class BinanceBookTicker:
    name = "binance"

    def __init__(self, ws_url: str = WS):
        self.ws_url = ws_url

    async def stream(self, symbol: str) -> AsyncIterator[Tuple[str, Any]]:
        import websockets

        url = f"{self.ws_url}/{symbol.lower()}@bookTicker"
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as socket:
            async for raw in socket:
                yield ("bookTicker", json.loads(raw))


__all__ = ["BinanceBookTicker"]
