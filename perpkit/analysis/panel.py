"""The daily panel schema every venue's builder emits, and coin-name matching.

`panel_blofin.py` and `panel_hyperliquid.py` (and any builder you add for
another venue) write the same CSV columns, so `factor_panel.py` reads any of
them unchanged. Two rules hold across all builders or cross-venue comparisons
are meaningless:

* a row closes at 00:00 UTC;
* funding for day D is what ACCRUED during day D: the settlement stamped
  00:00 on D+1 belongs to D, rounded to its nominal minute first because
  settlements print milliseconds late.

A daily funding TOTAL is the comparable unit, never a per-settlement rate:
some venues fund hourly and others every eight hours.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

DAILY_COLUMNS = [
    "date", "symbol", "open", "high", "low", "close",
    "quote_volume", "trades", "taker_buy_frac", "minutes", "rv_bps",
]
PANEL_COLUMNS = DAILY_COLUMNS + ["funding_bps", "funding_periods"]

QUOTES = ("USDT", "USDC", "USD")
# Contract multipliers. A perp on 1000 SHIB and one on 1 SHIB are the same
# underlying and quote the same funding RATE (the multiplier changes the size
# of a contract, not the percentage paid), so matching across them is correct.
MULTIPLIERS = ("1000000", "10000", "1000", "100", "10", "1M")


def write_panel(rows: Sequence[Dict[str, object]], path: Path) -> None:
    """Write rows in the shared schema; missing fields are left blank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PANEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PANEL_COLUMNS})


def canonical(symbol: str) -> str:
    """`1000BONKUSDT` and `kBONK` both become `BONK`; `MOVEUSDT` stays `MOVE`.

    Venues name the same coin differently: `<BASE><QUOTE>` on most CEXs, the
    bare base on Hyperliquid, and each picks its own contract multiplier.
    Matching on the raw string silently drops every multiplied contract,
    which is most of the meme perps, where funding differs most, so the loss
    would not be random.

    Stripping too much is the opposite failure and a worse one: two different
    coins become one base. So a prefix goes only when it cannot be the start
    of a real ticker. The `k` must be lower-case followed by an upper-case
    letter, which only Hyperliquid's multiplier is. A numeric prefix must be
    followed by a letter, so `10000SATS` is not read as `1000` + `0SATS`, and
    digit-led tickers such as `1INCH` and `0G` pass through whole.
    """
    name = symbol.upper()
    for quote in QUOTES:
        if name.endswith(quote) and len(name) > len(quote):
            name = name[:-len(quote)]
            break
    if symbol[:1] == "k" and symbol[1:2].isupper() and len(name) > 1:
        return name[1:]
    for multiplier in MULTIPLIERS:
        rest = name[len(multiplier):]
        if name.startswith(multiplier) and rest[:1].isalpha():
            return rest
    return name
